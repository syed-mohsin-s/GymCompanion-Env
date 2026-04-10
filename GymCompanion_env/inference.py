import os
import json
import asyncio
import sys
from openai import OpenAI

try:
    from GymCompanion_env.client import GymcompanionEnv
    from GymCompanion_env.models import GymcompanionAction, WorkoutCategory, TargetMuscle
except ImportError:
    from client import GymcompanionEnv
    from models import GymcompanionAction, WorkoutCategory, TargetMuscle

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

BENCHMARK = "GymCompanion-Env"
MAX_STEPS = 30
TASKS = ["couch-to-5k", "plateau-breaker", "injury-rehab"]


# ── Structured log helpers ─────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={error if error is not None else 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Connect to environment with retries ────────────────────────────────────────
async def connect_env(max_retries=10, delay=5):
    """Try to connect to the environment, retrying on failure."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if LOCAL_IMAGE_NAME:
                print(f"[ENV] from_docker_image({LOCAL_IMAGE_NAME}) attempt {attempt}", flush=True)
                env = await GymcompanionEnv.from_docker_image(LOCAL_IMAGE_NAME)
            else:
                env_url = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")
                env_url = env_url.replace("localhost", "127.0.0.1")
                print(f"[ENV] Connecting to {env_url} attempt {attempt}", flush=True)
                env = GymcompanionEnv(base_url=env_url)
                await env.connect()
            print(f"[ENV] Connected on attempt {attempt}", flush=True)
            return env
        except Exception as e:
            last_err = e
            print(f"[ENV] Attempt {attempt}/{max_retries} failed: {e}", flush=True)
            if attempt < max_retries:
                await asyncio.sleep(delay)
    raise ConnectionError(f"Failed after {max_retries} attempts: {last_err}")


# ── Main ───────────────────────────────────────────────────────────────────────
async def main() -> None:
    # Resolve API key: prefer API_KEY (validator proxy), fall back to HF_TOKEN
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

    # ── Diagnostic logging ─────────────────────────────────────────────────────
    print(f"[DEBUG] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME   = {MODEL_NAME}", flush=True)
    print(f"[DEBUG] HF_TOKEN     = {'set (' + HF_TOKEN[:8] + '...)' if HF_TOKEN else 'NOT SET'}", flush=True)
    print(f"[DEBUG] API_KEY env   = {'set' if os.environ.get('API_KEY') else 'NOT SET'}", flush=True)
    print(f"[DEBUG] api_key used  = {'set (' + api_key[:8] + '...)' if api_key else 'NONE'}", flush=True)
    print(f"[DEBUG] LOCAL_IMAGE   = {LOCAL_IMAGE_NAME or 'NOT SET'}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL  = {os.environ.get('ENV_BASE_URL', 'NOT SET')}", flush=True)
    sys.stdout.flush()
    # ───────────────────────────────────────────────────────────────────────────

    if not api_key:
        print("ERROR: No API key. Set API_KEY or HF_TOKEN.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    system_prompt = (
        "You are an AI Personal Trainer. You receive JSON data about a client's "
        "physiology. Output ONLY a JSON object (no markdown fences) with these "
        "exact keys:\n"
        '{"workout_category": "rest"|"liss_cardio"|"hiit"|"hypertrophy"|"strength", '
        '"target_muscle": "none"|"legs"|"push"|"pull"|"full_body", '
        '"intensity_rpe": <integer 1-10>}'
    )

    env = None
    try:
        env = await connect_env()

        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            rewards = []
            steps_taken = 0
            score = 0.0
            success = False

            try:
                result = await env.reset(task_name=task_name)
                print(f"[DEBUG] After reset: done={result.done}", flush=True)
                messages = [{"role": "system", "content": system_prompt}]

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    obs_dict = {
                        "fitness_capacity": result.observation.fitness_capacity,
                        "cns_fatigue": result.observation.cns_fatigue,
                        "muscle_soreness": result.observation.muscle_soreness,
                        "days_active": result.observation.days_active,
                    }
                    messages.append({"role": "user", "content": json.dumps(obs_dict)})

                    action_str = ""
                    error = None
                    try:
                        print(f"[DEBUG] LLM call step={step} base_url={API_BASE_URL}", flush=True)
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=0.2,
                        )
                        llm_output = response.choices[0].message.content or ""
                        print(f"[DEBUG] LLM response received, len={len(llm_output)}", flush=True)
                        messages.append({"role": "assistant", "content": llm_output})

                        # Strip markdown fences if model wraps output
                        clean = llm_output.strip()
                        for fence in ("```json", "```"):
                            if clean.startswith(fence):
                                clean = clean[len(fence):]
                        if clean.endswith("```"):
                            clean = clean[:-3]
                        clean = clean.strip()

                        action_data = json.loads(clean)
                        action = GymcompanionAction(**action_data)
                        action_str = json.dumps(action_data)

                    except Exception as e:
                        print(f"[LLM ERROR] step={step}: {type(e).__name__}: {e}", flush=True)
                        action = GymcompanionAction(
                            workout_category=WorkoutCategory.REST,
                            target_muscle=TargetMuscle.NONE,
                            intensity_rpe=1,
                        )
                        action_str = '{"workout_category":"rest","target_muscle":"none","intensity_rpe":1}'
                        error = e

                    result = await env.step(action)
                    reward = result.reward or 0.0
                    done = result.done
                    rewards.append(reward)
                    steps_taken = step

                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    if done:
                        break

                meta = result.observation.metadata or {}
                score = float(meta.get("score", 0.0))
                success = score > 0.0

            except Exception as task_err:
                print(f"[TASK ERROR] task={task_name}: {type(task_err).__name__}: {task_err}", flush=True)

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as fatal_err:
        print(f"[FATAL] {type(fatal_err).__name__}: {fatal_err}", flush=True)
        # Emit logs for tasks so validator sees structured output
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
