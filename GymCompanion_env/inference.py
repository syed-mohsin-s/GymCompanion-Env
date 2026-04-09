import os
import json
import asyncio
from openai import OpenAI

try:
    from GymCompanion_env.client import GymcompanionEnv
    from GymCompanion_env.models import GymcompanionAction, WorkoutCategory, TargetMuscle
except ImportError:
    from client import GymcompanionEnv
    from models import GymcompanionAction, WorkoutCategory, TargetMuscle

# ── Environment variables (injected by the validator) ──────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")      # Optional – used as fallback key
# Validator injects API_KEY; fall back to HF_TOKEN only when API_KEY is absent
API_KEY      = os.environ.get("API_KEY") or HF_TOKEN

BENCHMARK  = "GymCompanion-Env"
MAX_STEPS  = 30
TASKS      = ["couch-to-5k", "plateau-breaker", "injury-rehab"]


# ── Structured log helpers (must match exactly) ────────────────────────────────
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_str = str(error) if error is not None else "null"
    done_str  = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── Main inference loop ────────────────────────────────────────────────────────
async def run_inference() -> None:
    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN or API_KEY environment variable.", flush=True)
        return

    # Synchronous OpenAI client routed through the validator's LiteLLM proxy
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_url = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000").replace("localhost", "127.0.0.1")

    system_prompt = (
        "You are an AI Personal Trainer. You receive JSON data about a client's physiology. "
        "Output ONLY a JSON object (no markdown fences) with these exact keys:\n"
        '{"workout_category": "rest"|"liss_cardio"|"hiit"|"hypertrophy"|"strength", '
        '"target_muscle": "none"|"legs"|"push"|"pull"|"full_body", '
        '"intensity_rpe": <integer 1-10>}'
    )

    for task_name in TASKS:
        log_start(task=task_name, model=MODEL_NAME)

        rewards: list[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with GymcompanionEnv(base_url=env_url) as env:
                    result = await env.reset(task_name=task_name)
                    messages = [{"role": "system", "content": system_prompt}]

                    for step in range(1, MAX_STEPS + 1):
                        if result.done:
                            break

                        obs_dict = {
                            "fitness_capacity": result.observation.fitness_capacity,
                            "cns_fatigue":      result.observation.cns_fatigue,
                            "muscle_soreness":  result.observation.muscle_soreness,
                            "days_active":      result.observation.days_active,
                        }
                        messages.append({"role": "user", "content": json.dumps(obs_dict)})

                        action_str = ""
                        error_val  = None
                        try:
                            response   = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                temperature=0.2,
                            )
                            llm_output = response.choices[0].message.content or ""
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
                            action      = GymcompanionAction(**action_data)
                            action_str  = json.dumps(action_data)

                        except Exception as llm_err:
                            print(f"[LLM ERROR] step={step}: {llm_err}", flush=True)
                            action     = GymcompanionAction(
                                workout_category=WorkoutCategory.REST,
                                target_muscle=TargetMuscle.NONE,
                                intensity_rpe=1,
                            )
                            action_str = '{"workout_category":"rest","target_muscle":"none","intensity_rpe":1}'
                            error_val  = llm_err

                        result      = await env.step(action)
                        reward      = result.reward or 0.0
                        done        = result.done
                        rewards.append(reward)
                        steps_taken = step

                        log_step(step=step, action=action_str, reward=reward, done=done, error=error_val)

                        if done:
                            break

                    # Score and success
                    meta        = result.observation.metadata or {}
                    score       = float(meta.get("score", 0.0))
                    success     = score > 0.0

                    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
                    break  # task complete – exit retry loop

            except Exception as conn_err:
                print(f"[RETRY] task={task_name} attempt={attempt+1}/{max_retries}: {conn_err}", flush=True)
                if attempt == max_retries - 1:
                    print(f"[SKIP] task={task_name} – max retries reached.", flush=True)
                    log_end(success=False, steps=0, score=0.0, rewards=[])
                else:
                    await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(run_inference())
