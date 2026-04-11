import os
import time
import json
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI

try:
    from GymCompanion_env.client import GymcompanionEnv
    from GymCompanion_env.models import GymcompanionAction, WorkoutCategory, TargetMuscle
except ImportError:
    from client import GymcompanionEnv
    from models import GymcompanionAction, WorkoutCategory, TargetMuscle

# ==========================================
# STRICT HACKATHON MANDATORY VARIABLES
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "GymCompanion-Env"
MAX_STEPS = 30
TASKS = ["couch-to-5k", "plateau-breaker", "injury-rehab"]


# ==========================================
# STRICT LOGGING FUNCTIONS
# ==========================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ==========================================
# ENV URL DISCOVERY WITH RETRIES
# ==========================================
def get_env_url(max_retries=15, delay=2):
    urls_to_try = [
        os.getenv("ENV_BASE_URL"),
        os.getenv("OPENENV_BASE_URL"),
        "http://server:8000",
        "http://environment:8000",
        "http://app:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "https://syedmohsin7-gymcompanion-env.hf.space",
    ]
    for attempt in range(max_retries):
        for url in urls_to_try:
            if not url:
                continue
            url = url.rstrip("/")
            try:
                req = urllib.request.Request(f"{url}/schema", method="GET")
                with urllib.request.urlopen(req, timeout=2) as response:
                    return url
            except urllib.error.HTTPError:
                return url
            except Exception:
                pass
        print(f"[ENV] Discovery attempt {attempt + 1}/{max_retries} - no server found, retrying...", flush=True)
        time.sleep(delay)
    return "http://localhost:8000"


# ==========================================
# MAIN INFERENCE
# ==========================================
def run_inference():
    ENV_URL = get_env_url()
    print(f"[DEBUG] Using ENV_URL={ENV_URL}", flush=True)
    print(f"[DEBUG] Using API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] Using MODEL_NAME={MODEL_NAME}", flush=True)

    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    system_prompt = (
        "You are an AI Personal Trainer. You receive JSON data about a client's "
        "physiology. You must output ONLY a valid JSON object with exactly three keys: "
        "'workout_category' (must be 'rest', 'liss_cardio', 'hiit', 'hypertrophy', or 'strength'), "
        "'target_muscle' (must be 'none', 'legs', 'push', 'pull', or 'full_body'), "
        "and 'intensity_rpe' (integer 1-10). Prioritize recovery if fatigue or soreness is high."
    )

    try:
        with GymcompanionEnv(base_url=ENV_URL).sync() as env:
            for task_name in TASKS:
                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

                steps_taken = 0
                rewards: List[float] = []
                success = False

                try:
                    result = env.reset(task_name=task_name)
                    done = False
                    messages = [{"role": "system", "content": system_prompt}]

                    while not done and steps_taken < MAX_STEPS:
                        obs = result.observation
                        obs_dict = {
                            "fitness_capacity": obs.fitness_capacity,
                            "cns_fatigue": obs.cns_fatigue,
                            "muscle_soreness": obs.muscle_soreness,
                            "days_active": obs.days_active,
                        }
                        messages.append({"role": "user", "content": json.dumps(obs_dict)})

                        action_str = ""
                        error_msg = None

                        try:
                            response = client.chat.completions.create(
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
                            action = GymcompanionAction(**action_data)
                            action_str = json.dumps(action_data)

                        except Exception as exc:
                            action = GymcompanionAction(
                                workout_category=WorkoutCategory.REST,
                                target_muscle=TargetMuscle.NONE,
                                intensity_rpe=1,
                            )
                            action_str = '{"workout_category":"rest","target_muscle":"none","intensity_rpe":1}'
                            error_msg = f"LLM Error: {str(exc)}"

                        result = env.step(action)

                        reward = float(result.reward or 0.0)
                        rewards.append(reward)
                        steps_taken += 1
                        done = result.done

                        log_step(
                            step=steps_taken,
                            action=action_str,
                            reward=reward,
                            done=done,
                            error=error_msg,
                        )

                    # Calculate final score
                    raw_score = sum(rewards) / len(rewards) if rewards else 0.0
                    final_score = max(0.0, min(1.0, raw_score))
                    success = final_score > 0.0

                    meta = result.observation.metadata or {}
                    if "score" in meta:
                        final_score = float(meta["score"])
                        success = final_score > 0.0

                    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

                except Exception as e:
                    log_end(success=False, steps=steps_taken, score=0.0, rewards=[0.0])

    except Exception as env_error:
        print(f"[FATAL] Environment error: {env_error}", flush=True)


if __name__ == "__main__":
    run_inference()
