import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI

try:
    from client import GymcompanionEnv
    from models import GymcompanionAction, WorkoutCategory, TargetMuscle
except ImportError:
    from GymCompanion_env.client import GymcompanionEnv
    from GymCompanion_env.models import GymcompanionAction, WorkoutCategory, TargetMuscle

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
TASKS = ["couch-to-5k", "plateau-breaker", "injury-rehab", "competition-prep", "overtraining-recovery"]


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
# TASK-SPECIFIC SYSTEM PROMPTS
# ==========================================
SYSTEM_PROMPTS = {
    "couch-to-5k": (
        "You are an AI Personal Trainer. Your client is SEDENTARY (fitness=20) and you must build them to fitness≥55. "
        "You receive JSON with fitness_capacity, cns_fatigue, muscle_soreness (legs/push/pull), days_active, goal_progress, and stress_event. "
        "Strategy: Start with LISS cardio at low RPE (4-5), gradually introduce hypertrophy at RPE 6-7. "
        "Always REST when cns_fatigue > 0.6, soreness > 0.6, or stress_event=true with cns > 0.4. "
        "Alternate muscle groups for periodization bonus. "
        "Output ONLY valid JSON: 'workout_category' (rest/liss_cardio/hiit/hypertrophy/strength), "
        "'target_muscle' (none/legs/push/pull/full_body), 'intensity_rpe' (1-10). No other text."
    ),
    "plateau-breaker": (
        "You are an AI Personal Trainer. Your client is PLATEAUED (fitness=60) and you must push them to fitness≥80. "
        "You receive JSON with fitness_capacity, cns_fatigue, muscle_soreness, goal_progress, and stress_event. "
        "Strategy: First REST if cns_fatigue > 0.5 or stress_event=true. "
        "Periodize: rotate strength (RPE 8) → hypertrophy (RPE 7) → HIIT to break adaptation. "
        "Rotate muscle groups (legs → push → pull) for periodization bonuses. Growth zone is RPE 6-8. "
        "Output ONLY valid JSON: 'workout_category' (rest/liss_cardio/hiit/hypertrophy/strength), "
        "'target_muscle' (none/legs/push/pull/full_body), 'intensity_rpe' (1-10). No other text."
    ),
    "injury-rehab": (
        "You are an AI Personal Trainer. Your client has a SEVERE LEG INJURY (legs soreness=0.85). "
        "CRITICAL: Training legs when soreness > 0.75 causes re-injury and IMMEDIATELY ends the episode. "
        "You receive JSON with fitness_capacity, cns_fatigue, muscle_soreness, goal_progress, and stress_event. "
        "Mandatory phases: "
        "Phase 1 (days 1-3): REST only — legs soreness must drop below 0.75. "
        "Phase 2 (days 4-10): Upper body only (push/pull hypertrophy, RPE 6-7), keep resting legs. "
        "Phase 3 (days 11+): Legs ONLY when soreness < 0.4, starting at RPE 5-6. "
        "If stress_event=true: REST regardless of phase (safety first). "
        "Target: fitness≥65, legs_soreness≤0.3, cns≤0.6. "
        "Output ONLY valid JSON: 'workout_category' (rest/liss_cardio/hiit/hypertrophy/strength), "
        "'target_muscle' (none/legs/push/pull/full_body), 'intensity_rpe' (1-10). No other text."
    ),
    "competition-prep": (
        "You are an elite AI Strength & Conditioning Coach preparing an athlete (fitness=75) for competition (target: fitness≥92). "
        "You receive JSON with fitness_capacity, cns_fatigue, muscle_soreness, sleep_quality, weekly_variety_score, and stress_event. "
        "CRITICAL RULES: "
        "1. CNS must stay below 0.5 — elite performance requires fresh nervous system. "
        "2. Must use at least 4 different workout modalities total to earn variety bonus. "
        "3. If stress_event=true: reduce RPE by 1-2 or REST. "
        "4. Near fitness ceiling (>85), gains are small — prioritize consistency. "
        "5. Use nutrition_protocol=surplus on hypertrophy/strength days for +20% gains. "
        "6. Monitor sleep_quality — if <0.5, prioritize REST. "
        "Rotate all 5 modalities. Never let cns_fatigue exceed 0.45. "
        "Output ONLY valid JSON: 'workout_category', 'target_muscle', 'intensity_rpe' (1-10), 'nutrition_protocol'. No other text."
    ),
    "overtraining-recovery": (
        "You are an AI Recovery Specialist. Your client is SEVERELY OVERTRAINED: cns=0.90, ALL soreness=0.60, sleep_quality=0.30. "
        "You receive JSON with fitness_capacity, cns_fatigue, muscle_soreness, sleep_quality, weekly_variety_score, and stress_event. "
        "MANDATORY 3-PHASE PROTOCOL: "
        "Phase 1 (days 1-7): REST ONLY, nutrition_protocol=high_protein. client WILL re-injure if trained now. "
        "Phase 2 (days 8-18): LISS cardio only (RPE 3-4), nutrition_protocol=maintenance. Train only when cns<0.5 AND all soreness<0.4. "
        "Phase 3 (days 19-30): Gradually introduce hypertrophy (RPE 6-7). Use surplus nutrition for gains. "
        "ALWAYS check sleep_quality — if <0.5, the client needs REST regardless of phase. "
        "Target: fitness≥70, cns≤0.4. Skip Phase 1 = automatic failure (re-injury). "
        "Output ONLY valid JSON: 'workout_category', 'target_muscle', 'intensity_rpe' (1-10), 'nutrition_protocol'. No other text."
    ),
}


# ==========================================
# OBSERVATION BUILDER
# ==========================================
def build_obs_dict(obs, task_name: str, steps_taken: int, training_streak: int = 0) -> dict:
    return {
        "fitness_capacity": obs.fitness_capacity,
        "cns_fatigue": obs.cns_fatigue,
        "muscle_soreness": obs.muscle_soreness,
        "days_active": obs.days_active,
        "goal_progress": getattr(obs, "goal_progress", 0.0),
        # days_since_last_rest = training streak (days trained without a rest)
        "days_since_last_rest": training_streak,
        "stress_event": getattr(obs, "stress_event", False),
        "sleep_quality": getattr(obs, "sleep_quality", 0.8),
        "weekly_variety_score": getattr(obs, "weekly_variety_score", 0.0),
        "task": task_name,
        "step": steps_taken + 1,
        "steps_remaining": MAX_STEPS - steps_taken,
    }

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI Personal Trainer. You receive JSON data about a client's physiology. "
    "You must output ONLY a valid JSON object with exactly three keys: "
    "'workout_category' (must be 'rest', 'liss_cardio', 'hiit', 'hypertrophy', or 'strength'), "
    "'target_muscle' (must be 'none', 'legs', 'push', 'pull', or 'full_body'), "
    "and 'intensity_rpe' (integer 1-10). "
    "REST when cns_fatigue > 0.6. Avoid training sore muscles (soreness > 0.7). "
    "Alternate muscle groups. Use RPE 6-8 for growth."
)


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

    try:
        with GymcompanionEnv(base_url=ENV_URL).sync() as env:
            for task_name in TASKS:
                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

                system_prompt = SYSTEM_PROMPTS.get(task_name, DEFAULT_SYSTEM_PROMPT)
                steps_taken = 0
                training_streak = 0  # days trained consecutively without REST
                rewards: List[float] = []
                success = False

                try:
                    result = env.reset(task_name=task_name)
                    done = False
                    messages = [{"role": "system", "content": system_prompt}]

                    while not done and steps_taken < MAX_STEPS:
                        obs = result.observation
                        # Use build_obs_dict so ALL fields (sleep_quality, weekly_variety_score,
                        # days_since_last_rest) are consistently sent to the model.
                        obs_dict = build_obs_dict(obs, task_name, steps_taken, training_streak)
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

                            # Strip markdown fences
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

                        # Track training streak for correct days_since_last_rest
                        last_cat = action.workout_category.value if hasattr(action.workout_category, "value") else str(action.workout_category)
                        if last_cat == "rest":
                            training_streak = 0
                        else:
                            training_streak += 1

                        log_step(
                            step=steps_taken,
                            action=action_str,
                            reward=reward,
                            done=done,
                            error=error_msg,
                        )

                        # Keep conversation context manageable
                        if len(messages) > 20:
                            messages = [messages[0]] + messages[-10:]

                    # Compute final score
                    meta = result.observation.metadata or {}
                    if "score" in meta:
                        final_score = float(meta["score"])
                    else:
                        raw_score = sum(rewards) / len(rewards) if rewards else 0.0
                        final_score = max(0.0, min(1.0, raw_score))

                    success = final_score > 0.0
                    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

                except Exception as e:
                    print(f"[ERROR] Task {task_name} failed: {e}", flush=True)
                    log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards or [0.0])

    except Exception as env_error:
        print(f"[FATAL] Environment error: {env_error}", flush=True)


if __name__ == "__main__":
    run_inference()
