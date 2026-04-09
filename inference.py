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

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
API_KEY = os.environ.get("API_KEY", HF_TOKEN)
BENCHMARK = "GymCompanion-Env"

def log_start(task_name: str, model_name: str):
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}")

def log_step(step: int, action: str, reward: float, done: bool, error: str = "null"):
    done_str = "true" if done else "false"
    reward_str = f"{reward:.2f}"
    print(f"[STEP] step={step} action={action} reward={reward_str} done={done_str} error={error}")

def log_end(success: bool, steps: int, score: float, rewards: list):
    success_str = "true" if success else "false"
    score_str = f"{score:.2f}"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} score={score_str} rewards={rewards_str}")

async def run_inference():
    if not API_BASE_URL or not MODEL_NAME or not API_KEY:
        print("Missing required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN/API_KEY")
        return

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    env_url = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")
    env_url = env_url.replace("localhost", "127.0.0.1")

    # Run all 3 core hackathon tasks
    tasks = ["couch-to-5k", "plateau-breaker", "injury-rehab"]

    for task_name in tasks:
        log_start(task_name, MODEL_NAME)

        max_retries = 6
        for attempt in range(max_retries):
            try:
                async with GymcompanionEnv(base_url=env_url) as env:
                    result = await env.reset(task_name=task_name)
                    
                    done = False
                    step = 0
                    MAX_STEPS = 30
                    rewards = []
                    
                    system_prompt = (
                        "You are an AI Personal Trainer. You receive JSON data about a client's physiology. "
                        "You must output a JSON action dictating their workout. Prioritize recovery if fatigue or soreness is high.\n\n"
                        "Your output must be strict JSON matching this schema:\n"
                        "{\n"
                        '  "workout_category": "rest" | "liss_cardio" | "hiit" | "hypertrophy" | "strength",\n'
                        '  "target_muscle": "none" | "legs" | "push" | "pull" | "full_body",\n'
                        '  "intensity_rpe": <int between 1 and 10>\n'
                        "}"
                    )
                    messages = [{"role": "system", "content": system_prompt}]

                    while not done and step < MAX_STEPS:
                        step += 1
                        
                        obs_dict = {
                            "fitness_capacity": result.observation.fitness_capacity,
                            "cns_fatigue": result.observation.cns_fatigue,
                            "muscle_soreness": result.observation.muscle_soreness,
                            "days_active": result.observation.days_active
                        }
                        messages.append({"role": "user", "content": json.dumps(obs_dict)})

                        action_str = ""
                        error_msg = "null"
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                temperature=0.2,
                            )
                            
                            llm_output = response.choices[0].message.content
                            messages.append({"role": "assistant", "content": llm_output})
                            
                            # Safely parse markdown-wrapped JSON the proxy might emit
                            cleaned_output = llm_output.strip()
                            if cleaned_output.startswith("```json"):
                                cleaned_output = cleaned_output[7:]
                            elif cleaned_output.startswith("```"):
                                cleaned_output = cleaned_output[3:]
                            if cleaned_output.endswith("```"):
                                cleaned_output = cleaned_output[:-3]
                            cleaned_output = cleaned_output.strip()

                            action_data = json.loads(cleaned_output)
                            action = GymcompanionAction(**action_data)
                            action_str = json.dumps(action_data)
                        except Exception as e:
                            print(f"[LLM ERROR] Exception during chat.completions or JSON parsing: {e}")
                            action = GymcompanionAction(
                                workout_category=WorkoutCategory.REST,
                                target_muscle=TargetMuscle.NONE,
                                intensity_rpe=1
                            )
                            error_msg = str(e).replace('\n', ' ')
                            action_str = '{"workout_category": "rest", "target_muscle": "none", "intensity_rpe": 1}'

                        result = await env.step(action)
                        done = result.done
                        rewards.append(result.reward)
                        
                        log_step(step, action_str, result.reward, done, error_msg)

                    # Evaluate success and final score
                    final_score = result.observation.metadata.get("score", 0.0) if result.observation.metadata else 0.0
                    success = final_score > 0.0

                    log_end(success, step, final_score, rewards)
                    break  # Success! Exit the retry loop for this task.
            except Exception as e:
                print(f"[RETRY] Task '{task_name}' connection/execution failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Max retries reached for task '{task_name}'. Skipping.")
                    log_end(False, 0, 0.0, [])
                else:
                    await asyncio.sleep(5)  # Wait giving container time to boot up


if __name__ == "__main__":
    asyncio.run(run_inference())
