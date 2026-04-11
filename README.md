---
title: GymCompanion-Env
emoji: 🏋️
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - health-tech
  - reinforcement-learning
  - gym
---

# 🏋️ GymCompanion-Env

**Physiological Simulator for Health-Tech AI Agents**

> An OpenEnv-compatible reinforcement learning environment that models a virtual human client's physiology — fitness capacity, CNS fatigue, and per-muscle-group soreness — enabling AI personal-trainer agents to learn optimal workout programming.

🔗 **Live Space:** [syedmohsin7/GymCompanion-Env](https://huggingface.co/spaces/syedmohsin7/GymCompanion-Env)

---

## Observation Space

| Field              | Type               | Range        | Description                                    |
| ------------------ | ------------------ | ------------ | ---------------------------------------------- |
| `fitness_capacity` | `float`            | 0.0 – 100.0 | Overall cardiovascular & muscular fitness       |
| `cns_fatigue`      | `float`            | 0.0 – 1.0   | Central Nervous System fatigue                  |
| `muscle_soreness`  | `Dict[str, float]` | 0.0 – 1.0   | Per-group soreness (`legs`, `push`, `pull`)     |
| `days_active`      | `int`              | 0+           | Consecutive days in the program                 |

## Action Space

| Field              | Type   | Values                                                     |
| ------------------ | ------ | ---------------------------------------------------------- |
| `workout_category` | `Enum` | `rest`, `liss_cardio`, `hiit`, `hypertrophy`, `strength`   |
| `target_muscle`    | `Enum` | `none`, `legs`, `push`, `pull`, `full_body`                |
| `intensity_rpe`    | `int`  | 1 – 10 (Rate of Perceived Exertion)                        |

---

## Quick Start

```python
from GymCompanion_env import (
    GymcompanionAction, GymcompanionEnv,
    WorkoutCategory, TargetMuscle,
)

# Connect to a running server (e.g., local or HF Space)
with GymcompanionEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(f"Fitness: {result.observation.fitness_capacity}")
    print(f"CNS fatigue: {result.observation.cns_fatigue}")
    print(f"Soreness: {result.observation.muscle_soreness}")

    # Prescribe a workout
    action = GymcompanionAction(
        workout_category=WorkoutCategory.HYPERTROPHY,
        target_muscle=TargetMuscle.PUSH,
        intensity_rpe=7,
    )
    result = env.step(action)
    print(f"Reward: {result.reward:+.4f}")
```

---

## Tasks

Three built-in scenarios with increasing difficulty:

| Task ID            | Difficulty | Starting Fitness | Goal                                      |
| ------------------ | ---------- | ---------------- | ----------------------------------------- |
| `couch-to-5k`     | 🟢 Easy   | 20.0             | Build a sedentary client to ≥55 fitness   |
| `plateau-breaker`  | 🟡 Medium | 60.0             | Push a stalled client past ≥80 fitness    |
| `injury-rehab`     | 🔴 Hard   | 40.0             | Rehabilitate a leg injury to ≥65 fitness  |

```python
result = env.reset(task_name="couch-to-5k")
```

---

## Reward Signal

A multi-component reward designed to teach safe, effective training:

| Component              | Formula                                   | Effect    |
| ---------------------- | ----------------------------------------- | --------- |
| **Fitness gain**       | `(fitness - 50) / 50`                     | Primary   |
| **Overtraining penalty** | Quadratic escalation when CNS > 0.7     | Negative  |
| **Injury penalty**     | High penalty when any soreness > 0.9      | Negative  |
| **Consistency bonus**  | `0.02 × log(1 + days_active)`             | Positive  |
| **Smart rest bonus**   | +0.3 when resting with high CNS fatigue   | Positive  |

---

## Simulation Dynamics

Each step simulates **one day**:

1. **Natural recovery** — soreness decays by 0.15, CNS recovers by 0.08
2. **Workout effects** — fitness gain scaled by RPE and dampened by existing CNS fatigue
3. **Soreness amplification** — training an already-sore muscle causes amplified soreness
4. **Episode length** — 90 steps (≈ 3 months / one quarter)

---

## Environment Variables

The inference script requires these environment variables:

| Variable       | Description                         | Default                                  |
| -------------- | ----------------------------------- | ---------------------------------------- |
| `API_BASE_URL` | LLM API endpoint                    | `https://router.huggingface.co/v1`       |
| `MODEL_NAME`   | Model identifier for inference      | `Qwen/Qwen2.5-72B-Instruct`             |
| `HF_TOKEN`     | Hugging Face / API key (**required**) | —                                      |

---

## Deployment

### Deploy to Hugging Face Spaces

```bash
cd GymCompanion_env
openenv push
```

### Run Locally

```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Run inference
HF_TOKEN=your_token python inference.py
```

### Build Docker Image

```bash
docker build -t gymcompanion-env:latest .
docker run -p 7860:7860 gymcompanion-env:latest
```

---

## Deployed Endpoints

| Endpoint     | Description                                |
| ------------ | ------------------------------------------ |
| `/web`       | Interactive web UI for exploring the env   |
| `/docs`      | OpenAPI / Swagger documentation            |
| `/health`    | Container health check                     |
| `/ws`        | WebSocket endpoint for persistent sessions |
| `/schema`    | Action/observation schema                  |

---

## Project Structure

```
GymCompanion_env/
├── inference.py           # Hackathon inference loop (LLM → actions)
├── client.py              # GymcompanionEnv WebSocket client
├── models.py              # Observation, Action, and Enum models
├── openenv.yaml           # OpenEnv manifest (tasks, schemas)
├── pyproject.toml         # Project metadata and dependencies
├── Dockerfile             # Container image for HF Spaces
├── README.md              # This file
└── server/
    ├── app.py             # FastAPI application (HTTP + WebSocket)
    ├── GymCompanion_env_environment.py  # Core gymnasium-like environment
    ├── physiology_engine.py             # Physiological simulation engine
    └── requirements.txt                 # Server-side dependencies
```

---

## License

BSD-style license. See [LICENSE](LICENSE) for details.
