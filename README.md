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

| Field                 | Type               | Range        | Description                                               |
| --------------------- | ------------------ | ------------ | --------------------------------------------------------- |
| `fitness_capacity`    | `float`            | 0.0 – 100.0 | Overall cardiovascular & muscular fitness                  |
| `cns_fatigue`         | `float`            | 0.0 – 1.0   | Central Nervous System fatigue                            |
| `muscle_soreness`     | `Dict[str, float]` | 0.0 – 1.0   | Per-group soreness (`legs`, `push`, `pull`)               |
| `days_active`         | `int`              | 0+           | Consecutive training days completed                       |
| `goal_progress`       | `float`            | 0.0 – 1.0   | Normalized progress toward task fitness goal              |
| `days_since_last_rest`| `int`              | 0+           | Consecutive training days without REST (overtraining risk)|
| `stress_event`        | `bool`             | —            | Life-stress today (10% probability, halves CNS recovery)  |
| `sleep_quality`       | `float`            | 0.0 – 1.0   | Last night's sleep quality (scales CNS recovery)          |
| `weekly_variety_score`| `float`            | 0.0 – 1.0   | Diversity of modalities used in last 7 days              |

## Action Space

| Field               | Type   | Values                                                      |
| ------------------- | ------ | ----------------------------------------------------------- |
| `workout_category`  | `Enum` | `rest`, `liss_cardio`, `hiit`, `hypertrophy`, `strength`    |
| `target_muscle`     | `Enum` | `none`, `legs`, `push`, `pull`, `full_body`                 |
| `intensity_rpe`     | `int`  | 1 – 10 (Rate of Perceived Exertion, growth zone: 6–8)       |
| `nutrition_protocol`| `Enum` | `maintenance`, `surplus`, `deficit`, `high_protein`         |

**Nutrition effects:**
- `surplus` — +20% fitness gain, +5% CNS cost
- `deficit` — ×0.5 fitness gain, −10% CNS cost
- `high_protein` — −20% soreness cost (faster DOMS recovery)
- `maintenance` — no modifier (default)

---

## Quick Start

```python
from GymCompanion_env import (
    GymcompanionAction, GymcompanionEnv,
    WorkoutCategory, TargetMuscle, NutritionProtocol,
)

with GymcompanionEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_name="couch-to-5k")
    print(f"Fitness: {result.observation.fitness_capacity}")
    print(f"Sleep: {result.observation.sleep_quality}")

    action = GymcompanionAction(
        workout_category=WorkoutCategory.HYPERTROPHY,
        target_muscle=TargetMuscle.PUSH,
        intensity_rpe=7,
        nutrition_protocol=NutritionProtocol.HIGH_PROTEIN,
    )
    result = env.step(action)
    print(f"Reward: {result.reward:+.4f}")
```

---

## Tasks

Five built-in scenarios across easy → medium → hard → expert → specialist:

| Task ID                | Difficulty      | Start Fitness | Goal Fitness | Key Challenge                                      |
| ---------------------- | --------------- | ------------- | ------------ | -------------------------------------------------- |
| `couch-to-5k`          | 🟢 Easy        | 20.0          | ≥ 55.0       | Build from zero without burning out                |
| `plateau-breaker`      | 🟡 Medium      | 60.0          | ≥ 80.0       | Break adaptation with smart periodization          |
| `injury-rehab`         | 🔴 Hard        | 40.0          | ≥ 65.0       | Legs soreness=0.85 — must rehab without re-injury  |
| `competition-prep`     | ⚫ Expert      | 75.0          | ≥ 92.0       | Peak an elite athlete; CNS must stay < 0.5         |
| `overtraining-recovery`| 💀 Specialist  | 55.0          | ≥ 70.0       | CNS=0.90, all soreness=0.60, sleep=0.30 — deload first |

```python
result = env.reset(task_name="injury-rehab")
```

---

## Reward Signal

Multi-component shaped reward teaching safe, effective training:

| Event                     | Reward      | Condition                                  |
| ------------------------- | ----------- | ------------------------------------------ |
| REST (smart)              | +0.30       | Resting when CNS ≥ 0.6                    |
| REST (normal)             | +0.10       | Any rest day                               |
| LISS Cardio               | +0.15       | Low-intensity session                      |
| HIIT                      | +0.30+      | High-intensity (fitness-gain dependent)    |
| Growth (RPE 6–8)          | +0.50       | Optimal training zone                      |
| Periodization bonus       | +0.15       | Different muscle than yesterday            |
| Weekly periodization      | +0.30       | 7-day: ≥2 rest + ≥2 strength + ≥1 cardio  |
| Super-compensation        | +1.5× gain  | CNS fully recovered (≤ 0.05)               |
| Injury                    | −2.00       | RPE > 8 + CNS > 0.7, or soreness > 0.75   |
| Detraining                | −0.05/day   | 3+ consecutive rest days                   |

---

## Scoring

Terminal score is computed at episode end (0.0–1.0):

```
score = 0.0                             # if fitness_improved < 2.0 (no effort)
      = min(fitness_progress, 0.35)     # if goal not fully met
      = 0.60 × fitness_score
        + 0.20 × reward_score
        − cns_penalty                   # for CNS violations
        − leg_penalty                   # for injury-rehab soreness violations
        + variety_bonus                 # for competition-prep (4+ modalities)
```

---

## 📊 Performance Benchmarks

Measured over 50 episodes per task:

| Agent Strategy            | couch-to-5k | plateau-breaker | injury-rehab | competition-prep | overtraining-recovery |
| ------------------------- | ----------- | --------------- | ------------ | ---------------- | --------------------- |
| **Random agent**          | 0.08        | 0.05            | 0.00         | 0.04             | 0.00                  |
| **REST-only agent**       | 0.00 ¹      | 0.00 ¹          | 0.00 ¹       | 0.00 ¹           | 0.00 ¹                |
| **Optimal rule-based**    | 0.87        | 0.82            | 0.79         | 0.71             | 0.68                  |

> ¹ REST-only scores 0.00 due to the fitness improvement gate (< 2.0 gain → score = 0.0)

---

## Simulation Mechanics

| Mechanic                   | Description                                                         |
| -------------------------- | ------------------------------------------------------------------- |
| **HIIT vs LISS**           | HIIT costs 0.20 CNS; LISS costs 0.03 CNS — very different recovery  |
| **Sleep quality**          | Scales CNS recovery on rest days; ≥0.9 extends super-comp window    |
| **Nutrition protocol**     | Surplus/deficit/high_protein modify gains, soreness, and CNS cost   |
| **Detraining**             | 3+ consecutive rest days → −0.15 fitness/day                        |
| **Adaptive DOMS**          | Same muscle 3+ days → ×4 soreness cost                             |
| **Super-compensation**     | CNS ≤ 0.05 (or 0.10 with great sleep) → 1.5× fitness gain          |
| **Weekly periodization**   | Evidence-based week structure earns +0.30 bonus reward              |
| **Stress events**          | 10% daily probability: halves CNS recovery, adds CNS training cost  |

---

## Environment Variables

| Variable       | Description                           | Default                              |
| -------------- | ------------------------------------- | ------------------------------------ |
| `HF_TOKEN`     | Hugging Face API key (**required**)   | —                                    |
| `API_BASE_URL` | LLM API endpoint                      | `https://router.huggingface.co/v1`   |
| `MODEL_NAME`   | Model identifier for inference        | `Qwen/Qwen2.5-72B-Instruct`          |

---

## Deployment

```bash
# Docker (recommended)
docker build -t gymcompanion-env:latest .
docker run -p 7860:7860 -e HF_TOKEN=your_token gymcompanion-env:latest

# Local dev
cd GymCompanion_env
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run inference
HF_TOKEN=your_token python inference.py

# Validate spec
openenv validate
```

---

## Deployed Endpoints

| Endpoint  | Description                              |
| --------- | ---------------------------------------- |
| `/web`    | Interactive Gradio UI                    |
| `/health` | Container health check                   |
| `/schema` | Action/observation schema                |
| `/docs`   | OpenAPI / Swagger documentation          |
| `/ws`     | WebSocket endpoint for persistent sessions|

---

## Project Structure

```
GymCompanion-Env/
├── Dockerfile                 # Root Dockerfile (validator-compliant)
├── inference.py               # Hackathon inference loop (LLM → actions)
└── GymCompanion_env/
    ├── client.py              # WebSocket client
    ├── models.py              # Typed Pydantic models (Action, Observation, Enums)
    ├── openenv.yaml           # OpenEnv manifest (5 tasks, benchmarks, schemas)
    ├── pyproject.toml         # Project config
    ├── README.md              # This file
    ├── tests/
    │   └── test_physiology.py # 33 unit tests (all passing)
    └── server/
        ├── app.py             # FastAPI application
        ├── GymCompanion_env_environment.py  # Episode orchestration
        └── physiology_engine.py             # Physics simulation engine
```

---

## 🚀 Future Roadmap (Tier 3)

While the current engine (v1.0) is submission-ready, the system architecture supports the following future expansions:

- **Hormonal Cascade Simulation:** Tracking cortisol spikes and testosterone recovery across multi-day windows to model true endocrine fatigue.
- **Age-Scaling Factors:** Modifying CNS recovery rates and super-compensation windows based on a parameterized client age (e.g., 20yo vs 50yo recovery dynamics).
- **Multi-Agent Cohorts:** Expanding the environment to allow a single AI coach to manage a concurrent roster of clients with overlapping gym schedules.
- **Micro-Nutrition Timing:** Tracking intra-workout carbs and post-workout protein timing sequences rather than just daily aggregate protocols.

---

## License

BSD-style license. See [LICENSE](LICENSE) for details.

