---
title: GymCompanion-Env
emoji: 🏋️
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - health-tech
  - reinforcement-learning
---

# GymCompanion-Env

**Physiological Simulator for Health-Tech AI Agents.**

GymCompanion-Env models a virtual human client's physiology — fitness capacity, CNS fatigue, and per-muscle-group soreness — so an AI personal-trainer agent can learn optimal workout programming through reinforcement learning.

| Observation Field    | Type               | Range        | Description                                   |
| -------------------- | ------------------ | ------------ | --------------------------------------------- |
| `fitness_capacity`   | `float`            | 0.0 – 100.0 | Overall cardiovascular & muscular fitness      |
| `cns_fatigue`        | `float`            | 0.0 – 1.0   | Central Nervous System fatigue                 |
| `muscle_soreness`    | `Dict[str, float]` | 0.0 – 1.0   | Per-group soreness (`legs`, `push`, `pull`)    |
| `days_active`        | `int`              | 0+           | Consecutive days in the program                |

| Action Field         | Type   | Values                                                     |
| -------------------- | ------ | ---------------------------------------------------------- |
| `workout_category`   | `Enum` | `REST`, `LISS_CARDIO`, `HIIT`, `HYPERTROPHY`, `STRENGTH`  |
| `target_muscle`      | `Enum` | `NONE`, `LEGS`, `PUSH`, `PULL`, `FULL_BODY`               |
| `intensity_rpe`      | `int`  | 1 – 10 (Rate of Perceived Exertion)                       |

## Quick Start

```python
from GymCompanion_env import (
    GymcompanionAction,
    GymcompanionEnv,
    WorkoutCategory,
    TargetMuscle,
)

try:
    # Create environment from Docker image
    env = GymcompanionEnv.from_docker_image("GymCompanion_env-env:latest")

    # Reset — returns a baseline sedentary client
    result = env.reset()
    print(f"Fitness: {result.observation.fitness_capacity}")
    print(f"CNS fatigue: {result.observation.cns_fatigue}")
    print(f"Soreness: {result.observation.muscle_soreness}")

    # Prescribe a week of training
    week = [
        GymcompanionAction(workout_category=WorkoutCategory.HYPERTROPHY,
                           target_muscle=TargetMuscle.PUSH, intensity_rpe=7),
        GymcompanionAction(workout_category=WorkoutCategory.HYPERTROPHY,
                           target_muscle=TargetMuscle.PULL, intensity_rpe=7),
        GymcompanionAction(workout_category=WorkoutCategory.STRENGTH,
                           target_muscle=TargetMuscle.LEGS, intensity_rpe=8),
        GymcompanionAction(workout_category=WorkoutCategory.REST,
                           target_muscle=TargetMuscle.NONE, intensity_rpe=1),
        GymcompanionAction(workout_category=WorkoutCategory.LISS_CARDIO,
                           target_muscle=TargetMuscle.FULL_BODY, intensity_rpe=5),
        GymcompanionAction(workout_category=WorkoutCategory.HIIT,
                           target_muscle=TargetMuscle.FULL_BODY, intensity_rpe=8),
        GymcompanionAction(workout_category=WorkoutCategory.REST,
                           target_muscle=TargetMuscle.NONE, intensity_rpe=1),
    ]

    for day, action in enumerate(week, 1):
        result = env.step(action)
        obs = result.observation
        print(f"Day {day}: {action.workout_category.value:>12} RPE {action.intensity_rpe} → "
              f"fitness={obs.fitness_capacity:.1f}  cns={obs.cns_fatigue:.2f}  "
              f"soreness={obs.muscle_soreness}  reward={result.reward:+.4f}")

finally:
    env.close()
```

The `GymcompanionEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Tasks

Three built-in task scenarios with increasing difficulty:

| Task ID            | Difficulty | Starting Fitness | Goal                                       |
| ------------------ | ---------- | ---------------- | ------------------------------------------ |
| `couch-to-5k`     | Easy       | 20.0             | Build a sedentary client to ≥55 fitness    |
| `plateau-breaker`  | Medium     | 60.0             | Push a stalled client past ≥80 fitness     |
| `injury-rehab`     | Hard       | 40.0             | Rehabilitate a leg injury to ≥65 fitness   |

## Reward Signal

The reward is a multi-component signal designed to teach safe, effective training:

| Component              | Formula                                              | Weight    |
| ---------------------- | ---------------------------------------------------- | --------- |
| **Fitness gain**       | `(fitness - 50) / 50`                                | Primary   |
| **Overtraining penalty** | Quadratic escalation when CNS > 0.7                | Negative  |
| **Injury penalty**     | High penalty when any soreness > 0.9                 | Negative  |
| **Consistency bonus**  | `0.02 × log(1 + days_active)`                        | Positive  |
| **Smart rest bonus**   | +0.3 when resting with high CNS fatigue              | Positive  |

## Simulation Dynamics

Each step simulates one day:

1. **Natural recovery** — soreness decays by 0.15, CNS recovers by 0.08
2. **Workout effects** — fitness gain scaled by RPE and dampened by existing CNS fatigue
3. **Soreness amplification** — training an already-sore muscle causes amplified soreness
4. **Episode length** — 90 steps (≈ one quarter / 3 months)

## Building the Docker Image

```bash
# From project root
docker build -t GymCompanion_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Advanced Usage

### Connecting to an Existing Server

```python
from GymCompanion_env import GymcompanionEnv, GymcompanionAction, WorkoutCategory, TargetMuscle

env = GymcompanionEnv(base_url="<ENV_HTTP_URL_HERE>")
result = env.reset()

action = GymcompanionAction(
    workout_category=WorkoutCategory.HYPERTROPHY,
    target_muscle=TargetMuscle.PUSH,
    intensity_rpe=7,
)
result = env.step(action)
print(f"Fitness: {result.observation.fitness_capacity}")
```

### Using the Context Manager

```python
from GymCompanion_env import GymcompanionAction, GymcompanionEnv, WorkoutCategory, TargetMuscle

with GymcompanionEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Fitness: {result.observation.fitness_capacity}")

    for _ in range(7):
        result = env.step(GymcompanionAction(
            workout_category=WorkoutCategory.LISS_CARDIO,
            target_muscle=TargetMuscle.FULL_BODY,
            intensity_rpe=5,
        ))
        print(f"Fitness: {result.observation.fitness_capacity:.1f}  "
              f"CNS: {result.observation.cns_fatigue:.2f}")
```

### Concurrent WebSocket Sessions

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    GymcompanionEnvironment,
    GymcompanionAction,
    GymcompanionObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

```python
from GymCompanion_env import GymcompanionAction, GymcompanionEnv, WorkoutCategory, TargetMuscle
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with GymcompanionEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(30):
            result = env.step(GymcompanionAction(
                workout_category=WorkoutCategory.HYPERTROPHY,
                target_muscle=TargetMuscle.PUSH,
                intensity_rpe=6,
            ))
        return client_id, result.observation.fitness_capacity

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
python -m server.GymCompanion_env_environment
```

This verifies that:
- Environment resets to baseline physiology
- Workouts modify fitness, CNS fatigue, and soreness correctly
- Natural recovery applies each timestep
- Reward signal responds to good and bad training decisions

### Running Locally

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
GymCompanion_env/
├── __init__.py            # Module exports (models, client, enums)
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest (tasks, spaces)
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # GymcompanionEnv WebSocket client
├── models.py              # Observation, Action, and Enum models
└── server/
    ├── __init__.py        # Server module exports
    ├── GymCompanion_env_environment.py  # Physiological simulator
    ├── app.py             # FastAPI application (HTTP + WebSocket)
    └── Dockerfile         # Container image definition
```
