# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GymCompanion Environment Implementation.

Thin orchestration layer that delegates all physiological math to
:class:`PhysiologyEngine` and translates between OpenEnv types and
plain-Python dataclasses.
"""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        GymcompanionAction,
        GymcompanionObservation,
        TargetMuscle,
        WorkoutCategory,
    )
    from .physiology_engine import PhysiologyEngine, PhysiologyState
except ImportError:
    from models import (  # type: ignore[no-redef]
        GymcompanionAction,
        GymcompanionObservation,
        TargetMuscle,
        WorkoutCategory,
    )
    from server.physiology_engine import PhysiologyEngine, PhysiologyState  # type: ignore[no-redef]


# Maximum number of steps before the episode ends
MAX_EPISODE_STEPS = 90  # ~90 days (one quarter)


class GymcompanionEnvironment(Environment):
    """
    OpenEnv environment that wraps :class:`PhysiologyEngine`.

    Responsibilities
    ----------------
    * Manage episode lifecycle (``reset`` / ``step`` / ``state``).
    * Convert Pydantic models ↔ plain dataclasses.
    * Track step count and episode termination (time-limit or injury).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._physiology = PhysiologyState()
        self._engine = PhysiologyEngine()
        self._done = False
        self._cumulative_reward = 0.0
        self._target_max_reward = 16.5

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, *args, **kwargs) -> GymcompanionObservation:
        """Reset the environment based on task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Retrieve task_name from kwargs or kwargs["options"]
        task_name = kwargs.get("task_name")
        if task_name is None and "options" in kwargs and isinstance(kwargs["options"], dict):
            task_name = kwargs["options"].get("task_name")

        # Initialize physiology based on task
        if task_name == "couch-to-5k":
            self._target_max_reward = 16.5  # Assumes mostly growth days
            self._physiology = PhysiologyState(
                fitness_capacity=20.0,
                cns_fatigue=0.0,
                muscle_soreness={"legs": 0.0, "push": 0.0, "pull": 0.0}
            )
        elif task_name == "plateau-breaker":
            self._target_max_reward = 14.0  # Requires initial rest, then growth
            self._physiology = PhysiologyState(
                fitness_capacity=60.0,
                cns_fatigue=0.3,
                muscle_soreness={"legs": 0.2, "push": 0.2, "pull": 0.2}
            )
        elif task_name == "injury-rehab":
            self._target_max_reward = 10.0  # Requires heavy rest and light work
            self._physiology = PhysiologyState(
                fitness_capacity=40.0,
                cns_fatigue=0.5,
                muscle_soreness={"legs": 0.8, "push": 0.1, "pull": 0.1}
            )
        else:
            self._target_max_reward = 16.5
            self._physiology = PhysiologyState()

        self._done = False
        self._cumulative_reward = 0.0

        return self._build_observation(reward=0.0, done=False)

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, action: GymcompanionAction) -> GymcompanionObservation:  # type: ignore[override]
        """
        Advance the simulation by one day.

        Delegates all state-transition math to :class:`PhysiologyEngine`.
        """
        if self._done:
            return self._build_observation(reward=0.0, done=True)

        self._state.step_count += 1

        result = self._engine.transition(
            state=self._physiology,
            workout_category=action.workout_category.value,
            target_muscle=action.target_muscle.value,
            intensity_rpe=action.intensity_rpe,
        )

        self._physiology = result.next_state
        self._cumulative_reward += result.reward

        # Episode terminates on injury, reaching 30 active days, OR hard step cap
        done = result.injured or self._physiology.days_active >= 30 or self._state.step_count >= MAX_EPISODE_STEPS
        self._done = done

        return self._build_observation(reward=result.reward, done=done)

    # ── helpers ──────────────────────────────────────────────────────────

    def _build_observation(
        self, *, reward: float, done: bool
    ) -> GymcompanionObservation:
        p = self._physiology

        metadata = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
        }

        if done:
            # Normalise cumulative reward to [0.0, 1.0] using task-specific denominator
            final_mapped = max(0.0, min(1.0, self._cumulative_reward / self._target_max_reward))
            metadata["score"] = final_mapped

        return GymcompanionObservation(
            fitness_capacity=round(p.fitness_capacity, 2),
            cns_fatigue=round(p.cns_fatigue, 2),
            muscle_soreness={k: round(v, 2) for k, v in p.muscle_soreness.items()},
            days_active=p.days_active,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    @property
    def state(self) -> State:
        return self._state


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = GymcompanionEnvironment()
    obs = env.reset()
    print(f"[reset] fitness={obs.fitness_capacity}  cns={obs.cns_fatigue}  "
          f"soreness={obs.muscle_soreness}  days={obs.days_active}")

    weekly_plan = [
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

    for i, action in enumerate(weekly_plan, 1):
        obs = env.step(action)
        print(f"[day {i:>2}] {action.workout_category.value:>12} RPE {action.intensity_rpe} | "
              f"fitness={obs.fitness_capacity:>5.1f}  cns={obs.cns_fatigue:.2f}  "
              f"soreness={obs.muscle_soreness}  reward={obs.reward:+.2f}  done={obs.done}")
