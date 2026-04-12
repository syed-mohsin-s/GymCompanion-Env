# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GymCompanion Environment — orchestration layer over PhysiologyEngine.

5 tasks (easy → medium → hard → expert → specialist):
  couch-to-5k          fitness 20 → 55   (easy, fresh client)
  plateau-breaker      fitness 60 → 80   (medium, stalled athlete)
  injury-rehab         fitness 40 → 65   (hard, severe leg injury)
  competition-prep     fitness 75 → 92   (expert, strict CNS ceiling)
  overtraining-recovery fitness 55 → 70  (specialist, massively overtrained client)

Scoring rules:
  - Score = 0.0 if fitness improved by < 2.0 (no effort detected)
  - Score capped at 0.35 if fitness goal not fully met
  - Composite: 60% fitness + 20% reward + 20% CNS health
  - competition-prep: +0.05 variety bonus (4+ modalities)
"""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        GymcompanionAction,
        GymcompanionObservation,
        NutritionProtocol,
        TargetMuscle,
        WorkoutCategory,
    )
    from .physiology_engine import (
        PhysiologyEngine,
        PhysiologyState,
        compute_weekly_variety_score,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        GymcompanionAction,
        GymcompanionObservation,
        NutritionProtocol,
        TargetMuscle,
        WorkoutCategory,
    )
    from server.physiology_engine import (  # type: ignore[no-redef]
        PhysiologyEngine,
        PhysiologyState,
        compute_weekly_variety_score,
    )


MAX_EPISODE_STEPS = 30
STRESS_EVENT_PROBABILITY = 0.10

# Task definitions — mirrors openenv.yaml success_criteria exactly
TASK_CONFIGS = {
    "couch-to-5k": {
        "fitness_capacity": 20.0,
        "cns_fatigue": 0.0,
        "sleep_quality": 0.8,
        "muscle_soreness": {"legs": 0.0, "push": 0.0, "pull": 0.0},
        "goal_fitness": 55.0,
        "max_cns": 0.6,
        "target_max_reward": 20.0,
    },
    "plateau-breaker": {
        "fitness_capacity": 60.0,
        "cns_fatigue": 0.3,
        "sleep_quality": 0.7,
        "muscle_soreness": {"legs": 0.2, "push": 0.2, "pull": 0.2},
        "goal_fitness": 80.0,
        "max_cns": 0.7,
        "target_max_reward": 15.0,
    },
    "injury-rehab": {
        "fitness_capacity": 40.0,
        "cns_fatigue": 0.5,
        "sleep_quality": 0.6,
        "muscle_soreness": {"legs": 0.85, "push": 0.1, "pull": 0.1},
        "goal_fitness": 65.0,
        "max_cns": 0.6,
        "max_leg_soreness": 0.3,
        "target_max_reward": 12.0,
    },
    "competition-prep": {
        "fitness_capacity": 75.0,
        "cns_fatigue": 0.0,
        "sleep_quality": 0.9,
        "muscle_soreness": {"legs": 0.0, "push": 0.0, "pull": 0.0},
        "goal_fitness": 92.0,
        "max_cns": 0.5,
        "target_max_reward": 18.0,
    },
    "overtraining-recovery": {
        # Severely overtrained: very high CNS + high soreness everywhere + poor sleep
        "fitness_capacity": 55.0,
        "cns_fatigue": 0.90,
        "sleep_quality": 0.3,   # chronic overtraining → terrible sleep
        "muscle_soreness": {"legs": 0.6, "push": 0.6, "pull": 0.6},
        "goal_fitness": 70.0,
        "max_cns": 0.4,          # stricter than normal — must fully recover first
        "target_max_reward": 14.0,
    },
}


class GymcompanionEnvironment(Environment):
    """
    OpenEnv environment wrapping PhysiologyEngine.

    Manages episode lifecycle, stochastic stress events, nutrition relay,
    sleep tracking, weekly variety scoring, and strict score gating.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._physiology = PhysiologyState()
        self._engine = PhysiologyEngine()
        self._done = False
        self._cumulative_reward = 0.0
        self._task_config: dict = TASK_CONFIGS["couch-to-5k"]
        self._task_name: str = "couch-to-5k"
        self._stress_event_active: bool = False
        self._modalities_used: set = set()

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, *args, **kwargs) -> GymcompanionObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        task_name = kwargs.get("task_name")
        if task_name is None and "options" in kwargs and isinstance(kwargs["options"], dict):
            task_name = kwargs["options"].get("task_name")

        cfg = TASK_CONFIGS.get(task_name)
        if cfg is None:
            # Unknown task — fall back to couch-to-5k so config and task_name stay in sync
            task_name = "couch-to-5k"
            cfg = TASK_CONFIGS["couch-to-5k"]
        self._task_config = cfg
        self._task_name = task_name

        self._physiology = PhysiologyState(
            fitness_capacity=cfg["fitness_capacity"],
            cns_fatigue=cfg["cns_fatigue"],
            sleep_quality=cfg.get("sleep_quality", 0.8),
            muscle_soreness=dict(cfg["muscle_soreness"]),
        )
        self._done = False
        self._cumulative_reward = 0.0
        self._stress_event_active = False
        self._modalities_used = set()

        return self._build_observation(reward=0.0, done=False)

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, action: GymcompanionAction) -> GymcompanionObservation:  # type: ignore[override]
        if self._done:
            return self._build_observation(reward=0.0, done=True)

        self._state.step_count += 1
        self._stress_event_active = random.random() < STRESS_EVENT_PROBABILITY

        # Extract nutrition from action (default to maintenance if missing)
        nutrition = getattr(action, "nutrition_protocol", NutritionProtocol.MAINTENANCE)
        nutrition_val = nutrition.value if hasattr(nutrition, "value") else "maintenance"

        result = self._engine.transition(
            state=self._physiology,
            workout_category=action.workout_category.value,
            target_muscle=action.target_muscle.value,
            intensity_rpe=action.intensity_rpe,
            stress_event=self._stress_event_active,
            nutrition_protocol=nutrition_val,
        )

        self._physiology = result.next_state
        self._cumulative_reward += result.reward
        self._modalities_used.add(action.workout_category.value)

        done = result.injured or self._state.step_count >= MAX_EPISODE_STEPS
        self._done = done

        return self._build_observation(reward=result.reward, done=done)

    # ── scoring ──────────────────────────────────────────────────────────

    def _compute_score(self) -> float:
        """
        Normalized score in [0.0, 1.0] strictly aligned with success_criteria.

        Gates:
        1. No meaningful effort (fitness improved < 2.0) → 0.0
        2. Goal not met → score capped at 0.35
        3. CNS violations → penalty
        4. Leg soreness violations (injury-rehab) → penalty
        5. competition-prep variety bonus: 4+ modalities used
        """
        cfg = self._task_config
        p = self._physiology
        goal_fitness = cfg["goal_fitness"]
        start_fitness = cfg["fitness_capacity"]
        max_cns = cfg.get("max_cns", 0.7)

        # ── Gate 1: No effort ──────────────────────────────────────────
        fitness_improvement = p.fitness_capacity - start_fitness
        if fitness_improvement < 2.0:
            return 0.0

        # ── Fitness attainment ─────────────────────────────────────────
        fitness_gap = max(1.0, goal_fitness - start_fitness)
        fitness_score = min(1.0, fitness_improvement / fitness_gap)

        # ── Gate 2: Fitness goal gate ──────────────────────────────────
        goal_met = p.fitness_capacity >= goal_fitness
        if not goal_met:
            fitness_score = min(fitness_score, 0.35)

        # ── CNS penalty ────────────────────────────────────────────────
        cns_penalty = 0.0
        if p.cns_fatigue > max_cns:
            cns_penalty = min(0.3, (p.cns_fatigue - max_cns) / (1.0 - max_cns) * 0.3)

        # ── Injury-rehab leg soreness penalty ─────────────────────────
        leg_penalty = 0.0
        if "max_leg_soreness" in cfg:
            max_leg = cfg["max_leg_soreness"]
            leg_sor = p.muscle_soreness.get("legs", 0.0)
            if leg_sor > max_leg:
                leg_penalty = min(0.4, (leg_sor - max_leg) / 0.7 * 0.4)

        # ── Competition-prep variety bonus ─────────────────────────────
        variety_bonus = 0.0
        if self._task_name == "competition-prep" and len(self._modalities_used) >= 4:
            variety_bonus = 0.05

        # ── Reward component ───────────────────────────────────────────
        reward_score = min(1.0, max(0.0, self._cumulative_reward / cfg["target_max_reward"]))

        raw = (
            0.60 * fitness_score
            + 0.20 * reward_score
            + variety_bonus
            - cns_penalty
            - leg_penalty
        )
        # ── Gate 2 enforcement on FINAL score (not just fitness component) ──
        if not goal_met:
            raw = min(raw, 0.35)
        return round(max(0.0, min(1.0, raw)), 4)

    # ── observation builder ───────────────────────────────────────────────

    def _build_observation(self, *, reward: float, done: bool) -> GymcompanionObservation:
        p = self._physiology
        cfg = self._task_config
        goal_fitness = cfg["goal_fitness"]
        start_fitness = cfg["fitness_capacity"]

        fitness_gap = max(1.0, goal_fitness - start_fitness)
        goal_progress = round(
            min(1.0, max(0.0, (p.fitness_capacity - start_fitness) / fitness_gap)), 3
        )
        weekly_variety = compute_weekly_variety_score(p.week_history)

        metadata: dict = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "task": self._task_name,
            "goal_fitness": goal_fitness,
            "goal_progress": goal_progress,
            "days_since_last_rest": p.training_streak,
            "stress_event": self._stress_event_active,
            "sleep_quality": round(p.sleep_quality, 2),
            "weekly_variety_score": weekly_variety,
        }

        if done:
            final_score = self._compute_score()
            goal_achieved = p.fitness_capacity >= goal_fitness
            metadata["score"] = final_score
            metadata["goal_achieved"] = goal_achieved
            metadata["final_fitness"] = round(p.fitness_capacity, 2)
            metadata["final_cns"] = round(p.cns_fatigue, 2)
            metadata["final_sleep_quality"] = round(p.sleep_quality, 2)
            metadata["modalities_used"] = list(self._modalities_used)
            metadata["weekly_variety_score"] = weekly_variety

        return GymcompanionObservation(
            fitness_capacity=round(p.fitness_capacity, 2),
            cns_fatigue=round(p.cns_fatigue, 2),
            muscle_soreness={k: round(v, 2) for k, v in p.muscle_soreness.items()},
            days_active=p.days_active,
            goal_progress=goal_progress,
            days_since_last_rest=p.training_streak,
            stress_event=self._stress_event_active,
            sleep_quality=round(p.sleep_quality, 2),
            weekly_variety_score=weekly_variety,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    @property
    def state(self) -> State:
        return self._state
