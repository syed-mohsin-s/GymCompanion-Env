# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the GymCompanion Environment.

The GymCompanion environment is a Physiological Simulator for Health-Tech
AI Agents. It models a human client's physiology and accepts personal-trainer
actions (workout prescriptions) to advance the simulation.
"""

from enum import Enum
from typing import Dict, Optional

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from pydantic import BaseModel
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        metadata: dict = {}
from pydantic import Field


# ── Enums ────────────────────────────────────────────────────────────────────


class WorkoutCategory(str, Enum):
    """Type of workout session prescribed by the trainer agent."""

    REST = "rest"
    LISS_CARDIO = "liss_cardio"
    HIIT = "hiit"
    HYPERTROPHY = "hypertrophy"
    STRENGTH = "strength"


class TargetMuscle(str, Enum):
    """Muscle group targeted by the prescribed workout."""

    NONE = "none"
    LEGS = "legs"
    PUSH = "push"
    PULL = "pull"
    FULL_BODY = "full_body"


class NutritionProtocol(str, Enum):
    """
    Daily nutrition strategy chosen by the trainer agent.

    Directly affects growth rate, soreness recovery, and CNS demands:
    - MAINTENANCE: no modifier (baseline)
    - SURPLUS:     +20% fitness gain, +5% CNS cost (caloric excess supports growth)
    - DEFICIT:     ×0.5 fitness gain, −10% CNS cost (caloric restriction limits gains)
    - HIGH_PROTEIN: −20% soreness cost (protein accelerates muscle repair)
    """

    MAINTENANCE  = "maintenance"
    SURPLUS      = "surplus"
    DEFICIT      = "deficit"
    HIGH_PROTEIN = "high_protein"


# ── Observation ──────────────────────────────────────────────────────────────


class GymcompanionObservation(Observation):
    """
    Observation representing a human client's current physiological state.

    Returned by the environment after each step so the trainer agent can
    decide the next workout prescription.
    """

    fitness_capacity: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description=(
            "Overall fitness capacity of the client on a 0-100 scale. "
            "Higher values indicate better cardiovascular and muscular fitness."
        ),
    )

    cns_fatigue: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Central Nervous System fatigue level (0.0 = fully recovered, "
            "1.0 = completely fatigued). High CNS fatigue impairs performance "
            "and increases injury risk."
        ),
    )

    muscle_soreness: Dict[str, float] = Field(
        default_factory=lambda: {"legs": 0.0, "push": 0.0, "pull": 0.0},
        description=(
            "Per-muscle-group soreness levels. Keys are strictly 'legs', "
            "'push', and 'pull', each with a value from 0.0 (no soreness) "
            "to 1.0 (maximum soreness)."
        ),
    )

    days_active: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive days the client has been active in the program.",
    )

    goal_progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized progress toward the current task goal (0.0 = start, "
            "1.0 = goal achieved). Useful for the agent to gauge how well it is doing."
        ),
    )

    days_since_last_rest: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive training days since the last REST day. High values indicate overtraining risk.",
    )

    stress_event: bool = Field(
        default=False,
        description=(
            "Whether a life-stress event (bad sleep, work pressure) occurred today. "
            "A stress event halves CNS recovery on rest days and adds extra CNS cost "
            "on training days. Agents should adapt by choosing lighter workouts or resting."
        ),
    )

    sleep_quality: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Quality of last night's sleep (0.0 = terrible, 1.0 = perfect). "
            "Scales CNS recovery on rest days. Sleep ≥ 0.9 extends the super-compensation "
            "window. Stress events and very hard training reduce sleep quality."
        ),
    )

    weekly_variety_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Diversity score of workout modalities used in the last 7 days (0.0–1.0). "
            "Using all 5 modalities (rest, liss_cardio, hiit, hypertrophy, strength) gives 1.0. "
            "A score ≥ 0.6 (3+ modalities) indicates good periodization."
        ),
    )


# ── Action ───────────────────────────────────────────────────────────────────


class GymcompanionAction(Action):
    """
    Action representing a Personal Trainer's workout prescription.

    The trainer agent emits an action each step to prescribe the client's
    next training session based on the observed physiological state.
    """

    workout_category: WorkoutCategory = Field(
        ...,
        description=(
            "Category of workout to prescribe. REST for recovery days, "
            "LISS_CARDIO for low-intensity steady-state cardio, HIIT for "
            "high-intensity interval training, HYPERTROPHY for muscle-growth "
            "focused resistance training, STRENGTH for maximal-load training."
        ),
    )

    target_muscle: TargetMuscle = Field(
        ...,
        description=(
            "Muscle group to target. Use NONE for rest days or pure cardio, "
            "LEGS / PUSH / PULL for split routines, or FULL_BODY for "
            "compound sessions."
        ),
    )

    intensity_rpe: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "Rate of Perceived Exertion (1-10). 1 = minimal effort, "
            "10 = maximal effort. Sweet spot for growth is 6-8."
        ),
    )

    nutrition_protocol: NutritionProtocol = Field(
        default=NutritionProtocol.MAINTENANCE,
        description=(
            "Daily nutrition strategy. SURPLUS (+20% gains), DEFICIT (×0.5 gains, faster CNS recovery), "
            "HIGH_PROTEIN (−20% soreness cost), or MAINTENANCE (no modifier). "
            "Choose wisely: surplus during injury-risk phases increases CNS overhead."
        ),
    )
