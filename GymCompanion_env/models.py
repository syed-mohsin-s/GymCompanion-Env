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
from typing import Dict

from openenv.core.env_server.types import Action, Observation
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
            "10 = maximal effort. Guides the simulator in scaling volume "
            "and recovery impact."
        ),
    )
