# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GymCompanion Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        GymcompanionAction,
        GymcompanionObservation,
        TargetMuscle,
        WorkoutCategory,
    )
except ImportError:
    from models import (
        GymcompanionAction,
        GymcompanionObservation,
        TargetMuscle,
        WorkoutCategory,
    )


class GymcompanionEnv(
    EnvClient[GymcompanionAction, GymcompanionObservation, State]
):
    """
    Client for the GymCompanion Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Example:
        >>> from GymCompanion_env import (
        ...     GymcompanionEnv, GymcompanionAction,
        ...     WorkoutCategory, TargetMuscle,
        ... )
        >>> with GymcompanionEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.fitness_capacity)
        ...
        ...     action = GymcompanionAction(
        ...         workout_category=WorkoutCategory.HYPERTROPHY,
        ...         target_muscle=TargetMuscle.PUSH,
        ...         intensity_rpe=7,
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.muscle_soreness)
    """

    def _step_payload(self, action: GymcompanionAction) -> Dict:
        """
        Convert GymcompanionAction to JSON payload for step message.

        Args:
            action: GymcompanionAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "workout_category": action.workout_category.value,
            "target_muscle": action.target_muscle.value,
            "intensity_rpe": action.intensity_rpe,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GymcompanionObservation]:
        """
        Parse server response into StepResult[GymcompanionObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with GymcompanionObservation
        """
        obs_data = payload.get("observation", {})
        observation = GymcompanionObservation(
            fitness_capacity=obs_data.get("fitness_capacity", 50.0),
            cns_fatigue=obs_data.get("cns_fatigue", 0.0),
            muscle_soreness=obs_data.get(
                "muscle_soreness", {"legs": 0.0, "push": 0.0, "pull": 0.0}
            ),
            days_active=obs_data.get("days_active", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
