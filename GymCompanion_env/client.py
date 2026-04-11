# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GymCompanion Environment Client."""

from typing import Dict

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
except ImportError:
    # --- FALLBACK STUBS FOR OPENENV CORE ---
    import urllib.request
    import json
    from typing import Generic, TypeVar, Dict, Any, Optional

    ActionT = TypeVar("ActionT")
    ObsT = TypeVar("ObsT")
    StateT = TypeVar("StateT")

    class StepResult(Generic[ObsT]):
        def __init__(self, observation: ObsT, reward: float, done: bool):
            self.observation = observation
            self.reward = float(reward) if reward else 0.0
            self.done = bool(done)

    class State:
        def __init__(self, episode_id: Optional[str] = None, step_count: int = 0):
            self.episode_id = episode_id
            self.step_count = step_count

    class _SyncEnvClient(Generic[ActionT, ObsT, StateT]):
        def __init__(self, client):
            self.client = client
            self.session_id = "default_fallback_session"
            
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def reset(self, task_name: str = "default_task") -> StepResult[ObsT]:
            url = f"{self.client.base_url}/reset"
            data = json.dumps({"task_name": task_name, "session_id": self.session_id}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode())
                return self.client._parse_result(resp_data)
                
        def step(self, action: ActionT) -> StepResult[ObsT]:
            url = f"{self.client.base_url}/step"
            action_payload = self.client._step_payload(action)
            data = json.dumps({"action": action_payload, "session_id": self.session_id}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode())
                return self.client._parse_result(resp_data)

    class EnvClient(Generic[ActionT, ObsT, StateT]):
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")

        def sync(self) -> _SyncEnvClient[ActionT, ObsT, StateT]:
            return _SyncEnvClient(self)
            
        def _step_payload(self, action: ActionT) -> Dict:
            raise NotImplementedError

        def _parse_result(self, payload: Dict) -> StepResult[ObsT]:
            raise NotImplementedError
    # ---------------------------------------

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
            goal_progress=obs_data.get("goal_progress", 0.0),
            days_since_last_rest=obs_data.get("days_since_last_rest", 0),
            stress_event=obs_data.get("stress_event", False),
            sleep_quality=obs_data.get("sleep_quality", 0.8),
            weekly_variety_score=obs_data.get("weekly_variety_score", 0.0),
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
