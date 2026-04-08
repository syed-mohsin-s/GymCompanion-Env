# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Physiology Engine — pure-Python state-transition math for GymCompanion-Env.

This module is intentionally **framework-free**: no Pydantic, no OpenEnv, no
FastAPI.  It receives plain Python values (floats, dicts, strings/enums) and
returns plain Python values so it can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ── Constants ────────────────────────────────────────────────────────────────

# Bounds (mirrored from Pydantic model constraints)
FITNESS_MIN: float = 0.0
FITNESS_MAX: float = 100.0
FATIGUE_MIN: float = 0.0
FATIGUE_MAX: float = 1.0
SORENESS_MIN: float = 0.0
SORENESS_MAX: float = 1.0

VALID_MUSCLE_KEYS = frozenset({"legs", "push", "pull"})

# Recovery deltas applied on REST
REST_CNS_RECOVERY: float = 0.3
REST_SORENESS_RECOVERY: float = 0.4

# Injury thresholds
INJURY_RPE_THRESHOLD: int = 8
INJURY_CNS_THRESHOLD: float = 0.7
INJURY_SORENESS_THRESHOLD: float = 0.8

# Growth (productive training) constants
GROWTH_RPE_LOW: int = 6
GROWTH_RPE_HIGH: int = 8
GROWTH_FITNESS_GAIN: float = 0.5
GROWTH_CNS_COST: float = 0.1
GROWTH_SORENESS_COST: float = 0.2

# Rewards
REWARD_REST: float = 0.1
REWARD_INJURY: float = -1.0
REWARD_GROWTH: float = 0.5
REWARD_NEUTRAL: float = 0.0

# Muscle-group mapping for target_muscle → soreness keys
_MUSCLE_KEY_MAP: Dict[str, list[str]] = {
    "none": [],
    "legs": ["legs"],
    "push": ["push"],
    "pull": ["pull"],
    "full_body": ["legs", "push", "pull"],
}


# ── Data containers ──────────────────────────────────────────────────────────


@dataclass
class PhysiologyState:
    """Plain-data snapshot of a client's physiological state."""

    fitness_capacity: float = 50.0
    cns_fatigue: float = 0.0
    muscle_soreness: Dict[str, float] = field(
        default_factory=lambda: {"legs": 0.0, "push": 0.0, "pull": 0.0}
    )
    days_active: int = 0


@dataclass
class TransitionResult:
    """Output of a single state transition."""

    next_state: PhysiologyState
    reward: float
    injured: bool  # True → episode should end immediately


# ── Clamp helper ─────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi] and round to 4 dp to eliminate fp noise."""
    return round(max(lo, min(hi, value)), 4)


# ── Engine ───────────────────────────────────────────────────────────────────


class PhysiologyEngine:
    """
    Stateless calculator for physiological state transitions.

    All methods are pure functions — the engine holds **no mutable state**.
    Call :meth:`transition` with the current state and an action to obtain
    the next state, the step reward, and an injury flag.

    Business rules
    --------------
    1. **REST** → CNS fatigue −0.3, all soreness −0.4, reward +0.1.
    2. **Injury** → RPE > 8 with CNS > 0.7, OR training a muscle with
       soreness > 0.8 → reward −1.0, ``injured = True``.
    3. **Growth** → RPE 6–8 on a rested muscle → fitness +0.5,
       CNS fatigue + cost, local soreness + cost, reward +0.5.
    4. All values clamped to Pydantic-model bounds.
    """

    # ── public API ───────────────────────────────────────────────────────

    @staticmethod
    def transition(
        state: PhysiologyState,
        workout_category: str,
        target_muscle: str,
        intensity_rpe: int,
    ) -> TransitionResult:
        """
        Compute the next physiological state given the current state and a
        trainer action.

        Parameters
        ----------
        state:
            Current physiology snapshot.
        workout_category:
            One of ``"rest"``, ``"liss_cardio"``, ``"hiit"``,
            ``"hypertrophy"``, ``"strength"`` (the ``.value`` of the enum).
        target_muscle:
            One of ``"none"``, ``"legs"``, ``"push"``, ``"pull"``,
            ``"full_body"`` (the ``.value`` of the enum).
        intensity_rpe:
            Rate of Perceived Exertion, integer 1–10.

        Returns
        -------
        TransitionResult
            Contains ``next_state``, ``reward``, and ``injured`` flag.
        """
        # Normalise inputs
        workout_category = workout_category.lower().strip()
        target_muscle = target_muscle.lower().strip()

        # Copy mutable state so we never mutate the caller's object
        fitness = state.fitness_capacity
        cns = state.cns_fatigue
        soreness = dict(state.muscle_soreness)
        days_active = state.days_active

        affected_keys = _MUSCLE_KEY_MAP.get(target_muscle, [])

        # ── Rule 1: REST ─────────────────────────────────────────────────
        if workout_category == "rest":
            cns = _clamp(cns - REST_CNS_RECOVERY, FATIGUE_MIN, FATIGUE_MAX)
            for key in soreness:
                soreness[key] = _clamp(
                    soreness[key] - REST_SORENESS_RECOVERY,
                    SORENESS_MIN,
                    SORENESS_MAX,
                )

            days_active += 1

            next_state = PhysiologyState(
                fitness_capacity=_clamp(fitness, FITNESS_MIN, FITNESS_MAX),
                cns_fatigue=cns,
                muscle_soreness=soreness,
                days_active=days_active,
            )
            return TransitionResult(
                next_state=next_state, reward=REWARD_REST, injured=False
            )

        # ── Rule 2: Injury check (must come before growth) ───────────────
        # Condition A: extreme RPE when CNS is already dangerously high
        rpe_cns_injury = (
            intensity_rpe > INJURY_RPE_THRESHOLD
            and cns > INJURY_CNS_THRESHOLD
        )

        # Condition B: training a muscle that is already very sore
        sore_muscle_injury = any(
            soreness.get(k, 0.0) > INJURY_SORENESS_THRESHOLD
            for k in affected_keys
        )

        if rpe_cns_injury or sore_muscle_injury:
            # Injury occurred — state is frozen, episode should end
            next_state = PhysiologyState(
                fitness_capacity=_clamp(fitness, FITNESS_MIN, FITNESS_MAX),
                cns_fatigue=_clamp(cns, FATIGUE_MIN, FATIGUE_MAX),
                muscle_soreness={
                    k: _clamp(v, SORENESS_MIN, SORENESS_MAX)
                    for k, v in soreness.items()
                },
                days_active=days_active,
            )
            return TransitionResult(
                next_state=next_state, reward=REWARD_INJURY, injured=True
            )

        # ── Rule 3: Growth (productive training at sweet-spot RPE) ───────
        if GROWTH_RPE_LOW <= intensity_rpe <= GROWTH_RPE_HIGH:
            fitness = _clamp(
                fitness + GROWTH_FITNESS_GAIN, FITNESS_MIN, FITNESS_MAX
            )
            cns = _clamp(cns + GROWTH_CNS_COST, FATIGUE_MIN, FATIGUE_MAX)
            for key in affected_keys:
                soreness[key] = _clamp(
                    soreness[key] + GROWTH_SORENESS_COST,
                    SORENESS_MIN,
                    SORENESS_MAX,
                )
            days_active += 1

            next_state = PhysiologyState(
                fitness_capacity=fitness,
                cns_fatigue=cns,
                muscle_soreness=soreness,
                days_active=days_active,
            )
            return TransitionResult(
                next_state=next_state, reward=REWARD_GROWTH, injured=False
            )

        # ── Fallback: sub-optimal training (RPE too low or moderate-high) ─
        # Still train, but no growth bonus.  Light CNS / soreness cost.
        scale = intensity_rpe / 10.0
        cns = _clamp(cns + 0.05 * scale, FATIGUE_MIN, FATIGUE_MAX)
        for key in affected_keys:
            soreness[key] = _clamp(
                soreness[key] + 0.1 * scale,
                SORENESS_MIN,
                SORENESS_MAX,
            )
        # Small fitness bump for effort, but less than growth zone
        fitness = _clamp(fitness + 0.2 * scale, FITNESS_MIN, FITNESS_MAX)
        days_active += 1

        next_state = PhysiologyState(
            fitness_capacity=fitness,
            cns_fatigue=cns,
            muscle_soreness=soreness,
            days_active=days_active,
        )
        return TransitionResult(
            next_state=next_state, reward=REWARD_NEUTRAL, injured=False
        )


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = PhysiologyEngine()
    s = PhysiologyState()

    print("=== REST ===")
    s_high_cns = PhysiologyState(cns_fatigue=0.6, muscle_soreness={"legs": 0.5, "push": 0.3, "pull": 0.2})
    r = engine.transition(s_high_cns, "rest", "none", 1)
    print(f"  CNS: {s_high_cns.cns_fatigue} → {r.next_state.cns_fatigue}  (expected 0.3)")
    print(f"  Legs soreness: {s_high_cns.muscle_soreness['legs']} → {r.next_state.muscle_soreness['legs']}  (expected 0.1)")
    print(f"  Reward: {r.reward}  (expected +0.1)")
    print(f"  Injured: {r.injured}  (expected False)")

    print("\n=== GROWTH (RPE 7, rested push) ===")
    r = engine.transition(s, "hypertrophy", "push", 7)
    print(f"  Fitness: {s.fitness_capacity} → {r.next_state.fitness_capacity}  (expected 50.5)")
    print(f"  CNS: {s.cns_fatigue} → {r.next_state.cns_fatigue}  (expected 0.1)")
    print(f"  Push soreness: {s.muscle_soreness['push']} → {r.next_state.muscle_soreness['push']}  (expected 0.2)")
    print(f"  Reward: {r.reward}  (expected +0.5)")

    print("\n=== INJURY (RPE 9, CNS 0.8) ===")
    s_tired = PhysiologyState(cns_fatigue=0.8)
    r = engine.transition(s_tired, "strength", "legs", 9)
    print(f"  Reward: {r.reward}  (expected -1.0)")
    print(f"  Injured: {r.injured}  (expected True)")

    print("\n=== INJURY (sore muscle > 0.8) ===")
    s_sore = PhysiologyState(muscle_soreness={"legs": 0.85, "push": 0.0, "pull": 0.0})
    r = engine.transition(s_sore, "hypertrophy", "legs", 6)
    print(f"  Reward: {r.reward}  (expected -1.0)")
    print(f"  Injured: {r.injured}  (expected True)")

    print("\n=== SUB-OPTIMAL (RPE 4, low) ===")
    r = engine.transition(s, "liss_cardio", "full_body", 4)
    print(f"  Fitness: {s.fitness_capacity} → {r.next_state.fitness_capacity}")
    print(f"  Reward: {r.reward}  (expected 0.0)")
    print(f"  Injured: {r.injured}  (expected False)")

    print("\n✓ All smoke tests complete.")
