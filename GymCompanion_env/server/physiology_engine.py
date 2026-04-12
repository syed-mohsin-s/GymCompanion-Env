# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Physiology Engine — pure-Python state-transition math for GymCompanion-Env.

Framework-free: no Pydantic, no OpenEnv. Pure dataclasses + floats.

Physics features:
  REST         — CNS recovery scaled by sleep_quality; smart-rest bonus when CNS ≥ 0.6
  Sleep        — quality set each night (0.0–1.0); stress & hard RPE impair it
  Nutrition    — surplus: +20% growth | deficit: ×0.5 growth | high_protein: −20% soreness
  Growth       — RPE 6–8; super-comp at full CNS recovery (1.5×, extended by good sleep)
  HIIT         — high CNS cost (0.20), moderate fitness gain
  LISS         — low CNS cost (0.03), small fitness gain
  Detraining   — 3+ consecutive rest days → slight fitness decay
  Periodization— rotating muscle groups earns +0.15 reward bonus
  Adaptive DOMS— same muscle 3+ days cascades soreness ×4
  Weekly bonus — 7-day window with ≥2 rest, ≥2 strength/hypertrophy, ≥1 cardio → +0.30
  Injury       — RPE > 8 + CNS > 0.70 OR soreness > 0.75; stress adds +0.05 CNS overhead
  Continuous   — partial fitness gains produce partial rewards
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Constants ────────────────────────────────────────────────────────────────

FITNESS_MIN, FITNESS_MAX = 0.0, 100.0
FATIGUE_MIN, FATIGUE_MAX = 0.0, 1.0
SORENESS_MIN, SORENESS_MAX = 0.0, 1.0

VALID_MUSCLE_KEYS = frozenset({"legs", "push", "pull"})

# Recovery
REST_CNS_RECOVERY: float = 0.3
REST_SORENESS_RECOVERY: float = 0.4

# Injury thresholds
INJURY_RPE_THRESHOLD: int = 8
INJURY_CNS_THRESHOLD: float = 0.7
INJURY_SORENESS_THRESHOLD: float = 0.75

# Growth zone
GROWTH_RPE_LOW: int = 6
GROWTH_RPE_HIGH: int = 8
BASE_GROWTH_FITNESS_GAIN: float = 0.5
GROWTH_CNS_COST: float = 0.1
GROWTH_SORENESS_COST: float = 0.2

# Detraining
DETRAINING_REST_THRESHOLD: int = 3
DETRAINING_DECAY: float = 0.15

# Super-compensation CNS thresholds
SUPERCOMP_CNS_NORMAL: float = 0.05     # standard window
SUPERCOMP_CNS_GREAT_SLEEP: float = 0.10  # extended when sleep_quality >= 0.9
SUPERCOMP_FITNESS_MULTIPLIER: float = 1.5

# Periodization & weekly
PERIODIZATION_BONUS: float = 0.15
WEEKLY_PERIODIZATION_BONUS: float = 0.30

# Rewards
REWARD_REST: float = 0.1
REWARD_INJURY: float = -2.0
REWARD_GROWTH_BASE: float = 0.5

# HIIT
HIIT_CNS_COST: float = 0.20
HIIT_FITNESS_GAIN: float = 0.35
HIIT_SORENESS_COST: float = 0.15

# LISS
LISS_CNS_COST: float = 0.03
LISS_FITNESS_GAIN: float = 0.20
LISS_SORENESS_COST: float = 0.05

# Muscle-group mapping
_MUSCLE_KEY_MAP: Dict[str, list] = {
    "none": [],
    "legs": ["legs"],
    "push": ["push"],
    "pull": ["pull"],
    "full_body": ["legs", "push", "pull"],
}

# Nutrition multipliers
_NUTRITION_FITNESS_MULT = {
    "maintenance": 1.00,
    "surplus":     1.20,
    "deficit":     0.50,
    "high_protein":1.00,
}
_NUTRITION_SORENESS_MULT = {
    "maintenance": 1.00,
    "surplus":     1.00,
    "deficit":     1.00,
    "high_protein":0.80,  # easier soreness recovery
}
_NUTRITION_CNS_MULT = {
    "maintenance": 1.00,
    "surplus":     1.05,  # slight metabolic overhead
    "deficit":     0.90,  # less demand, faster CNS recovery
    "high_protein":1.00,
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

    # Sleep tracking
    sleep_quality: float = 0.8   # last night's sleep quality (0.0–1.0)

    # Internal tracking (not directly exposed)
    consecutive_rest_days: int = 0
    # training_streak: consecutive training days since the last REST day.
    # Resets to 0 on REST; increments on every training day.
    # This is the correct basis for days_since_last_rest in observations.
    training_streak: int = 0
    last_muscle_trained: Optional[str] = None
    consecutive_same_muscle: int = 0
    muscle_session_history: List[str] = field(default_factory=list)  # last 3 muscles
    week_history: List[str] = field(default_factory=list)  # last 7 workout categories


@dataclass
class TransitionResult:
    """Output of a single state transition."""

    next_state: PhysiologyState
    reward: float
    injured: bool


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return round(max(lo, min(hi, value)), 4)


def compute_weekly_variety_score(week_history: List[str]) -> float:
    """0.0–1.0 score of how many distinct modalities used in last 7 days."""
    if not week_history:
        return 0.0
    unique = len(set(week_history))
    return round(min(1.0, unique / 5.0), 3)


def _weekly_periodization_bonus(week_hist: List[str]) -> float:
    """Return +0.30 bonus reward if the last 7 days follow evidence-based structure."""
    if len(week_hist) < 7:
        return 0.0
    has_rest = week_hist.count("rest") >= 2
    strength_days = sum(1 for c in week_hist if c in ("hypertrophy", "strength"))
    has_strength = strength_days >= 2
    has_cardio = any(c in ("hiit", "liss_cardio") for c in week_hist)
    if has_rest and has_strength and has_cardio:
        return WEEKLY_PERIODIZATION_BONUS
    return 0.0


def _next_sleep_quality(
    workout_category: str,
    intensity_rpe: int,
    stress_event: bool,
    current_sleep: float,
) -> float:
    """Compute tonight's sleep quality based on today's activity."""
    base = 0.8
    if stress_event:
        base -= 0.3     # stress strongly impairs sleep
    if intensity_rpe >= 9:
        base -= 0.10    # very hard training → disrupted sleep
    elif intensity_rpe >= 7:
        base -= 0.05    # moderate training → slight impairment
    if workout_category == "rest":
        base += 0.10    # good rest day → better recovery sleep
    # Momentum: very poor or very good sleep tends to persist slightly
    base = 0.7 * base + 0.3 * current_sleep
    return _clamp(base, 0.1, 1.0)


# ── Engine ───────────────────────────────────────────────────────────────────

class PhysiologyEngine:
    """
    Stateless calculator for physiological state transitions.

    All methods are pure functions — no mutable engine state.

    Business rules (summary)
    -------------------------
    REST: CNS recovery = base × sleep_quality × (0.5 if stress else 1.0)
    Sleep: computed nightly; stress/hard RPE → lower; rest days → higher
    Nutrition: surplus +20% fitness, deficit ×0.5, high_protein −20% soreness
    Growth (RPE 6–8): super-comp window extended to cns≤0.10 if sleep_quality ≥ 0.9
    Weekly bonus: ≥2 rest + ≥2 strength/hypertrophy + ≥1 cardio in last 7 days → +0.30
    Detraining: 3+ consecutive rest → fitness −0.15/day
    Periodization: different muscle than last → +0.15 reward
    Adaptive DOMS: same muscle 3+ days → ×4 soreness cost
    Injury: RPE>8+CNS>0.70 OR soreness>0.75; stress event adds 0.05 CNS
    """

    @staticmethod
    def transition(
        state: PhysiologyState,
        workout_category: str,
        target_muscle: str,
        intensity_rpe: int,
        stress_event: bool = False,
        nutrition_protocol: str = "maintenance",
    ) -> TransitionResult:
        """
        Compute the next physiological state.

        Parameters
        ----------
        state               : Current physiology snapshot.
        workout_category    : rest | liss_cardio | hiit | hypertrophy | strength
        target_muscle       : none | legs | push | pull | full_body
        intensity_rpe       : 1–10 (Rate of Perceived Exertion)
        stress_event        : Life-stress today (bad sleep, work pressure)
        nutrition_protocol  : maintenance | surplus | deficit | high_protein
        """
        workout_category = workout_category.lower().strip()
        target_muscle = target_muscle.lower().strip()
        nutrition_protocol = nutrition_protocol.lower().strip()

        fitness = state.fitness_capacity
        cns = state.cns_fatigue
        soreness = dict(state.muscle_soreness)
        days_active = state.days_active
        consecutive_rest = state.consecutive_rest_days
        training_streak = state.training_streak
        last_muscle = state.last_muscle_trained
        consec_same = state.consecutive_same_muscle
        muscle_history = list(state.muscle_session_history)
        week_hist = list(state.week_history)
        sleep = state.sleep_quality

        affected_keys = _MUSCLE_KEY_MAP.get(target_muscle, [])

        # Compute tonight's sleep quality (propagated into next_state)
        next_sleep = _next_sleep_quality(workout_category, intensity_rpe, stress_event, sleep)

        # Update 7-day week history
        week_hist.append(workout_category)
        if len(week_hist) > 7:
            week_hist = week_hist[-7:]

        reward = 0.0

        # ── RULE 1: REST ──────────────────────────────────────────────────
        if workout_category == "rest":
            # CNS recovery: sleep quality × stress modifier
            cns_recovery = REST_CNS_RECOVERY * sleep * (0.5 if stress_event else 1.0)
            cns = _clamp(cns - cns_recovery, FATIGUE_MIN, FATIGUE_MAX)
            for key in soreness:
                soreness[key] = _clamp(
                    soreness[key] - REST_SORENESS_RECOVERY, SORENESS_MIN, SORENESS_MAX
                )
            consecutive_rest += 1
            training_streak = 0  # reset training streak on REST
            reward = REWARD_REST

            # Smart rest bonus: rewarded when resting is the correct call (high CNS)
            if state.cns_fatigue >= 0.6:
                reward += 0.2

            # Detraining: 3+ consecutive rest days
            if consecutive_rest >= DETRAINING_REST_THRESHOLD:
                fitness = _clamp(fitness - DETRAINING_DECAY, FITNESS_MIN, FITNESS_MAX)
                reward -= 0.05

            # Weekly periodization bonus (check after updating history)
            reward += _weekly_periodization_bonus(week_hist)

            next_state = PhysiologyState(
                fitness_capacity=_clamp(fitness, FITNESS_MIN, FITNESS_MAX),
                cns_fatigue=cns,
                muscle_soreness={k: _clamp(v, SORENESS_MIN, SORENESS_MAX) for k, v in soreness.items()},
                days_active=days_active,
                sleep_quality=next_sleep,
                consecutive_rest_days=consecutive_rest,
                training_streak=training_streak,
                last_muscle_trained=last_muscle,
                consecutive_same_muscle=consec_same,
                muscle_session_history=muscle_history,
                week_history=week_hist,
            )
            return TransitionResult(next_state=next_state, reward=reward, injured=False)

        # ── RULE 2: Injury check ──────────────────────────────────────────
        # Stress event adds CNS overhead before checking injury threshold
        if stress_event:
            cns = _clamp(cns + 0.05, FATIGUE_MIN, FATIGUE_MAX)

        rpe_cns_injury = (intensity_rpe > INJURY_RPE_THRESHOLD and cns > INJURY_CNS_THRESHOLD)
        sore_muscle_injury = any(
            soreness.get(k, 0.0) > INJURY_SORENESS_THRESHOLD for k in affected_keys
        )

        if rpe_cns_injury or sore_muscle_injury:
            next_state = PhysiologyState(
                fitness_capacity=_clamp(fitness, FITNESS_MIN, FITNESS_MAX),
                cns_fatigue=_clamp(cns, FATIGUE_MIN, FATIGUE_MAX),
                muscle_soreness={k: _clamp(v, SORENESS_MIN, SORENESS_MAX) for k, v in soreness.items()},
                days_active=days_active,
                sleep_quality=next_sleep,
                consecutive_rest_days=0,
                training_streak=0,
                last_muscle_trained=last_muscle,
                consecutive_same_muscle=0,
                muscle_session_history=muscle_history,
                week_history=week_hist,
            )
            return TransitionResult(next_state=next_state, reward=REWARD_INJURY, injured=True)

        # Reset consecutive rest counter; increment training streak
        consecutive_rest = 0
        training_streak += 1

        # ── Periodization bonus ───────────────────────────────────────────
        muscle_key = target_muscle if target_muscle != "none" else None
        periodization_bonus = 0.0
        if muscle_key and last_muscle and muscle_key != last_muscle:
            periodization_bonus = PERIODIZATION_BONUS

        # ── Adaptive DOMS: same muscle 3+ days ───────────────────────────
        if muscle_key and muscle_key == last_muscle:
            consec_same += 1
        else:
            consec_same = 0 if muscle_key else consec_same

        soreness_multiplier = 4.0 if consec_same >= 2 else 1.0

        # Update muscle history (last 3)
        if muscle_key:
            muscle_history.append(muscle_key)
            if len(muscle_history) > 3:
                muscle_history = muscle_history[-3:]

        # ── Nutrition multipliers ─────────────────────────────────────────
        nut_fit   = _NUTRITION_FITNESS_MULT.get(nutrition_protocol, 1.0)
        nut_sor   = _NUTRITION_SORENESS_MULT.get(nutrition_protocol, 1.0)
        nut_cns   = _NUTRITION_CNS_MULT.get(nutrition_protocol, 1.0)

        # ── Super-compensation window (extended by good sleep) ────────────
        supercomp_threshold = SUPERCOMP_CNS_GREAT_SLEEP if sleep >= 0.9 else SUPERCOMP_CNS_NORMAL
        supercomp = SUPERCOMP_FITNESS_MULTIPLIER if cns <= supercomp_threshold else 1.0

        # ── RULE 3: HIIT ──────────────────────────────────────────────────
        if workout_category == "hiit":
            cns = _clamp(cns + HIIT_CNS_COST * nut_cns, FATIGUE_MIN, FATIGUE_MAX)
            fitness_gain = HIIT_FITNESS_GAIN * nut_fit * (1 - cns * 0.5)
            fitness = _clamp(fitness + fitness_gain, FITNESS_MIN, FITNESS_MAX)
            for key in affected_keys:
                soreness[key] = _clamp(
                    soreness[key] + HIIT_SORENESS_COST * soreness_multiplier * nut_sor,
                    SORENESS_MIN, SORENESS_MAX,
                )
            gain = fitness - state.fitness_capacity
            reward = 0.3 + (gain / 2.0) + periodization_bonus
            days_active += 1

        # ── RULE 4: LISS Cardio ───────────────────────────────────────────
        elif workout_category == "liss_cardio":
            cns = _clamp(cns + LISS_CNS_COST * nut_cns, FATIGUE_MIN, FATIGUE_MAX)
            fitness = _clamp(fitness + LISS_FITNESS_GAIN * nut_fit, FITNESS_MIN, FITNESS_MAX)
            for key in affected_keys:
                soreness[key] = _clamp(
                    soreness[key] + LISS_SORENESS_COST * soreness_multiplier * nut_sor,
                    SORENESS_MIN, SORENESS_MAX,
                )
            reward = 0.15 + periodization_bonus
            days_active += 1

        # ── RULE 5: Growth (RPE 6–8) ──────────────────────────────────────
        elif GROWTH_RPE_LOW <= intensity_rpe <= GROWTH_RPE_HIGH:
            fitness_gain = BASE_GROWTH_FITNESS_GAIN * supercomp * nut_fit
            fitness = _clamp(fitness + fitness_gain, FITNESS_MIN, FITNESS_MAX)
            cns = _clamp(cns + GROWTH_CNS_COST * nut_cns, FATIGUE_MIN, FATIGUE_MAX)
            for key in affected_keys:
                soreness[key] = _clamp(
                    soreness[key] + GROWTH_SORENESS_COST * soreness_multiplier * nut_sor,
                    SORENESS_MIN, SORENESS_MAX,
                )
            gain = fitness - state.fitness_capacity
            reward = REWARD_GROWTH_BASE * (gain / BASE_GROWTH_FITNESS_GAIN) + periodization_bonus
            days_active += 1

        # ── RULE 6: Fallback (sub-optimal RPE, no injury) ─────────────────
        else:
            scale = intensity_rpe / 10.0
            cns = _clamp(cns + 0.05 * scale, FATIGUE_MIN, FATIGUE_MAX)
            for key in affected_keys:
                soreness[key] = _clamp(
                    soreness[key] + 0.1 * scale * soreness_multiplier * nut_sor,
                    SORENESS_MIN, SORENESS_MAX,
                )
            fitness = _clamp(fitness + 0.1 * scale * nut_fit, FITNESS_MIN, FITNESS_MAX)
            reward = 0.05 + periodization_bonus
            days_active += 1

        # Weekly periodization bonus (check after appending to week history)
        reward += _weekly_periodization_bonus(week_hist)

        next_state = PhysiologyState(
            fitness_capacity=_clamp(fitness, FITNESS_MIN, FITNESS_MAX),
            cns_fatigue=_clamp(cns, FATIGUE_MIN, FATIGUE_MAX),
            muscle_soreness={k: _clamp(v, SORENESS_MIN, SORENESS_MAX) for k, v in soreness.items()},
            days_active=days_active,
            sleep_quality=next_sleep,
            consecutive_rest_days=consecutive_rest,
            training_streak=training_streak,
            last_muscle_trained=muscle_key if muscle_key else last_muscle,
            consecutive_same_muscle=consec_same,
            muscle_session_history=muscle_history,
            week_history=week_hist,
        )
        return TransitionResult(next_state=next_state, reward=reward, injured=False)
