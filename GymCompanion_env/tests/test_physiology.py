# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for the GymCompanion PhysiologyEngine.

Run with:
    pytest tests/test_physiology.py -v
"""

import sys
import os
import importlib.util

# Load physiology_engine directly without triggering server/__init__.py
# (which imports openenv and would fail outside the uv venv)
_engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "server", "physiology_engine.py")
)
_spec = importlib.util.spec_from_file_location("physiology_engine", _engine_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
# Must register in sys.modules BEFORE exec so dataclasses can resolve __module__
sys.modules["physiology_engine"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore

PhysiologyEngine = _mod.PhysiologyEngine
PhysiologyState = _mod.PhysiologyState

import pytest  # noqa: E402 — must come after physiology_engine setup


@pytest.fixture
def engine():
    return PhysiologyEngine()


@pytest.fixture
def fresh_state():
    return PhysiologyState(fitness_capacity=50.0, cns_fatigue=0.0,
                           muscle_soreness={"legs": 0.0, "push": 0.0, "pull": 0.0})


# ── REST tests ────────────────────────────────────────────────────────────────

class TestRest:
    def test_rest_reduces_cns(self, engine):
        # CNS recovery = REST_CNS_RECOVERY (0.30) × sleep_quality (0.8 default) = 0.24
        # So cns: 0.60 - 0.24 = 0.36
        state = PhysiologyState(cns_fatigue=0.6)
        result = engine.transition(state, "rest", "none", 1)
        assert result.next_state.cns_fatigue == pytest.approx(0.36, abs=0.001)

    def test_rest_reduces_all_soreness(self, engine):
        state = PhysiologyState(muscle_soreness={"legs": 0.5, "push": 0.3, "pull": 0.4})
        result = engine.transition(state, "rest", "none", 1)
        assert result.next_state.muscle_soreness["legs"] == pytest.approx(0.1, abs=0.001)
        assert result.next_state.muscle_soreness["push"] == 0.0
        assert result.next_state.muscle_soreness["pull"] == 0.0

    def test_rest_base_reward(self, engine, fresh_state):
        result = engine.transition(fresh_state, "rest", "none", 1)
        assert result.reward == pytest.approx(0.1, abs=0.001)

    def test_smart_rest_bonus_when_cns_high(self, engine):
        state = PhysiologyState(cns_fatigue=0.8)
        result = engine.transition(state, "rest", "none", 1)
        assert result.reward >= 0.3  # base 0.1 + smart rest bonus 0.2

    def test_not_injured_on_rest(self, engine, fresh_state):
        result = engine.transition(fresh_state, "rest", "none", 1)
        assert result.injured is False

    def test_detraining_after_3_consecutive_rest(self, engine):
        state = PhysiologyState(fitness_capacity=60.0, consecutive_rest_days=3)
        result = engine.transition(state, "rest", "none", 1)
        assert result.next_state.fitness_capacity < 60.0  # fitness decays

    def test_no_detraining_before_threshold(self, engine):
        # consecutive_rest_days=1 → becomes 2 after REST → no detraining (threshold is 3)
        state = PhysiologyState(fitness_capacity=60.0, consecutive_rest_days=1)
        result = engine.transition(state, "rest", "none", 1)
        assert result.next_state.fitness_capacity == pytest.approx(60.0, abs=0.01)


# ── Injury tests ──────────────────────────────────────────────────────────────

class TestInjury:
    def test_injury_high_rpe_high_cns(self, engine):
        state = PhysiologyState(cns_fatigue=0.8)
        result = engine.transition(state, "strength", "legs", 9)
        assert result.injured is True
        assert result.reward == pytest.approx(-2.0, abs=0.001)

    def test_injury_sore_muscle_over_threshold(self, engine):
        # 0.8 > 0.75 threshold
        state = PhysiologyState(muscle_soreness={"legs": 0.8, "push": 0.0, "pull": 0.0})
        result = engine.transition(state, "hypertrophy", "legs", 6)
        assert result.injured is True
        assert result.reward == pytest.approx(-2.0, abs=0.001)

    def test_no_injury_when_soreness_just_under_threshold(self, engine):
        state = PhysiologyState(muscle_soreness={"legs": 0.7, "push": 0.0, "pull": 0.0})
        result = engine.transition(state, "hypertrophy", "legs", 6)
        assert result.injured is False

    def test_injury_rehab_start_state_triggers_injury(self, engine):
        """Injury-rehab initial state (legs=0.85) must trigger injury if legs trained."""
        state = PhysiologyState(cns_fatigue=0.5,
                                muscle_soreness={"legs": 0.85, "push": 0.1, "pull": 0.1})
        result = engine.transition(state, "hypertrophy", "legs", 6)
        assert result.injured is True

    def test_no_injury_training_different_muscle_while_one_sore(self, engine):
        """Training push while legs are very sore is safe."""
        state = PhysiologyState(muscle_soreness={"legs": 0.9, "push": 0.1, "pull": 0.0})
        result = engine.transition(state, "hypertrophy", "push", 7)
        assert result.injured is False


# ── Growth tests ──────────────────────────────────────────────────────────────

class TestGrowth:
    def test_growth_increases_fitness(self, engine, fresh_state):
        result = engine.transition(fresh_state, "hypertrophy", "push", 7)
        assert result.next_state.fitness_capacity > fresh_state.fitness_capacity

    def test_growth_increases_cns(self, engine, fresh_state):
        result = engine.transition(fresh_state, "hypertrophy", "push", 7)
        assert result.next_state.cns_fatigue > fresh_state.cns_fatigue

    def test_super_compensation_on_full_recovery(self, engine):
        """After full CNS recovery (≤0.05), should get 1.5x fitness gain."""
        state_fresh = PhysiologyState(fitness_capacity=50.0, cns_fatigue=0.0)
        state_fatigued = PhysiologyState(fitness_capacity=50.0, cns_fatigue=0.3)
        r_fresh = engine.transition(state_fresh, "hypertrophy", "push", 7)
        r_fatigued = engine.transition(state_fatigued, "hypertrophy", "push", 7)
        assert r_fresh.next_state.fitness_capacity > r_fatigued.next_state.fitness_capacity


# ── Periodization bonus tests ─────────────────────────────────────────────────

class TestPeriodization:
    def test_periodization_bonus_when_alternating(self, engine):
        state = PhysiologyState(last_muscle_trained="push")
        result = engine.transition(state, "hypertrophy", "pull", 7)
        # Should include periodization bonus
        result_no_bonus = engine.transition(PhysiologyState(last_muscle_trained="pull"),
                                            "hypertrophy", "pull", 7)
        assert result.reward > result_no_bonus.reward

    def test_no_periodization_bonus_same_muscle(self, engine):
        state = PhysiologyState(last_muscle_trained="push")
        result = engine.transition(state, "hypertrophy", "push", 7)
        state2 = PhysiologyState(last_muscle_trained="legs")
        result2 = engine.transition(state2, "hypertrophy", "push", 7)
        # result2 should have periodization bonus, result should not
        assert result2.reward > result.reward


# ── HIIT vs LISS tests ────────────────────────────────────────────────────────

class TestHIITvsLISS:
    def test_hiit_higher_cns_cost_than_liss(self, engine, fresh_state):
        r_hiit = engine.transition(fresh_state, "hiit", "full_body", 7)
        r_liss = engine.transition(fresh_state, "liss_cardio", "full_body", 5)
        assert r_hiit.next_state.cns_fatigue > r_liss.next_state.cns_fatigue

    def test_liss_has_minimal_cns_cost(self, engine, fresh_state):
        result = engine.transition(fresh_state, "liss_cardio", "none", 4)
        assert result.next_state.cns_fatigue < 0.1

    def test_hiit_increases_fitness(self, engine, fresh_state):
        result = engine.transition(fresh_state, "hiit", "full_body", 7)
        assert result.next_state.fitness_capacity > fresh_state.fitness_capacity


# ── Adaptive soreness tests ───────────────────────────────────────────────────

class TestAdaptiveSoreness:
    def test_same_muscle_twice_amplifies_soreness(self, engine):
        state = PhysiologyState(consecutive_same_muscle=2, last_muscle_trained="push",
                                muscle_soreness={"legs": 0.0, "push": 0.2, "pull": 0.0})
        result = engine.transition(state, "hypertrophy", "push", 7)
        # Soreness should be significantly amplified
        assert result.next_state.muscle_soreness["push"] > 0.5


# ── Score bounds ──────────────────────────────────────────────────────────────

class TestScoreBounds:
    def test_reward_values_are_finite(self, engine, fresh_state):
        for cat in ["rest", "liss_cardio", "hiit", "hypertrophy", "strength"]:
            for muscle in ["none", "legs", "push", "pull", "full_body"]:
                for rpe in [1, 5, 7, 9]:
                    result = engine.transition(fresh_state, cat, muscle, rpe)
                    assert -10 < result.reward < 10, f"Unexpected reward for {cat}/{muscle}/rpe={rpe}"

    def test_fitness_stays_in_bounds(self, engine, fresh_state):
        state = fresh_state
        for _ in range(50):
            result = engine.transition(state, "hypertrophy", "push", 7)
            state = result.next_state
            assert 0.0 <= state.fitness_capacity <= 100.0

    def test_cns_stays_in_bounds(self, engine, fresh_state):
        state = fresh_state
        for _ in range(20):
            result = engine.transition(state, "strength", "legs", 9)
            if result.injured:
                break
            state = result.next_state
            assert 0.0 <= state.cns_fatigue <= 1.0

    def test_soreness_stays_in_bounds(self, engine, fresh_state):
        state = fresh_state
        for _ in range(10):
            result = engine.transition(state, "hypertrophy", "legs", 8)
            if result.injured:
                break
            state = result.next_state
            for v in state.muscle_soreness.values():
                assert 0.0 <= v <= 1.0


# ── Stress event tests ────────────────────────────────────────────────────────

class TestStressEvent:
    def test_stress_event_halves_cns_recovery_on_rest(self, engine):
        """Stress during rest = only half CNS recovery."""
        state = PhysiologyState(cns_fatigue=0.6)
        r_normal = engine.transition(state, "rest", "none", 1, stress_event=False)
        r_stress = engine.transition(state, "rest", "none", 1, stress_event=True)
        # Stress should result in higher CNS (less recovery)
        assert r_stress.next_state.cns_fatigue > r_normal.next_state.cns_fatigue

    def test_stress_event_adds_cns_overhead_on_training(self, engine, fresh_state):
        """Stress during training adds extra CNS cost."""
        r_normal = engine.transition(fresh_state, "hypertrophy", "push", 7, stress_event=False)
        r_stress = engine.transition(fresh_state, "hypertrophy", "push", 7, stress_event=True)
        assert r_stress.next_state.cns_fatigue > r_normal.next_state.cns_fatigue

    def test_stress_event_can_push_into_injury_zone(self, engine):
        """High CNS state + stress event overhead can cross injury threshold."""
        # cns=0.68, stress adds 0.05 → cns=0.73 > 0.7 threshold, RPE=9 > 8
        state = PhysiologyState(cns_fatigue=0.68)
        r_no_stress = engine.transition(state, "strength", "legs", 9, stress_event=False)
        r_stress = engine.transition(state, "strength", "legs", 9, stress_event=True)
        # No stress: 0.68 < 0.7 threshold → no injury
        # Stress: 0.68 + 0.05 = 0.73 > 0.7 → injury
        assert r_stress.injured is True
        assert r_no_stress.injured is False

    def test_stress_does_not_affect_soreness_recovery(self, engine):
        """Soreness recovery on REST is unaffected by stress events."""
        state = PhysiologyState(muscle_soreness={"legs": 0.5, "push": 0.3, "pull": 0.4})
        r_normal = engine.transition(state, "rest", "none", 1, stress_event=False)
        r_stress = engine.transition(state, "rest", "none", 1, stress_event=True)
        # Soreness recovery should be the same
        assert r_normal.next_state.muscle_soreness == r_stress.next_state.muscle_soreness

    def test_no_stress_default_behavior_unchanged(self, engine, fresh_state):
        """Default stress_event=False must produce same result as before."""
        r1 = engine.transition(fresh_state, "hypertrophy", "push", 7)
        r2 = engine.transition(fresh_state, "hypertrophy", "push", 7, stress_event=False)
        assert r1.next_state.fitness_capacity == r2.next_state.fitness_capacity
        assert r1.next_state.cns_fatigue == r2.next_state.cns_fatigue
        assert r1.reward == r2.reward


# ── Cross-feature integration tests ──────────────────────────────────────────

class TestCrossFeature:
    def test_stress_plus_high_soreness_is_extra_risky(self, engine):
        """Stress during high-soreness training scenario should still detect injury."""
        state = PhysiologyState(
            muscle_soreness={"legs": 0.78, "push": 0.0, "pull": 0.0},
        )
        result = engine.transition(state, "hypertrophy", "legs", 7, stress_event=True)
        assert result.injured is True  # soreness > threshold regardless of stress

    def test_full_recovery_supercomp_with_stress_at_rest(self, engine):
        """Super-compensation window (cns≤0.05) still applies even after stress rest."""
        # Start fresh, apply stress rest day → CNS should still be ≤ 0.05 baseline
        state = PhysiologyState(cns_fatigue=0.0)
        r_rest = engine.transition(state, "rest", "none", 1, stress_event=True)
        # After stress rest, CNS = 0.0 (was already 0) → super-comp window still open
        assert r_rest.next_state.cns_fatigue == pytest.approx(0.0, abs=0.001)
        r_train = engine.transition(r_rest.next_state, "hypertrophy", "push", 7)
        # Should still get super-compensation (1.5x gain)
        r_no_rest = engine.transition(
            PhysiologyState(cns_fatigue=0.3, fitness_capacity=50.0),
            "hypertrophy", "push", 7
        )
        assert r_train.next_state.fitness_capacity > r_no_rest.next_state.fitness_capacity

    def test_all_categories_work_with_stress(self, engine, fresh_state):
        """All 5 workout categories should work without runtime errors under stress."""
        categories = ["rest", "liss_cardio", "hiit", "hypertrophy", "strength"]
        for cat in categories:
            result = engine.transition(fresh_state, cat, "none", 5, stress_event=True)
            assert isinstance(result.reward, float)
            assert isinstance(result.injured, bool)

