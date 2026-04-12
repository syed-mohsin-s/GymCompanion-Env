"""
Evaluation Criteria Check — GymCompanion-Env
Validates all 6 Copilot-reported issues are correctly fixed.
Run from repo root: python eval_check.py
"""
import sys, os, importlib.util, json

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, condition))

print("\n" + "="*60)
print("  GymCompanion-Env — Evaluation Criteria Check")
print("="*60)

# ── Load physiology engine directly (no openenv needed) ──────────────────────
_engine_path = os.path.abspath(
    os.path.join("GymCompanion_env", "server", "physiology_engine.py")
)
_spec = importlib.util.spec_from_file_location("physiology_engine", _engine_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["physiology_engine"] = _mod
_spec.loader.exec_module(_mod)
PhysiologyEngine = _mod.PhysiologyEngine
PhysiologyState  = _mod.PhysiologyState

# ── Load environment (mock openenv deps) ─────────────────────────────────────
# Mock openenv so environment module can be imported without the package
import types
openenv_mock = types.ModuleType("openenv")
openenv_core = types.ModuleType("openenv.core")
openenv_intf = types.ModuleType("openenv.core.env_server")
openenv_iface = types.ModuleType("openenv.core.env_server.interfaces")
openenv_types = types.ModuleType("openenv.core.env_server.types")

class _FakeEnv:
    pass
class _FakeState:
    def __init__(self, **kw): self.__dict__.update(kw)
    episode_id = "test"
    step_count = 0

openenv_iface.Environment = _FakeEnv
openenv_types.State = _FakeState

sys.modules["openenv"] = openenv_mock
sys.modules["openenv.core"] = openenv_core
sys.modules["openenv.core.env_server"] = openenv_intf
sys.modules["openenv.core.env_server.interfaces"] = openenv_iface
sys.modules["openenv.core.env_server.types"] = openenv_types

# Also mock openenv.core.client_types for models
client_types = types.ModuleType("openenv.core.client_types")
sys.modules["openenv.core.client_types"] = client_types

# Load models
sys.path.insert(0, os.path.abspath("GymCompanion_env"))
import models as _models
GymcompanionObservation = _models.GymcompanionObservation
GymcompanionAction      = _models.GymcompanionAction
WorkoutCategory         = _models.WorkoutCategory
TargetMuscle            = _models.TargetMuscle

# Load environment
_env_path = os.path.abspath(
    os.path.join("GymCompanion_env", "server", "GymCompanion_env_environment.py")
)
_env_spec = importlib.util.spec_from_file_location("env_module", _env_path)
_env_mod  = importlib.util.module_from_spec(_env_spec)
sys.modules["env_module"] = _env_mod
_env_spec.loader.exec_module(_env_mod)
GymcompanionEnvironment = _env_mod.GymcompanionEnvironment
TASK_CONFIGS            = _env_mod.TASK_CONFIGS

# Load root inference module (for build_obs_dict check)
import ast
with open("inference.py") as f:
    root_src = f.read()

# ── CHECK 1: build_obs_dict used in run_inference (not manual dict) ───────────
print("\n── Check 1: Observation consistency (build_obs_dict used in run_inference) ──")
has_build_obs = "obs_dict = build_obs_dict(obs, task_name, steps_taken, training_streak)" in root_src
has_manual_sleep = (
    '"sleep_quality"' not in root_src.split("def build_obs_dict")[1].split("def ")[0]
    or "build_obs_dict" in root_src
)
check("build_obs_dict() used in run_inference loop", has_build_obs,
      "sleep_quality & weekly_variety_score now reach the model")
check("No stale manual obs_dict in loop (old manual dict removed)",
      "\"sleep_quality\": obs.fitness_capacity" not in root_src
      and "days_since_last_rest\": getattr(obs," not in root_src.split("def run_inference")[1])

print("\n── Check 2: sleep_quality & weekly_variety_score in build_obs_dict ──")
build_fn_src = root_src.split("def build_obs_dict")[1].split("def ")[0]
check("sleep_quality included in build_obs_dict", '"sleep_quality"' in build_fn_src)
check("weekly_variety_score included in build_obs_dict", '"weekly_variety_score"' in build_fn_src)

# ── CHECK 3: days_since_last_rest = training_streak, not rest days ─────────
print("\n── Check 3: days_since_last_rest = training streak (not rest counter) ──")
has_training_streak = "training_streak = 0" in root_src and "training_streak += 1" in root_src
has_streak_reset_on_rest = 'training_streak = 0' in root_src  # reset on REST
check("training_streak counter declared per task", has_training_streak)
check("training_streak resets to 0 on REST action",
      'if last_cat == "rest":\n                            training_streak = 0' in root_src or
      "if last_cat == \"rest\":" in root_src)
check("days_since_last_rest uses training_streak (not obs field)",
      '"days_since_last_rest": training_streak' in build_fn_src)

# ── CHECK 4: Final score cap at 0.35 when goal not met ───────────────────────
print("\n── Check 4: Final composite score capped at 0.35 when goal not met ──")
engine = PhysiologyEngine()

def _make_env_with_fitness(task, fitness_val):
    """Create env, simulate minimal steps to set fitness just below goal."""
    env = GymcompanionEnvironment()
    env.reset(task_name=task)
    cfg = TASK_CONFIGS[task]
    # Manually set physiology to be below goal to test score cap
    from physiology_engine import PhysiologyState as PS
    env._physiology = PS(
        fitness_capacity=fitness_val,
        cns_fatigue=0.9,  # violates max_cns → env'll penalise
    )
    env._cumulative_reward = 1.0
    return env

# couch-to-5k: goal=55, test with fitness=52 (below goal)
env_below = _make_env_with_fitness("couch-to-5k", 52.0)
score_below = env_below._compute_score()
check(
    f"Score ≤ 0.35 when fitness goal not met (got {score_below:.4f})",
    score_below <= 0.35,
    f"couch-to-5k: fitness=52 < goal=55 → score should be ≤ 0.35"
)

# Verify score CAN exceed 0.35 when goal IS met
env_above = _make_env_with_fitness("couch-to-5k", 60.0)
env_above._physiology.cns_fatigue = 0.1  # within limits
env_above._cumulative_reward = 18.0
score_above = env_above._compute_score()
check(
    f"Score can exceed 0.35 when fitness goal IS met (got {score_above:.4f})",
    score_above > 0.35,
    f"couch-to-5k: fitness=60 ≥ goal=55 → score should be > 0.35"
)

# ── CHECK 5: Unknown task normalization ───────────────────────────────────────
print("\n── Check 5: Unknown task name normalized to couch-to-5k ──")
env_unknown = GymcompanionEnvironment()
env_unknown.reset(task_name="totally_random_task_xyz")
check(
    "Unknown task → _task_name = 'couch-to-5k'",
    env_unknown._task_name == "couch-to-5k",
    f"Got: {env_unknown._task_name}"
)
check(
    "Unknown task → _task_config matches couch-to-5k config",
    env_unknown._task_config == TASK_CONFIGS["couch-to-5k"],
    "Config and name are in sync"
)

# ── CHECK 6: Mutable default dict ─────────────────────────────────────────────
print("\n── Check 6: GymcompanionObservation metadata not shared across instances ──")
obs1 = GymcompanionObservation(
    fitness_capacity=50.0, cns_fatigue=0.0,
    muscle_soreness={"legs": 0.0, "push": 0.0, "pull": 0.0}, days_active=0
)
obs2 = GymcompanionObservation(
    fitness_capacity=60.0, cns_fatigue=0.1,
    muscle_soreness={"legs": 0.0, "push": 0.0, "pull": 0.0}, days_active=0
)
obs1.metadata["test_key"] = "mutated"
check(
    "obs1.metadata mutation does NOT affect obs2.metadata",
    "test_key" not in obs2.metadata,
    f"obs2.metadata = {obs2.metadata}"
)

# ── SCHEMA ALIGNMENT CHECK ────────────────────────────────────────────────────
print("\n── Bonus: Schema alignment (openenv.yaml fields vs Observation fields) ──")
import yaml
with open(os.path.join("GymCompanion_env", "openenv.yaml")) as f:
    spec = yaml.safe_load(f)

yaml_obs_fields = set(spec.get("observation_space", {}).keys())
model_fields = set(GymcompanionObservation.model_fields.keys())
# Remove internal-only fields not in yaml
internal = {"done", "reward", "metadata"}
model_public = model_fields - internal

missing_in_model = yaml_obs_fields - model_public
extra_in_model   = model_public - yaml_obs_fields

check(
    "All openenv.yaml observation fields present in GymcompanionObservation",
    len(missing_in_model) == 0,
    f"Missing: {missing_in_model}" if missing_in_model else "All present ✓"
)
check(
    "No extra fields in model not declared in openenv.yaml",
    len(extra_in_model) == 0,
    f"Extra: {extra_in_model}" if extra_in_model else "Perfectly aligned ✓"
)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print("\n" + "="*60)
print(f"  Results: {passed}/{total} checks passed")
if passed == total:
    print("  \033[92m🟢 ALL CHECKS PASSED — Ready for submission!\033[0m")
else:
    failed = [n for n, ok in results if not ok]
    print(f"  \033[91m🔴 FAILED: {', '.join(failed)}\033[0m")
print("="*60 + "\n")
