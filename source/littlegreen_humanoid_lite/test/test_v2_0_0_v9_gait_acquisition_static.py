"""Static contract checks for Littlegreen Hardware-ST3215-Loaded-v9."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = (
    ROOT
    / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid"
)
MDP_DIR = (
    ROOT
    / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp"
)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class not found: {name}")


def test_v9_task_is_registered_additively():
    text = (TASK_DIR / "__init__.py").read_text()
    assert 'id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v9"' in text
    assert "LilgreenHardwareST3215LoadedV9GaitAcquisitionEnvCfg" in text
    assert "LilgreenHardwareST3215LoadedV9PPORunnerCfg" in text


def test_v9_reward_set_is_standalone_and_has_22_terms():
    tree = ast.parse((TASK_DIR / "env_cfg_hardware_st3215_loaded_v9.py").read_text())
    reward_cfg = _class(tree, "V9SlimGaitRewardsCfg")
    assert reward_cfg.bases == []
    terms = []
    for node in reward_cfg.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "RewTerm":
                terms.append(node.targets[0].id)
    assert len(terms) == 22
    assert "feet_air_time" not in terms
    assert "moving_planted_no_progress" not in terms
    assert "phase_swing_clearance" in terms
    assert "phase_foot_placement" in terms


def test_v9_curriculum_matches_gentle_requested_envelope():
    text = (MDP_DIR / "curriculums.py").read_text()
    assert "def st3215_loaded_v9_gait_acquisition_curriculum" in text
    assert "(128000, 320000, 512000)" in text
    assert "(0.25, 0.55)" in text
    assert "(-0.08, 0.08)" in text
    assert "(-0.22, 0.22)" in text
    assert '"push_xy_abs_max": 0.0' in text


def test_stateful_terminations_are_episode_reset_aware():
    text = (MDP_DIR / "terminations.py").read_text()
    assert "def _episode_reset_mask" in text
    assert "torch.where(reset_mask" in text
    assert "_littlegreen_v147_had_support" in text
    assert "arming_timeout_s" in text


def test_policy_only_warm_start_zero_initializes_added_inputs():
    text = (ROOT / "scripts/rsl_rl/train_eval.py").read_text()
    assert "expanded = torch.zeros_like(current[key])" in text
    assert "new input columns zero-initialized" in text


def test_v9_reuses_v5s3_action_profile_and_47d_observations():
    text = (TASK_DIR / "env_cfg_hardware_st3215_loaded_v9.py").read_text()
    assert "LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg" in text
    assert "HardwareV147PhaseGuidedObservationsCfg" in text
    assert "ACTIONABLE_JOINTS_V1_2_3" in text
    assert "HARDWARE_LOWER_LIMIT_RAD" in text
    assert "HARDWARE_UPPER_LIMIT_RAD" in text
