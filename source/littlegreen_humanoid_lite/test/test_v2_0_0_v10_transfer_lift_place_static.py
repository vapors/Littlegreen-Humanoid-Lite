"""Static contract checks for Littlegreen Hardware-ST3215-Loaded-v10."""

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


def _reward_term_names(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    reward_cfg = _class(tree, class_name)
    names: list[str] = []
    for node in reward_cfg.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "RewTerm":
                names.append(node.targets[0].id)
    return names


def test_v10_task_is_registered_additively():
    text = (TASK_DIR / "__init__.py").read_text()
    assert 'id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v10"' in text
    assert "LilgreenHardwareST3215LoadedV10TransferLiftPlaceEnvCfg" in text
    assert "LilgreenHardwareST3215LoadedV10PPORunnerCfg" in text
    assert 'id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v9"' in text


def test_v10_reward_set_is_standalone_and_contains_only_direct_gait_terms():
    path = TASK_DIR / "env_cfg_hardware_st3215_loaded_v10.py"
    tree = ast.parse(path.read_text())
    reward_cfg = _class(tree, "V10TransferLiftPlaceRewardsCfg")
    assert reward_cfg.bases == []
    terms = _reward_term_names(path, "V10TransferLiftPlaceRewardsCfg")
    assert len(terms) == 24
    assert "signed_phase_contact" in terms
    assert "support_force_transfer" in terms
    assert "phase_com_shift" in terms
    assert "swing_clearance_trajectory" in terms
    assert "clearance_gated_placement" in terms
    assert "feet_air_time" not in terms
    assert "moving_planted_no_progress" not in terms
    assert "phase_swing_clearance_shortfall" not in terms
    assert "phase_step_shortfall" not in terms


def test_v10_uses_command_synchronized_47d_phase_semantics():
    text = (TASK_DIR / "env_cfg_hardware_st3215_loaded_v10.py").read_text()
    assert "V10CommandSynchronizedObservationsCfg" in text
    assert "command_synchronized_gait_phase_sin_cos" in text
    assert "HardwareObservationsCfg" in text
    assert "gait_phase = ObsTerm" in text


def test_v10_phase_is_reset_on_movement_onset_and_alternates_first_swing():
    text = (MDP_DIR / "rewards.py").read_text()
    assert "def _v10_command_synchronized_phase" in text
    assert "onset = moving & (~was_moving)" in text
    assert "next_first_swing_left = torch.where(onset, ~next_first_swing_left" in text
    assert "phase = torch.where(moving, phase, torch.zeros_like(phase))" in text
    assert "_littlegreen_v10_phase_last_common_step" in text
    assert "current_episode_length < last_episode_length" in text


def test_v10_transition_window_is_one_sided_and_condensed():
    text = (MDP_DIR / "rewards.py").read_text()
    assert "half_phase = torch.remainder(2.0 * phase, 1.0)" in text
    assert "half_transition_fraction = 2.0 * active_transition_fraction" in text
    assert "transition = half_phase < half_transition_fraction" in text
    assert "double support per full cycle" in text
    curriculum = (MDP_DIR / "curriculums.py").read_text()
    assert "transition_fractions: tuple[float, float, float, float] = (0.08, 0.07, 0.06, 0.05)" in curriculum
    assert '"double_support_fraction_total": float(2.0 * transition_fractions[stage])' in curriculum


def test_v10_zero_lift_and_zero_step_have_zero_reward_baselines():
    text = (MDP_DIR / "rewards.py").read_text()
    assert "def v10_swing_clearance_trajectory_reward" in text
    assert "zero_baseline = torch.exp" in text
    assert "(raw - zero_baseline)" in text
    assert "def v10_clearance_gated_placement_reward" in text
    assert "clearance_gate" in text
    assert "active_scale * normalized * progress_gate * clearance_gate" in text


def test_v10_curriculum_is_condensed_to_5000_iteration_go_no_go():
    text = (MDP_DIR / "curriculums.py").read_text()
    assert "def st3215_loaded_v10_transfer_lift_place_curriculum" in text
    assert "(96000, 192000, 268800)" in text
    assert "iterations 1500" in text
    assert "3000" in text
    assert "4200" in text
    assert "(0.25, 0.55)" in text
    assert "(-0.08, 0.08)" in text
    assert "(-0.22, 0.22)" in text
    assert '"moving_height_target_m": 0.455' in text
    assert '"push_xy_abs_max": 0.0' in text


def test_v10_raises_moving_com_and_moves_it_forward():
    text = (TASK_DIR / "env_cfg_hardware_st3215_loaded_v10.py").read_text()
    assert "V10_MOVING_BASE_COM_HEIGHT_M = 0.455" in text
    assert "V10_INITIAL_COM_TARGET_FORWARD_M = 0.080" in text
    curriculum = (MDP_DIR / "curriculums.py").read_text()
    assert "(0.080, 0.082, 0.085, 0.085)" in curriculum


def test_v10_disables_no_progress_termination_during_skill_acquisition():
    path = TASK_DIR / "env_cfg_hardware_st3215_loaded_v10.py"
    tree = ast.parse(path.read_text())
    term_cfg = _class(tree, "V10TransferLiftPlaceTerminationsCfg")
    assigned = {
        node.targets[0].id
        for node in term_cfg.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert "moving_no_progress" not in assigned
    assert "moving_no_support" in assigned


def test_v10_preserves_v5s3_action_contract_and_stand3500_warm_start():
    text = (TASK_DIR / "env_cfg_hardware_st3215_loaded_v10.py").read_text()
    assert "LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg" in text
    assert "ACTIONABLE_JOINTS_V1_2_3" in text
    assert "HARDWARE_LOWER_LIMIT_RAD" in text
    assert "HARDWARE_UPPER_LIMIT_RAD" in text
    script = (ROOT / "scripts/rsl_rl/train_v10_from_stand3500.sh").read_text()
    assert "model_3500.pt" in script
    assert "--policy_only_warm_start" in script
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v10" in script
    assert 'MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"' in script


def test_v10_export_metadata_flags_revised_phase_semantics():
    text = (ROOT / "scripts/rsl_rl/export_policy_littlegreen.py").read_text()
    assert '"observation_contract_name": "47d_command_synchronized_phase_v10"' in text
    assert '"gait_phase_semantics": "command_synchronized_alternating_first_swing"' in text
    assert '"deployment_requires_command_synchronized_phase_v10": True' in text


def test_v10_analyzer_entry_point_is_present():
    text = (ROOT / "scripts/rsl_rl/analyze_policy_v10.py").read_text()
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v10" in text
    assert "analyze_policy_v10.py" in text
