"""Static checks for v1.4.3 move-now gait-pressure Hardware curriculum."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v143.py"
CURR = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py"
REG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
AGENT = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/agents/rsl_rl_ppo_cfg.py"
REWARDS = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py"
DIAG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/diagnostics.py"
ANALYZER = ROOT.parents[1] / "scripts/rsl_rl/analyze_policy_v1_4_3.py"


def test_v143_task_is_registered_additively():
    text = REG.read_text()
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v0" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v1" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v2" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v3" in text
    assert "LilgreenHardwareST3215LoadedV143EnvCfg" in text
    assert "LilgreenHardwareST3215LoadedV143PPORunnerCfg" in AGENT.read_text()


def test_v143_curriculum_keeps_continuous_commands_without_bins():
    text = CURR.read_text()
    assert "st3215_loaded_v143_hardware_stage_curriculum" in text
    assert "standing_fractions: tuple[float, float, float, float] = (0.45, 0.35, 0.28, 0.22)" in text
    assert "(-0.25, 0.25)" in text
    assert "(-0.85, 0.85)" in text
    assert '"curriculum_profile_id": 143.0' in text
    fn_text = text[text.index("def st3215_loaded_v143_hardware_stage_curriculum"):]
    assert "command_bins" not in fn_text
    assert "forward_only" not in fn_text


def test_v143_action_model_is_faster_training_response_only():
    text = CFG.read_text()
    assert "V143ST3215LoadedFastActionsCfg" in text
    assert "response_delay_scale=0.60" in text
    assert "_scaled(ST3215_TAU_MEDIAN_S, 0.78)" in text
    assert "_scaled_curves(ST3215_PEAK_VELOCITY_CURVES_RAD_S, 1.12)" in text
    assert "stage_b_loaded_v143_fast_train" in text
    assert "RESIDUAL_ACTION_SCALE_RAD_V1_2_3" in text


def test_v143_rewards_add_anti_brace_gait_pressure():
    text = CFG.read_text()
    rewards = REWARDS.read_text()
    for name in (
        "moving_velocity_along_command",
        "moving_no_progress_l1",
        "moving_single_support_time",
        "moving_double_support_penalty",
        "swing_foot_clearance",
        "swing_foot_velocity_along_command",
    ):
        assert name in text
        assert f"def {name}" in rewards
    assert 'params={"command_name": "base_velocity", "std": 0.30}' in text
    assert "weight=3.4" in text


def test_v143_leg_lift_diagnostics_are_present():
    diag = DIAG.read_text()
    analyzer = ANALYZER.read_text()
    for token in (
        "moving_single_support_fraction",
        "moving_double_support_fraction",
        "moving_swing_clearance_mean_m",
        "moving_swing_foot_forward_velocity_mean_mps",
        "foot_lift_count_total",
    ):
        assert token in analyzer or token in diag
    assert "--eval_force_command" in analyzer
