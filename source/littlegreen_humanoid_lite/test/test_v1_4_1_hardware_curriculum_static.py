"""Static checks for the additive v1.4.1 Hardware curriculum task."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v141.py"
CURR = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py"
REG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"


def test_v141_task_is_registered_additively():
    text = REG.read_text()
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v0" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v1" in text
    assert "LilgreenHardwareST3215LoadedV141EnvCfg" in text


def test_v141_curriculum_uses_slower_stage_boundaries_and_ranges():
    text = CURR.read_text()
    assert "st3215_loaded_v141_hardware_stage_curriculum" in text
    assert "stage_step_boundaries: tuple[int, int, int] = (64000, 192000, 320000)" in text
    assert "standing_fractions: tuple[float, float, float, float] = (0.85, 0.70, 0.55, 0.45)" in text
    assert "lin_vel_x_ranges" in text and "(-0.60, 0.60)" in text
    assert "push_xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.10)" in text


def test_v141_rewards_strengthen_retention_and_action_health():
    text = CFG.read_text()
    assert "raw_action_excess_l2" in text and "weight=-0.180" in text
    assert "knee_soft_torque_utilization" in text
    assert 'joint_names=[".*_knee_pitch_joint"]' in text
    assert "weight=-0.75" in text  # stand_default_pose
    assert "weight=1.00" in text   # base height / both-feet contact


def test_v141_curriculum_returns_only_numeric_episode_scalars():
    text = CURR.read_text()
    fn_start = text.index("def st3215_loaded_v141_hardware_stage_curriculum")
    fn_text = text[fn_start:]
    assert '"curriculum_profile":' not in fn_text
    assert '"curriculum_profile_id": 141.0' in fn_text
    assert 'v1.4.1_slow_retention' not in fn_text.split('return {', 1)[1]
