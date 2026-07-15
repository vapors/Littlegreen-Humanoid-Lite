"""Static checks for v1.4.2 locomotion-pressure Hardware curriculum and forced-command eval."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v142.py"
CURR = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py"
REG = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
AGENT = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/agents/rsl_rl_ppo_cfg.py"
TRAIN_EVAL = ROOT.parents[1] / "scripts/rsl_rl/train_eval.py"
PLAY_JOYSTICK = ROOT.parents[1] / "scripts/rsl_rl/play_joystick.py"
COMMAND_OVERRIDE = ROOT.parents[1] / "scripts/rsl_rl/command_override.py"


def test_v142_task_is_registered_additively():
    text = REG.read_text()
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v0" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v1" in text
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v2" in text
    assert "LilgreenHardwareST3215LoadedV142EnvCfg" in text
    assert "LilgreenHardwareST3215LoadedV142PPORunnerCfg" in AGENT.read_text()


def test_v142_curriculum_pushes_locomotion_pressure_without_bins():
    text = CURR.read_text()
    assert "st3215_loaded_v142_hardware_stage_curriculum" in text
    assert "stage_step_boundaries: tuple[int, int, int] = (32000, 128000, 256000)" in text
    assert "standing_fractions: tuple[float, float, float, float] = (0.50, 0.38, 0.30, 0.25)" in text
    assert "(-0.22, 0.22)" in text
    assert "(-0.80, 0.80)" in text
    assert "(-1.00, 1.00)" in text
    assert '"curriculum_profile_id": 142.0' in text
    # Keep the experiment simple: no command-bin task split in this curriculum.
    fn_text = text[text.index("def st3215_loaded_v142_hardware_stage_curriculum"):]
    assert "command_bins" not in fn_text
    assert "forward_only" not in fn_text


def test_v142_rewards_sharpen_tracking_and_reduce_standing_stickiness():
    text = CFG.read_text()
    assert 'params={"command_name": "base_velocity", "std": 0.35}' in text
    assert "weight=3.0" in text
    assert "weight=-1.35" in text  # stand_base_xy_speed less sticky than v1.4.1
    assert "weight=-0.45" in text  # stand_default_pose close to original Hardware reward
    assert "weight=0.85" in text   # stand_base_height between v1.4.0 and v1.4.1
    assert "knee_soft_torque_utilization" in text
    assert "weight=-0.160" in text # raw_action_excess safeguard retained


def test_forced_command_eval_and_joystick_marker_override_are_present():
    train = TRAIN_EVAL.read_text()
    play = PLAY_JOYSTICK.read_text()
    helper = COMMAND_OVERRIDE.read_text()
    assert "--eval_force_command" in train
    assert "apply_velocity_command_override" in train
    assert "format_command_suffix" in train
    assert "--no_command_manager_override" in play
    assert "apply_velocity_command_override" in play
    assert "vel_command_b" in helper
    assert "is_standing_env" in helper
    assert "is_heading_env" in helper
