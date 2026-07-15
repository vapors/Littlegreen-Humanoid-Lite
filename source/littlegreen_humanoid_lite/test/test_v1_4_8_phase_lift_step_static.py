
"""Static checks for v1.4.8 phase-lift step refinement."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v145.py"
REW = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py"
CURR = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py"
INIT = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
PPO = ROOT / "littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/agents/rsl_rl_ppo_cfg.py"


def test_v148_task_registered():
    text = INIT.read_text()
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v8" in text
    assert "LilgreenHardwareST3215LoadedV148PhaseLiftStepEnvCfg" in text
    assert "LilgreenHardwareST3215LoadedV148PhaseLiftStepPPORunnerCfg" in text


def test_v148_rewards_present():
    text = REW.read_text()
    for name in [
        "phase_guided_expected_swing_clearance_reward",
        "phase_guided_swing_clearance_shortfall_penalty",
        "phase_guided_step_placement_target_reward",
        "phase_guided_step_shortfall_penalty",
        "phase_guided_rocking_no_step_penalty",
        "moving_yaw_stability_penalty",
    ]:
        assert name in text


def test_v148_config_uses_lift_and_place_terms():
    text = ENV.read_text()
    assert "HardwareV148PhaseLiftStepRewardsCfg" in text
    assert "target_clearance_m" in text
    assert "min_clearance_m" in text
    assert "target_step_m" in text
    assert "moving_yaw_stability" in text
    assert "st3215_loaded_v148_phase_lift_step_hardware_stage_curriculum" in CURR.read_text()
    assert "lilgreen_v1_4_8_st3215_phase_lift_step" in PPO.read_text()
