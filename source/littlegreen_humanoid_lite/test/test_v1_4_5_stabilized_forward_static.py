from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/hardware_contract.py"
REWARDS = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/hardware_rewards.py"
ENV = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v145.py"
INIT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
PPO = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/agents/rsl_rl_ppo_cfg.py"
ANALYZER = ROOT / "scripts/rsl_rl/analyze_policy_v1_4_5.py"


def test_v145s2_forward_stand_constants():
    text = CONTRACT.read_text()
    assert "V1_4_5_STABILIZED_FORWARD_STAND_BASE_COM_HEIGHT_M = 0.450" in text
    assert "V1_4_5_STABILIZED_FORWARD_MOVING_BASE_COM_HEIGHT_M = 0.420" in text
    assert "V1_4_5_STABILIZED_FORWARD_COM_TARGET_X_M = 0.055" in text
    assert "V1_4_5_STABILIZED_FORWARD_COM_BAND_HALF_WIDTH_M = 0.010" in text
    assert "V1_4_5_STABILIZED_FORWARD_LEAN_PROJECTED_GRAVITY_X = 0.052" in text


def test_v145s2_forward_terms_and_tasks_registered():
    rewards = REWARDS.read_text()
    assert "standing_com_forward_over_feet_band_l2" in rewards
    assert "standing_forward_lean_projected_gravity_exp" in rewards
    env = ENV.read_text()
    assert "HardwareV145StabilizedForwardStandRewardsCfg" in env
    assert "standing_com_forward_over_feet_band_l2" in env
    assert "standing_forward_lean_projected_gravity_exp" in env
    init = INIT.read_text()
    assert "Velocity-Lilgreen-Stand-ST3215-Loaded-v5s2" in init
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s2" in init
    ppo = PPO.read_text()
    assert "lilgreen_v1_4_5_st3215_stabilized_forward" in ppo


def test_v145s2_analyzer_reports_forward_com_and_lean():
    text = ANALYZER.read_text()
    assert "standing_com_forward_offset_mean_m" in text
    assert "standing_projected_gravity_x_mean" in text
