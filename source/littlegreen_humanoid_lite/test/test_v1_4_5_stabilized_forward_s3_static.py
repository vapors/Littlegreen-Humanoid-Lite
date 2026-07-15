from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/hardware_contract.py"
ENV = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v145.py"
INIT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
PPO = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/agents/rsl_rl_ppo_cfg.py"
ANALYZER = ROOT / "scripts/rsl_rl/analyze_policy_v1_4_5.py"


def test_v145s3_forward_height_constants():
    text = CONTRACT.read_text()
    assert "V1_4_5_STABILIZED_FORWARD2_STAND_BASE_COM_HEIGHT_M = 0.460" in text
    assert "V1_4_5_STABILIZED_FORWARD2_MOVING_BASE_COM_HEIGHT_M = 0.430" in text
    assert "V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M = 0.070" in text
    assert "V1_4_5_STABILIZED_FORWARD2_COM_BAND_HALF_WIDTH_M = 0.010" in text
    assert "V1_4_5_STABILIZED_FORWARD2_LEAN_PROJECTED_GRAVITY_X = 0.065" in text


def test_v145s3_tasks_and_experiment_registered():
    env = ENV.read_text()
    assert "HardwareV145StabilizedForward2StandRewardsCfg" in env
    assert "V1_4_5_STABILIZED_FORWARD2_COM_TARGET_X_M" in env
    init = INIT.read_text()
    assert "Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3" in init
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s3" in init
    ppo = PPO.read_text()
    assert "lilgreen_v1_4_5_st3215_stabilized_forward_s3" in ppo


def test_v145s3_analyzer_hotfix_included():
    text = ANALYZER.read_text()
    assert "standing_com_forward_offset_chunks: list[np.ndarray] = []" in text
    assert "projected_gravity_x_chunks: list[np.ndarray] = []" in text
    assert "collected {step + 1}" in text
