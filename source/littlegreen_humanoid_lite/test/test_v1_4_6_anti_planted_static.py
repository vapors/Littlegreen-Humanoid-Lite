from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/env_cfg_hardware_st3215_loaded_v145.py"
INIT = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/__init__.py"
PPO = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/lilgreen_humanoid/agents/rsl_rl_ppo_cfg.py"
REWARDS = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py"
TERMS = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/terminations.py"
CURR = ROOT / "source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py"
TRAIN = ROOT / "scripts/rsl_rl/train_eval.py"


def test_v146_task_and_ppo_registered():
    assert "Velocity-Lilgreen-Hardware-ST3215-Loaded-v6" in INIT.read_text()
    assert "LilgreenHardwareST3215LoadedV146AntiPlantedEnvCfg" in ENV.read_text()
    assert "lilgreen_v1_4_6_st3215_anti_planted" in PPO.read_text()


def test_v146_anti_planted_terms_exist():
    env = ENV.read_text()
    assert "moving_planted_no_progress" in env
    assert "moving_long_double_support" in env
    assert "moving_no_progress = DoneTerm" in env
    assert "command_floor" in env
    assert "\"floor_mps\": 0.22" in env
    assert "def moving_planted_no_progress_penalty" in REWARDS.read_text()
    assert "def moving_long_double_support_penalty" in REWARDS.read_text()
    assert "def moving_no_progress_timeout" in TERMS.read_text()
    assert "def enforce_nonstanding_command_floor" in CURR.read_text()
    assert "floor_mps: float = 0.22" in CURR.read_text()


def test_v146_policy_only_warm_start_cli():
    text = TRAIN.read_text()
    assert "--policy_only_warm_start" in text
    assert "--resume_log_root" in text
    assert "_load_policy_only_warm_start" in text
    assert "left critic/optimizer fresh" in text
