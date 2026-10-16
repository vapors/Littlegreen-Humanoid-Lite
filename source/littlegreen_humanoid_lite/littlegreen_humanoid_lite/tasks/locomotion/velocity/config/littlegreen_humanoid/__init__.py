"""Canonical task registry: validated source implementation only. Future skills are planned, not registered."""
import gymnasium as gym

TASK_ID = "Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110"
gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": __name__ + ".tasks.locomotion:LittlegreenLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": __name__ + ".agents.rsl_rl_ppo_cfg:LittlegreenLocomotionPPORunnerCfg",
    },
)
