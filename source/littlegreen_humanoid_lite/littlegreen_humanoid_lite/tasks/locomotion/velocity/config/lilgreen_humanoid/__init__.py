import gymnasium as gym

from . import (
    agents,
    env_cfg,
    env_cfg_hardware,
    env_cfg_hardware_st3215,
    env_cfg_hardware_st3215_loaded,
    env_cfg_hardware_st3215_loaded_v141,
    env_cfg_hardware_st3215_loaded_v142,
    env_cfg_hardware_st3215_loaded_v143,
    env_cfg_hardware_st3215_loaded_v144,
    env_cfg_hardware_st3215_loaded_v145,
    env_cfg_hardware_st3215_loaded_v9,
    env_cfg_hardware_st3215_loaded_v10,
    env_cfg_stand,
    env_cfg_stand_st3215,
    env_cfg_stand_st3215_loaded,
)


# Legacy v1.1 baseline. Kept unchanged so historical checkpoints remain reproducible.
gym.register(
    id="Velocity-Lilgreen-Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.LilgreenHumanoidEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHumanoidPPORunnerCfg,
    },
)

# v1.2.3 standing baseline with symmetric residual action contract v3. Same 45-D actor observation and 12-D action dimensions
# as Hardware-v0 so actor/critic checkpoints can be continued directly.
gym.register(
    id="Velocity-Lilgreen-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_stand.LilgreenStandEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandPPORunnerCfg,
    },
)

# v1.2.3 hardware-oriented locomotion curriculum using the same action contract v3.
gym.register(
    id="Velocity-Lilgreen-Hardware-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware.LilgreenHardwareEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwarePPORunnerCfg,
    },
)


# v1.3.0 actuator-aware standing branch. v1.2.3 tasks above are intentionally frozen.
gym.register(
    id="Velocity-Lilgreen-Stand-ST3215-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_stand_st3215.LilgreenStandST3215EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandST3215PPORunnerCfg,
    },
)

# v1.3.0 actuator-aware locomotion curriculum. The 45-D/12-D policy interface and
# action contract v3 are shared with Stand-ST3215-v0.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215.LilgreenHardwareST3215EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215PPORunnerCfg,
    },
)


# v1.4.0 loaded-actuator standing branch. This preserves the v1.3.1 Stage-A
# model and adds the Track 2 loaded standing-transition calibration envelope.
gym.register(
    id="Velocity-Lilgreen-Stand-ST3215-Loaded-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_stand_st3215_loaded.LilgreenStandST3215LoadedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandST3215LoadedPPORunnerCfg,
    },
)

# v1.4.0 loaded-actuator locomotion curriculum. The policy interface remains
# action contract v3 with the same 45-D observation and 12-D action dimensions.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded.LilgreenHardwareST3215LoadedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedPPORunnerCfg,
    },
)


# v1.4.1 safer loaded-actuator Hardware curriculum. This is additive so v1.4.0
# Hardware-ST3215-Loaded-v0 remains reproducible.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v141.LilgreenHardwareST3215LoadedV141EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV141PPORunnerCfg,
    },
)


# v1.4.2 locomotion-pressure loaded-actuator Hardware curriculum. This is additive so
# v1.4.0/v1.4.1 Hardware tasks remain reproducible.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v142.LilgreenHardwareST3215LoadedV142EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV142PPORunnerCfg,
    },
)

# v1.4.3 move-now loaded-actuator Hardware curriculum. This is additive so
# v1.4.0/v1.4.1/v1.4.2 Hardware tasks remain reproducible.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v143.LilgreenHardwareST3215LoadedV143EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV143PPORunnerCfg,
    },
)

# v1.4.4 alternating-step loaded-actuator Hardware curriculum. This is additive so
# v1.4.0/v1.4.1/v1.4.2/v1.4.3 Hardware tasks remain reproducible.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v144.LilgreenHardwareST3215LoadedV144EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV144PPORunnerCfg,
    },
)
# v1.4.5 athletic Stand task. This is a new q_default/vector-residual profile,
# intended to be trained fresh before Hardware-v5.
gym.register(
    id="Velocity-Lilgreen-Stand-ST3215-Loaded-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenStandST3215LoadedV145EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandST3215LoadedV145PPORunnerCfg,
    },
)

# v1.4.5 athletic Hardware curriculum. Uses the same v1.4.5 q_default/vector
# residual profile as Stand-v5, with grounded alternating-step rewards.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV145EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV145PPORunnerCfg,
    },
)



# v1.4.5 Stand stabilization branch. Same contract v4/vector residual family, but
# a moderated athletic q_default, 0.43-0.44 m stand target, stand-only anti-lean
# terms, and gentle balance perturbations.
gym.register(
    id="Velocity-Lilgreen-Stand-ST3215-Loaded-v5s",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenStandST3215LoadedV145StabilizedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandST3215LoadedV145StabilizedPPORunnerCfg,
    },
)

# Matching Hardware task for the stabilized v1.4.5 Stand seed.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV145StabilizedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV145StabilizedPPORunnerCfg,
    },
)

# v1.4.5s2 Stand forward-COM stabilization branch. Same contract v4/vector
# residual family and q_default as v5s, but with 0.45 m height, a 5.5 cm
# forward COM-over-feet band, and a weak forward-lean cue.
gym.register(
    id="Velocity-Lilgreen-Stand-ST3215-Loaded-v5s2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenStandST3215LoadedV145StabilizedForwardEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandST3215LoadedV145StabilizedForwardPPORunnerCfg,
    },
)

# Matching Hardware task for the v5s2 forward-COM Stand seed.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV145StabilizedForwardEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV145StabilizedForwardPPORunnerCfg,
    },
)


# v1.4.5s3 Stand forward-COM/height refinement branch. Same contract v4/vector
# residual family and q_default as v5s2, but with 0.460 m height, a 7 cm
# forward COM-over-feet band, and a slightly stronger forward-lean cue.
gym.register(
    id="Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenStandST3215LoadedV145StabilizedForward2EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenStandST3215LoadedV145StabilizedForward2PPORunnerCfg,
    },
)

# Matching Hardware task for the v5s3 taller forward-COM Stand seed.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV145StabilizedForward2EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV145StabilizedForward2PPORunnerCfg,
    },
)

# v1.4.6 anti-planted locomotion. Same v5s3 contract/profile and stand seed,
# but the Hardware objective now makes planted no-progress under moving commands
# terminate early and trains with a non-standing command floor.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV146AntiPlantedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV146AntiPlantedPPORunnerCfg,
    },
)



# v1.4.7 phase-guided alternating gait scaffold.  This is a deliberate
# observation-contract change: v4/vector residual actions plus a 2-D gait phase
# observation for 47-D policy input.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV147PhaseGuidedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV147PhaseGuidedPPORunnerCfg,
    },
)


# v1.4.8 phase-lift step refinement.  Observation contract is unchanged from
# v1.4.7: 47-D with gait phase.  Rewards now require explicit swing clearance,
# commanded-direction foot placement, anti-rocking, and yaw stability.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v145.LilgreenHardwareST3215LoadedV148PhaseLiftStepEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV148PhaseLiftStepPPORunnerCfg,
    },
)

# Littlegreen v2.0.0 v9 gait acquisition.  Preserves the v5s3 action/actuator
# contract and v7/v8 47-D phase observation, but uses reset-safe terminations,
# a standalone slim reward set, and a gentle forward-only staged curriculum.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v9.LilgreenHardwareST3215LoadedV9GaitAcquisitionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV9PPORunnerCfg,
    },
)


# Littlegreen v2.0.0 v10 command-synchronized transfer/lift/place acquisition.
# Preserves the v5s3 action/actuator contract and 47-D observation shape, but
# revises the phase semantics and gait scaffold for a condensed 5k go/no-go run.
gym.register(
    id="Velocity-Lilgreen-Hardware-ST3215-Loaded-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_hardware_st3215_loaded_v10.LilgreenHardwareST3215LoadedV10TransferLiftPlaceEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LilgreenHardwareST3215LoadedV10PPORunnerCfg,
    },
)
