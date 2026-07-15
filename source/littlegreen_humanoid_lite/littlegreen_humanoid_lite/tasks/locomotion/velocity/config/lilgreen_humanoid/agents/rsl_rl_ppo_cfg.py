from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class LilgreenHumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "humanoid"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims = [256, 128, 128],
        critic_hidden_dims = [256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        #entropy_coef=0.01,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LilgreenHumanoidPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48  # slightly longer rollouts for stability
    max_iterations = 3000
    save_interval = 100
    experiment_name = "humanoid"
    empirical_normalization = True  # normalize observations/rewards
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # slightly higher exploration for smoke test
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,  # conservative for smoke test
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,  # slightly faster learning
        max_grad_norm=1.0,
    )


@configclass
class LilgreenStandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v1.2.3 standing-first PPO configuration.

    Stand-v0 and Hardware-v0 intentionally share the same experiment root and network
    architecture so Hardware-v0 can resume directly from a selected Stand-v0 checkpoint.
    """

    num_steps_per_env = 64
    max_iterations = 10000
    save_interval = 250
    experiment_name = "lilgreen_v1_2_3"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LilgreenHardwarePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v1.2.3 final hardware-oriented locomotion PPO configuration."""

    num_steps_per_env = 64
    max_iterations = 50000
    save_interval = 500
    experiment_name = "lilgreen_v1_2_3"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )



@configclass
class LilgreenStandST3215PPORunnerCfg(LilgreenStandPPORunnerCfg):
    """v1.3.0 fresh Stand-ST3215 PPO configuration."""

    experiment_name = "lilgreen_v1_3_0_st3215"


@configclass
class LilgreenHardwareST3215PPORunnerCfg(LilgreenHardwarePPORunnerCfg):
    """v1.3.0 ST3215-aware hardware locomotion PPO configuration."""

    experiment_name = "lilgreen_v1_3_0_st3215"


@configclass
class LilgreenStandST3215LoadedPPORunnerCfg(LilgreenStandPPORunnerCfg):
    """v1.4.0 fresh loaded-actuator Stand PPO configuration."""

    experiment_name = "lilgreen_v1_4_0_st3215_loaded"


@configclass
class LilgreenHardwareST3215LoadedPPORunnerCfg(LilgreenHardwarePPORunnerCfg):
    """v1.4.0 loaded-actuator hardware locomotion PPO configuration."""

    experiment_name = "lilgreen_v1_4_0_st3215_loaded"


@configclass
class LilgreenHardwareST3215LoadedV141PPORunnerCfg(LilgreenHardwareST3215LoadedPPORunnerCfg):
    """v1.4.1 safer Hardware curriculum PPO configuration.

    The experiment root intentionally remains ``lilgreen_v1_4_0_st3215_loaded`` so
    model_6500.pt and the v1.4.0 Stand model_5000.pt can be loaded without moving
    checkpoints. Use a v141-specific run_name to keep runs easy to inspect.
    """

    experiment_name = "lilgreen_v1_4_0_st3215_loaded"
    save_interval = 500


@configclass
class LilgreenHardwareST3215LoadedV142PPORunnerCfg(LilgreenHardwareST3215LoadedPPORunnerCfg):
    """v1.4.2 locomotion-pressure Hardware curriculum PPO configuration.

    The experiment root intentionally remains ``lilgreen_v1_4_0_st3215_loaded`` so
    the successful v1.4.0 Stand model_5000.pt can be loaded without moving
    checkpoints. Use a v142-specific run_name to keep runs easy to inspect.
    """

    experiment_name = "lilgreen_v1_4_0_st3215_loaded"
    save_interval = 500

@configclass
class LilgreenHardwareST3215LoadedV143PPORunnerCfg(LilgreenHardwareST3215LoadedPPORunnerCfg):
    """v1.4.3 move-now Hardware curriculum PPO configuration.

    The experiment root intentionally remains ``lilgreen_v1_4_0_st3215_loaded`` so
    the successful v1.4.0 Stand model_5000.pt can be loaded without moving
    checkpoints. Use a v143-specific run_name to keep runs easy to inspect.
    """

    experiment_name = "lilgreen_v1_4_0_st3215_loaded"
    save_interval = 500

@configclass
class LilgreenHardwareST3215LoadedV144PPORunnerCfg(LilgreenHardwareST3215LoadedPPORunnerCfg):
    """v1.4.4 alternating-step Hardware curriculum PPO configuration.

    The experiment root intentionally remains ``lilgreen_v1_4_0_st3215_loaded`` so
    the successful v1.4.0 Stand model_5000.pt can be loaded without moving
    checkpoints. Use a v144-specific run_name to keep runs easy to inspect.
    """

    experiment_name = "lilgreen_v1_4_0_st3215_loaded"
    save_interval = 500

@configclass
class LilgreenStandST3215LoadedV145PPORunnerCfg(LilgreenStandST3215LoadedPPORunnerCfg):
    """v1.4.5 athletic Stand PPO configuration.

    The experiment root changes because the q_default/residual-scale profile is
    deployment-impacting and should not be mixed with v1.4.0 scalar-residual
    checkpoints.
    """

    experiment_name = "lilgreen_v1_4_5_st3215_athletic"
    save_interval = 500


@configclass
class LilgreenHardwareST3215LoadedV145PPORunnerCfg(LilgreenHardwareST3215LoadedPPORunnerCfg):
    """v1.4.5 athletic Hardware PPO configuration.

    Resume this from a v1.4.5 Stand-v5 checkpoint, not from the old v1.4.0
    Stand model_5000, because q_default and residual scales are different.
    """

    experiment_name = "lilgreen_v1_4_5_st3215_athletic"
    save_interval = 500



@configclass
class LilgreenStandST3215LoadedV145StabilizedPPORunnerCfg(LilgreenStandST3215LoadedV145PPORunnerCfg):
    """v1.4.5 stabilized Stand PPO configuration.

    Same contract generation as v5, but a separate experiment root prevents
    mixing the first deep-athletic Stand-v5 checkpoints with stabilized runs.
    """

    experiment_name = "lilgreen_v1_4_5_st3215_stabilized"
    save_interval = 500


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedPPORunnerCfg(LilgreenHardwareST3215LoadedV145PPORunnerCfg):
    """v1.4.5 stabilized Hardware PPO configuration.

    Resume this from Velocity-Lilgreen-Stand-ST3215-Loaded-v5s, not the first
    v5 deep-athletic Stand run.
    """

    experiment_name = "lilgreen_v1_4_5_st3215_stabilized"
    save_interval = 500


@configclass
class LilgreenStandST3215LoadedV145StabilizedForwardPPORunnerCfg(LilgreenStandST3215LoadedV145StabilizedPPORunnerCfg):
    """v1.4.5s2 forward-COM Stand PPO configuration."""

    experiment_name = "lilgreen_v1_4_5_st3215_stabilized_forward"
    save_interval = 500


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedForwardPPORunnerCfg(LilgreenHardwareST3215LoadedV145StabilizedPPORunnerCfg):
    """v1.4.5s2 Hardware PPO configuration.

    Resume this from Velocity-Lilgreen-Stand-ST3215-Loaded-v5s2.
    """

    experiment_name = "lilgreen_v1_4_5_st3215_stabilized_forward"
    save_interval = 500


@configclass
class LilgreenStandST3215LoadedV145StabilizedForward2PPORunnerCfg(LilgreenStandST3215LoadedV145StabilizedForwardPPORunnerCfg):
    """v1.4.5s3 taller forward-COM Stand PPO configuration."""

    experiment_name = "lilgreen_v1_4_5_st3215_stabilized_forward_s3"
    save_interval = 500


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedForward2PPORunnerCfg(LilgreenHardwareST3215LoadedV145StabilizedForwardPPORunnerCfg):
    """v1.4.5s3 Hardware PPO configuration.

    Resume this from Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3.
    """

    experiment_name = "lilgreen_v1_4_5_st3215_stabilized_forward_s3"
    save_interval = 500

@configclass
class LilgreenHardwareST3215LoadedV146AntiPlantedPPORunnerCfg(LilgreenHardwareST3215LoadedV145StabilizedForward2PPORunnerCfg):
    """v1.4.6 anti-planted Hardware PPO configuration.

    Use --policy_only_warm_start from a v5s3 Stand checkpoint and keep this run in
    its own experiment root so it does not mix with stand-only stabilization runs.
    """

    experiment_name = "lilgreen_v1_4_6_st3215_anti_planted"
    save_interval = 500



@configclass
class LilgreenHardwareST3215LoadedV147PhaseGuidedPPORunnerCfg(LilgreenHardwareST3215LoadedV146AntiPlantedPPORunnerCfg):
    """v1.4.7 phase-guided alternating Hardware PPO configuration.

    Observation contract is 47-D because it adds [sin, cos] gait phase. Use
    --policy_only_warm_start from a v5s3 Stand checkpoint; train_eval.py will
    partially expand the actor input layer and reset critic/optimizer.
    """

    experiment_name = "lilgreen_v1_4_7_st3215_phase_guided"
    save_interval = 500



@configclass
class LilgreenHardwareST3215LoadedV148PhaseLiftStepPPORunnerCfg(LilgreenHardwareST3215LoadedV147PhaseGuidedPPORunnerCfg):
    """v1.4.8 phase-lift/foot-placement Hardware PPO configuration.

    Observation contract remains the v1.4.7 47-D phase-guided input.  Use
    --policy_only_warm_start from the best v1.4.7 checkpoint; critic and optimizer
    should be reset because clearance/placement/anti-rocking rewards are new.
    """

    experiment_name = "lilgreen_v1_4_8_st3215_phase_lift_step"
    save_interval = 500
