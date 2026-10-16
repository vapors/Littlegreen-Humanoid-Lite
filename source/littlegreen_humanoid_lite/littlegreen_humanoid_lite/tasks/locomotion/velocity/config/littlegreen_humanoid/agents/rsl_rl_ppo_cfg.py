"""Single canonical LittleGreen PPO recipe. Frozen v10.3.1 numeric values."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LittlegreenLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v10.3.1 numerical PPO settings. Fresh v11 init; guarded same-v11 resume is permitted."""
    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 250
    experiment_name = "littlegreen_v11_0_0_clean_core_bilateral_acquisition_fresh"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.80,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0060,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )
