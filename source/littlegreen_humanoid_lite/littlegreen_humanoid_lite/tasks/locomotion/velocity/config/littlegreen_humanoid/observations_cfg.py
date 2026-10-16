"""Shared 47-D actor observation contract for LittleGreen v2.4 foundation tasks.

All three canonical tasks expose the exact same tensor order, corruption, and
normalization.  Only the evolution of the final phase pair differs:

* Stand: randomized once per reset and held static.
* Transfer/March: continuous alternating phase independent of velocity command.
* Walk: command-synchronized continuous non-blocking phase.
"""

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
)

from . import mdp
from .base_cfg import FEET_BODY_PATTERN


OBSERVATION_LAYOUT_V24 = (
    "command3,base_ang_vel3,projected_gravity3,joint_pos_rel12,"
    "joint_vel12,previous_bounded_action12,phase_sin1,phase_cos1"
)
POLICY_OBSERVATION_COUNT_V24 = 47
CRITIC_OBSERVATION_COUNT_V24 = 50


@configclass
class LittlegreenVelocityObservationsCfg:
    """One shared observation class for Stand, Transfer, and Walk."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ACTIONABLE_JOINTS_V1_2_3,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.015, n_max=0.015),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ACTIONABLE_JOINTS_V1_2_3,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.40, n_max=0.40),
        )
        actions = ObsTerm(
            func=mdp.bounded_last_action,
            params={"action_name": "joint_pos"},
        )
        gait_phase = ObsTerm(
            func=mdp.littlegreen_phase_sin_cos_v24,
            params={
                "phase_mode": "random_static",
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=FEET_BODY_PATTERN,
                ),
                "linear_command_threshold": 0.08,
                "yaw_command_threshold": 0.08,
                "period_s": 0.90,
                "transition_fraction": 0.10,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


def set_phase_mode(
    observations: LittlegreenVelocityObservationsCfg,
    phase_mode: str,
) -> None:
    """Change only phase evolution; preserve the shared 47-D layout."""
    if phase_mode not in {
        "random_static",
        "neutral_static",
        "continuous_transfer",
        "command_synchronized",
    }:
        raise ValueError(f"Unsupported v2.4 phase mode: {phase_mode}")
    observations.policy.gait_phase.params["phase_mode"] = phase_mode
    observations.critic.gait_phase.params["phase_mode"] = phase_mode
