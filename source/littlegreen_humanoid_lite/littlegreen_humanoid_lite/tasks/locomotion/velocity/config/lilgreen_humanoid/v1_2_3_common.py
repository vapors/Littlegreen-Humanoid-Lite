"""Shared configuration blocks for Berkeley Humanoid Lite v1.2.3 tasks."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from littlegreen_humanoid_lite_assets.robots.lilgreen_humanoid import LILGREEN_CFG


FEET_BODY_PATTERN = ".*_ankle_roll"


@configclass
class V123ResidualActionsCfg:
    """v1.2.3 symmetric residual action contract around q_default."""

    joint_pos = mdp.BoundedDefaultResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_2_3,
        preserve_order=True,
        one_step_delay_probability=0.0,
    )


@configclass
class StandCommandsCfg:
    """Standing-heavy command distribution with only gentle motion commands."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(8.0, 12.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.75,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.15),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(-0.25, 0.25),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class HardwareCommandsCfg:
    """Initial Hardware-v0 command distribution; curriculum expands it in stages."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(8.0, 12.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.75,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.15),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(-0.25, 0.25),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class StandObservationsCfg:
    """45-D deployment-compatible observation with conservative initial sensor noise."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.bounded_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HardwareObservationsCfg:
    """45-D deployment-compatible observation with moderate hardware-oriented noise."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
            noise=Unoise(n_min=-0.015, n_max=0.015),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
            noise=Unoise(n_min=-0.40, n_max=0.40),
        )
        actions = ObsTerm(func=mdp.bounded_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class V123TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class StandEventsCfg:
    """Gentle randomization for balance discovery and quiet standing."""

    set_hardware_joint_limits = EventTerm(
        func=mdp.set_joint_position_limits,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
            "lower_limits": HARDWARE_LOWER_LIMIT_RAD,
            "upper_limits": HARDWARE_UPPER_LIMIT_RAD,
        },
        mode="startup",
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.1),
            "dynamic_friction_range": (0.75, 1.05),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 32,
            "make_consistent": True,
        },
        mode="startup",
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
        mode="startup",
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
        mode="startup",
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )


@configclass
class HardwareEventsCfg:
    """Fixed material buckets plus staged reset/mass/gain/push/latency curriculum."""

    set_hardware_joint_limits = EventTerm(
        func=mdp.set_joint_position_limits,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
            "lower_limits": HARDWARE_LOWER_LIMIT_RAD,
            "upper_limits": HARDWARE_UPPER_LIMIT_RAD,
        },
        mode="startup",
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.65, 1.20),
            "dynamic_friction_range": (0.60, 1.15),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
        },
        mode="startup",
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
        mode="reset",
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
        mode="reset",
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
        mode="interval",
        interval_range_s=(8.0, 12.0),
    )


@configclass
class V123HardwareAlignedEnvCfg(LocomotionVelocityEnvCfg):
    """Common post-init for v1.2.3 50 Hz hardware-aligned tasks."""

    def __post_init__(self):
        super().__post_init__()

        # 200 Hz physics / 50 Hz policy and action update.
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.render_interval = self.decimation

        robot_cfg = LILGREEN_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        robot_cfg.soft_joint_pos_limit_factor = 1.0
        for name, actuator in robot_cfg.actuators.items():
            if name == "toes":
                continue
            actuator.effort_limit_sim = ST3215_PEAK_TORQUE_NM
            actuator.effort_limit = ST3215_PEAK_TORQUE_NM
            actuator.velocity_limit_sim = ST3215_NO_LOAD_SPEED_RAD_S
            actuator.velocity_limit = ST3215_NO_LOAD_SPEED_RAD_S
        self.scene.robot = robot_cfg
