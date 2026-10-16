"""Shared manager-based foundation configuration for LittleGreen v2.4.

This module preserves the working Isaac Sim 5.0 / Isaac Lab manager-based API,
measured ST3215 actuator stack, 200 Hz physics, 50 Hz policy rate, calibrated
joint order, physical limits, and q_default.

Canonical skills are separated by command distribution, phase evolution,
rewards, and modest reset/disturbance curricula:

* Stand: zero command and randomized-static phase.
* Transfer/March: continuous alternating phase.
* Walk: command-synchronized continuous phase.
"""

from __future__ import annotations

import math

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
    athletic_stabilized_default_joint_pos_dict,
)
from littlegreen_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)
from littlegreen_humanoid_lite_assets.robots.lilgreen_humanoid import LILGREEN_CFG

from . import mdp


FEET_BODY_PATTERN = ".*_ankle_roll"


def apply_stabilized_default(scene_robot_cfg) -> None:
    scene_robot_cfg.init_state.joint_pos.update(
        athletic_stabilized_default_joint_pos_dict(include_toes=True)
    )


@configclass
class LittlegreenHardwareAlignedEnvCfg(LocomotionVelocityEnvCfg):
    """200 Hz physics / 50 Hz policy base with measured ST3215 limits."""

    def __post_init__(self):
        super().__post_init__()
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


@configclass
class StandCommandsCfg:
    """Pure canonical Stand: command observations are always zero."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(8.0, 12.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class TransferCommandsCfg:
    """Transfer starts in place; the curriculum adds gentle directional intent."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(6.0, 9.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class WalkCommandsCfg:
    """Narrow forward commands first; curriculum later enables full direction."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(6.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.28,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 0.22),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class SafetyTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


def _joint_limit_event() -> EventTerm:
    return EventTerm(
        func=mdp.set_joint_position_limits,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIONABLE_JOINTS_V1_2_3,
                preserve_order=True,
            ),
            "lower_limits": HARDWARE_LOWER_LIMIT_RAD,
            "upper_limits": HARDWARE_UPPER_LIMIT_RAD,
        },
        mode="startup",
    )


def _material_event(
    static_range: tuple[float, float],
    dynamic_range: tuple[float, float],
) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": static_range,
            "dynamic_friction_range": dynamic_range,
            "restitution_range": (0.0, 0.02),
            "num_buckets": 32,
            "make_consistent": True,
        },
        mode="startup",
    )


def _mass_event(initial_range: tuple[float, float]) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": initial_range,
            "operation": "scale",
        },
        mode="reset",
    )


def _reset_base_event(
    xy_position: float,
    velocity: float,
) -> EventTerm:
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {
                "x": (-xy_position, xy_position),
                "y": (-xy_position, xy_position),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-velocity, velocity),
                "y": (-velocity, velocity),
                "z": (0.0, 0.0),
                "roll": (-velocity, velocity),
                "pitch": (-velocity, velocity),
                "yaw": (-velocity, velocity),
            },
        },
        mode="reset",
    )


def _reset_joints_event(offset: float) -> EventTerm:
    return EventTerm(
        func=mdp.reset_joints_by_offset,
        params={
            "position_range": (-offset, offset),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )


def _push_event(initial_xy: float) -> EventTerm:
    return EventTerm(
        func=mdp.push_by_setting_velocity,
        params={
            "velocity_range": {
                "x": (-initial_xy, initial_xy),
                "y": (-initial_xy, initial_xy),
            }
        },
        mode="interval",
        interval_range_s=(5.0, 8.0),
    )


@configclass
class StandEventsCfg:
    reset_static_phase = EventTerm(
        func=mdp.reset_random_static_phase_v24,
        mode="reset",
    )
    set_hardware_joint_limits = _joint_limit_event()
    physics_material = _material_event((0.88, 1.06), (0.82, 1.00))
    base_mass = _mass_event((0.995, 1.005))
    actuator_gains = None
    reset_base = _reset_base_event(0.015, 0.015)
    reset_robot_joints = _reset_joints_event(0.008)
    push_robot = _push_event(0.0)


@configclass
class TransferEventsCfg:
    reset_transfer_phase = EventTerm(
        func=mdp.reset_continuous_transfer_phase_v24,
        mode="reset",
    )
    set_hardware_joint_limits = _joint_limit_event()
    physics_material = _material_event((0.88, 1.04), (0.83, 0.99))
    base_mass = _mass_event((0.997, 1.003))
    actuator_gains = None
    reset_base = _reset_base_event(0.012, 0.010)
    reset_robot_joints = _reset_joints_event(0.005)
    push_robot = _push_event(0.0)


@configclass
class WalkEventsCfg:
    reset_walk_phase = EventTerm(
        func=mdp.reset_command_synchronized_phase_v24,
        mode="reset",
    )
    set_hardware_joint_limits = _joint_limit_event()
    physics_material = _material_event((0.88, 1.04), (0.83, 0.99))
    base_mass = _mass_event((0.997, 1.003))
    actuator_gains = None
    reset_base = _reset_base_event(0.010, 0.008)
    reset_robot_joints = _reset_joints_event(0.004)
    push_robot = _push_event(0.0)
