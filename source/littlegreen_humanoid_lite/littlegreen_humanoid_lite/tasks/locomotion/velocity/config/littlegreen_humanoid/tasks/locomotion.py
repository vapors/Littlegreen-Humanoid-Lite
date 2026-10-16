"""LittleGreen v11 canonical gait task; exact v10.3.1 baseline mechanics.

The corrected v2926 USD is retained.  Compared with v10.2 this changes only
implicit actuator PD gains (k150/d0.75) and the sampled command envelope.
Action, observation, foot support events, reward mechanics and weights are
intentionally retained for an interpretable initial retraining baseline.
There is no PPO warm start or mass randomization.
"""
from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
)
from .. import mdp
from ..actions_cfg import LittlegreenActionsCfg
from ..base_cfg import LittlegreenHardwareAlignedEnvCfg, apply_stabilized_default
from ..observations_cfg import LittlegreenVelocityObservationsCfg, set_phase_mode
from ..rewards.locomotion import LocomotionRewardsCfg


@configclass
class LocomotionCommandsCfg:
    """Stage A: forward-only command acquisition, no automatic multi-axis curriculum.

    Near-zero command samples are needed for stable start / hold behavior.
    Commanded movement must exceed the existing gait acquisition 0.08 m/s gate.
    """
    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(6.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.12,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.12, 0.30),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class LocomotionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Restore the successful v6.8 failure boundary rather than the v8/v9 large
    # terminal-cost regime.  Falls remain cheap enough to support exploration.
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.20, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class LocomotionEventsCfg:
    set_hardware_joint_limits = EventTerm(
        func=mdp.set_joint_position_limits,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
            "lower_limits": HARDWARE_LOWER_LIMIT_RAD,
            "upper_limits": HARDWARE_UPPER_LIMIT_RAD,
        },
        mode="startup",
    )
    actuator_gains = None

    # Exact v6.8 reset disturbance envelope.  The larger v8/v9 reset velocities are
    # deliberately not carried over because they generated recovery steps that
    # competed with locomotion acquisition.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (0.0, 0.0),
                "roll": (-0.12, 0.12), "pitch": (-0.12, 0.12), "yaw": (-0.12, 0.12),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.070, 0.070), "velocity_range": (-0.030, 0.030)},
        mode="reset",
    )


@configclass
class LittlegreenLocomotionEnvCfg(LittlegreenHardwareAlignedEnvCfg):
    commands: LocomotionCommandsCfg = LocomotionCommandsCfg()
    observations: LittlegreenVelocityObservationsCfg = LittlegreenVelocityObservationsCfg()
    actions: LittlegreenActionsCfg = LittlegreenActionsCfg()
    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()
    terminations: LocomotionTerminationsCfg = LocomotionTerminationsCfg()
    events: LocomotionEventsCfg = LocomotionEventsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        apply_stabilized_default(self.scene.robot)
        # Keep the existing mass-corrected asset rather than reverting to the
        # obsolete ~10.60 kg USD, and make this an explicitly free-root task.
        if not self.scene.robot.spawn.usd_path.endswith("/lilgreen_mass_v2926.usd"):
            raise ValueError("v11 requires the corrected v2926 USD")
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        # Physical crouch-matched Stage 1 candidate: k150 and damping 0.75x.
        # These are simulation implicit-PD gains, NOT ST3215 firmware PID.
        for group in ("legs", "ankles"):
            self.scene.robot.actuators[group].stiffness = 18.0
            self.scene.robot.actuators[group].damping = 1.5
        set_phase_mode(self.observations, "neutral_static")

        action = self.actions.joint_pos
        if list(action.joint_names) != list(ACTIONABLE_JOINTS_V1_2_3):
            raise ValueError("v11 locomotion joint order mismatch")
        if list(action.lower_limits) != list(HARDWARE_LOWER_LIMIT_RAD):
            raise ValueError("v11 locomotion lower physical limit mismatch")
        if list(action.upper_limits) != list(HARDWARE_UPPER_LIMIT_RAD):
            raise ValueError("v11 locomotion upper physical limit mismatch")
