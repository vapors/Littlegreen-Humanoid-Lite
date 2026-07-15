import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from littlegreen_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from littlegreen_humanoid_lite_assets.robots.lilgreen_humanoid import LILGREEN_CFG, LILGREEN_JOINTS, ACTIONABLE_JOINTS



#=====================================
# === env_cfg weights and scaling  ===
#=====================================

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            #noise=Unoise(n_min=-0.3, n_max=0.3),
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            #noise=Unoise(n_min=-1.5, n_max=2.0),
            noise=Unoise(n_min=-2.0, n_max=2.0),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        #resampling_time_range=(10.0, 10.0),
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            #lin_vel_x=(-0.5, 0.5),lin_vel_y=(-0.5, 0.5),ang_vel_z=(-0.5, 0.5),
            #*lin_vel_x=(-1.0, 1.0),lin_vel_y=(-0.75, 0.75),ang_vel_z=(-1.5, 1.5),
            lin_vel_x=(-1.0, 1.0),lin_vel_y=(-0.5, 0.5),ang_vel_z=(-1.5, 1.5),
            #lin_vel_x=(-1.0, 1.0),lin_vel_y=(-0.5, 0.5),ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS,
        preserve_order=True,
        scale=0.35,
        #scale=0.25,
        use_default_offset=True,
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === Reward for basic survival ===
    # termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    # motion smoothness
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        #weight=-0.1,
        weight=-0.03,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.005,
        #weight=-0.005,
    )
    # ensure the robot is standing upright
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        #weight=-1.0,
        #weight=-1.0,
        weight=-0.05,
    )

    # joint efforts
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        #weight=-6.0e-6,
        
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        #weight=-1.0e-7,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.02,
        #weight=-0.1,
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        #weight = -0.001,  # stronger penalty for stiffness
        weight=-0.0005,

    )

    #=============================================
    # === Rewards for task-space performance ===
    #=============================================

    # command tracking performance
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        #weight=2.5,
        #params={"command_name": "base_velocity", "std": 0.5},
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        #weight=1.0,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )



    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.3,  # Base weight
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,               # Max reward per foot air time (seconds)
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            #"scaling_factor": 2.0,          # Command velocity scaling sensitivity
            #"max_reward_per_step": 1.0,     # Total reward cap per step
        },
    )


    # Reward robot for lifting the swing leg during step 
    swing_leg_lift = RewTerm(
        func=mdp.swing_leg_lift,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Reward robot for planting foot ind cmd vel direction to initiate step 
    swing_foot_plant = RewTerm(
        func=mdp.swing_foot_plant,
        weight=0.6,  # Tune this weight based on testing
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_hold_threshold": 0.1,     # Hold contact for 0.1s to count
            "alignment_tolerance": 0.1,        # Optional: tighter or looser alignment check
            "max_reward": 0.2,                 # Per foot cap
            "cmd_scaling_factor": 2.0          # Scale reward with command velocity
        },
    )

    # Reward robot for for leaning over stance leg 
    lean_over_stance_leg = RewTerm(
        func=mdp.lean_over_stance_leg,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

    # Reward robot for maintaining a base height
    base_height = RewTerm(
        func=mdp.maintain_base_height,
        weight=0.03,  # You can tune this
        params={
            "desired_height": 0.40,  #0.3 Adjust based on your robot's geometry
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.05,  # Optional: narrower = stricter penalty
        },
    )
    #=============================================
    # === Penalties for task-space performance ===
    #=============================================

    # penalize ground contact of key bodies
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_yaw", ".*_knee_pitch"]),"threshold": 1.0,
        },
    )

    # penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    
    # penalize deviation of ankle roll joints
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        #weight=-0.4,
        weight=-0.01,#success
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )

    # penalize moving into a splits position
    joint_deviation_splits= RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    # penalize feet sliding on the ground to exploit physics sim inaccuracies
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        #weight=-1.0,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )

    #=============================================================
    # === Shaped Reward / Penalties for task-space performance ===
    #=============================================================

    # reward/penalize steps not hops unless needed to balance
    step_no_hop = RewTerm(
        func=mdp.encourage_stepping_not_hopping_refined,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
            "lift_threshold": 0.5,              # Base threshold for upward foot velocity (m/s)
            "base_single_leg_reward": 0.05,      # Reward for single-leg stance without lift
            "hop_penalty_scale": -1.0,          # Base penalty for hopping
            "scaling_factor": 2.0,              # Command velocity scaling sensitivity
        },
    )


    # reward/penalize for alternating foot stance and swinng
    foot_alternation = RewTerm(
        func=mdp.encourage_foot_alternation,
        weight=1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
            "history_len": 5,
            "penalty_scale": 0.05,
            "ignore_double_contact": True,
            "scaling_factor": 8.0,
            "max_reward": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.3, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            #"mass_distribution_params": (0.0001, 2.0),
            "operation": "add",
        },
    )
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )

    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-2.0, 2.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                #"x": (-0.3, 0.3),
                #"y": (-0.3, 0.3),
                #"z": (0.0, 0.0),
                #"roll": (-0.3, 0.3),
                #"pitch": (-0.3, 0.3),
                #"yaw": (-0.3, 0.3),
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class LilgreenHumanoidEnvCfg(LocomotionVelocityEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Physics settings
        # 25 Hz override
        self.decimation = 8
        #self.episode_length_s = 25.0
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005

        # Scene
        self.scene.robot = LILGREEN_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None

        self.events.push_robot = None

