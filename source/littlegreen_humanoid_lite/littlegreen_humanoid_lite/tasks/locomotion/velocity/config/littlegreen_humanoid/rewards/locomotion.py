"""v11.1.0 bilateral acquisition experiment; v11.0 HF1 preserved as archive baseline."""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    ST3215_PEAK_TORQUE_NM,
)
from .. import mdp

WHOLE_FOOT_BODIES = [
    "leg_left_ankle_roll", "leg_left_toe_pitch",
    "leg_right_ankle_roll", "leg_right_toe_pitch",
]


def _whole(name: str) -> SceneEntityCfg:
    return SceneEntityCfg(name, body_names=WHOLE_FOOT_BODIES, preserve_order=True)


_VIABILITY = {
    "full_tilt_deg": 25.0,
    "zero_tilt_deg": 55.0,
    "full_height_m": 0.400,
    "zero_height_m": 0.340,
}

# Exact v6.8 support-baton mechanics.
_GAIT_STATE = {
    "command_threshold": 0.08,
    "start_unload_share": 0.45,
    "start_support_share": 0.55,
    "start_rise_speed_mps": 0.030,
    "readiness_load_full": 0.68,
    "readiness_com_full": 0.68,
    "clearance_milestones": (0.003, 0.010, 0.022, 0.040),
    "min_meaningful_clearance_m": 0.018,
    "min_meaningful_air_time_s": 0.08,
    "trajectory_start_clearance_m": 0.008,
    "full_trajectory_progress_m": 0.040,
    "completion_load_share": 0.25,
    "completion_replant_s": 0.06,
    "takeover_load_full": 0.55,
    "takeover_com_full": 0.68,
    "handoff_load_threshold": 0.50,
    "handoff_com_threshold": 0.56,
    "handoff_hold_s": 0.08,
    "takeover_timeout_s": 0.80,
    "release_air_time_s": 0.04,
    "release_clearance_m": 0.010,
    "continuation_multiplier": 1.50,
    "first_multiplier": 1.00,
    "event_hold_s": 0.06,
    "min_viability": 0.45,
    "handoff_min_viability": 0.60,
    "bodies_per_foot": 2,
    **_VIABILITY,
}

# Exact v6.8 soft command-conditioned stride / swing-speed saturation scales.
_V68_REFINEMENT = {
    "stride_base_m": 0.025,
    "stride_gain_s": 0.10,
    "min_stride_horizon_m": 0.030,
    "max_stride_horizon_m": 0.070,
    "swing_speed_base_mps": 0.12,
    "swing_speed_gain": 0.45,
    "min_swing_speed_mps": 0.15,
    "max_swing_speed_mps": 0.34,
    "frontier_deadband_m": 0.006,
    "frontier_zero_credit_m": 0.014,
    "retention_min_progress_m": 0.010,
}

_V68_STATE = {**_GAIT_STATE, **_V68_REFINEMENT}

_GAIT_ROUTING = {
    "command_name": "base_velocity",
    "sensor_cfg": _whole("contact_forces"),
    "asset_cfg": _whole("robot"),
}

# Proven v6.9 event-only translation horizons.  These activate only around
# successful replants/handoffs, so forward body fall cannot farm them.
_V69_TRANSLATION = {
    "command_name": "base_velocity",
    "sensor_cfg": _whole("contact_forces"),
    "asset_cfg": _whole("robot"),
    "command_threshold": 0.08,
    "support_progress_base_m": 0.010,
    "support_progress_gain_s": 0.067,
    "min_support_progress_horizon_m": 0.015,
    "max_support_progress_horizon_m": 0.040,
    "com_progress_base_m": 0.010,
    "com_progress_gain_s": 0.067,
    "min_com_progress_horizon_m": 0.015,
    "max_com_progress_horizon_m": 0.040,
    "post_touchdown_deadband_m": 0.005,
    "post_touchdown_full_retreat_m": 0.015,
    "handoff_retention_min_progress_m": 0.010,
    "direction_continuity_cos_min": 0.50,
    "translation_event_hold_s": 0.06,
}

# Shared routing for historical v6.6/v6.7 helpers that still use flat params.
_GAIT_FLAT = {**_GAIT_ROUTING, **_GAIT_STATE}


@configclass
class LocomotionRewardsCfg:
    # ------------------------------------------------------------------
    # Proven v6.8 gait acquisition, intentionally restored at full strength.
    # ------------------------------------------------------------------
    track_lin_vel_xy_acquisition = RewTerm(
            func=mdp.gait_conditioned_track_lin_vel_xy_v67,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": _whole("contact_forces"),
                "asset_cfg": _whole("robot"),
                "std": 0.50, "command_threshold": 0.08,
                "minimum_credit": 0.22, "full_clearance_m": 0.040,
                "clearance_credit": 0.48, "single_support_credit": 0.20,
                "load_asymmetry_credit": 0.10, **_VIABILITY,
            },
            weight=0.85,  # R3 slight tracking increase, paired with transfer bridge
        )
    track_ang_vel_z_acquisition = RewTerm(
        func=mdp.track_ang_vel_z_acquisition_v63,
        params={"command_name": "base_velocity", "std": 0.50, "command_threshold": 0.08, **_VIABILITY},
        weight=0.55,
    )

    base_com_height_deficit = RewTerm(
        func=mdp.base_com_height_deficit_v63,
        params={"full_penalty_height_m": 0.405, "zero_penalty_height_m": 0.445},
        weight=-0.40,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.18)

    support_readiness_progress = RewTerm(
            func=mdp.support_readiness_progress_v66,
            params=dict(_GAIT_FLAT),
            weight=0.35,  # was 0.50; ready alone is not a step
        )
    stance_relative_com_drive = RewTerm(
            func=mdp.body_yaw_lateral_support_transfer_v99,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": _whole("contact_forces"),
                "asset_cfg": _whole("robot"), "command_threshold": 0.08,
                "min_stance_load_share": 0.54, "target_com_closeness": 0.68,
                "full_relative_speed_mps": 0.15,
                "max_stance_speed_mps": 0.08, **_VIABILITY,
            },
            weight=0.18,  # was 0.25; lateral bridge is scaffolding, not goal
        )
    ready_no_liftoff = RewTerm(
        func=mdp.ready_no_liftoff_penalty_v67,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "asset_cfg": _whole("robot"), "command_threshold": 0.08,
            "ready_stance_load_share": 0.62, "ready_com_closeness": 0.58,
            "ready_swing_max_load_share": 0.38, "lift_clearance_m": 0.003,
            "lift_rise_speed_mps": 0.030, "grace_s": 0.18, "full_penalty_s": 0.45,
        },
        weight=-0.45,
    )

    finite_swing_initiation = RewTerm(
            func=mdp.finite_whole_foot_swing_initiation_v66,
            params=dict(_GAIT_FLAT),
            weight=1.90,  # was 1.35
        )
    coherent_swing_rise_velocity = RewTerm(
        func=mdp.coherent_swing_rise_velocity_v67,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "asset_cfg": _whole("robot"), "command_threshold": 0.08,
            "max_swing_load_share": 0.45, "min_stance_load_share": 0.55,
            "onset_speed_mps": 0.015, "full_speed_mps": 0.15,
            "full_clearance_m": 0.040, **_VIABILITY,
        },
        weight=0.90,
    )

    # v6.8 directional on-ramp, now frame-corrected and explicitly whole-foot
    # airborne.  Weight is raised only ~10% from 1.45 -> 1.60.
    swing_foot_command_velocity = RewTerm(
            func=mdp.swing_foot_command_velocity_v95,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": _whole("contact_forces"),
                "asset_cfg": _whole("robot"),
                "v68_params": dict(_V68_STATE),
                "min_clearance_m": 0.003,
                "full_clearance_factor_m": 0.018,
                "max_stance_speed_mps": 0.12,
                "wrong_way_deadband_mps": 0.020,
                "wrong_way_scale": 0.18,
            },
            weight=1.85,  # was 1.60
        )
    finite_directional_swing_progress = RewTerm(
        func=mdp.command_conditioned_directional_swing_progress_v95,
        params={**_GAIT_ROUTING, "v68_params": dict(_V68_STATE)},
        weight=1.20,
    )
    swing_backtrack = RewTerm(
        func=mdp.swing_backtrack_penalty_v95,
        params={
            **_GAIT_ROUTING, "v68_params": dict(_V68_STATE),
            "min_frontier_m": 0.020, "full_backtrack_m": 0.025,
        },
        weight=-0.30,
    )
    swing_behind_liftoff = RewTerm(
        func=mdp.swing_behind_liftoff_penalty_v95,
        params={
            **_GAIT_ROUTING, "v68_params": dict(_V68_STATE),
            "free_deadband_m": 0.002, "full_penalty_behind_m": 0.020,
        },
        weight=-0.18,
    )
    loaded_replant = RewTerm(
        func=mdp.loaded_replant_retention_v95,
        params={**_GAIT_ROUTING, "v68_params": dict(_V68_STATE), "minimum_multiplier": 0.25},
        weight=0.45,
    )

    support_takeover_progress = RewTerm(
            func=mdp.support_takeover_progress_v66,
            params=dict(_GAIT_FLAT),
            weight=1.85,  # was 1.55
        )
    support_handoff = RewTerm(
        func=mdp.support_handoff_event_v66,
        params=dict(_GAIT_FLAT),
        weight=1.85,
    )

    # Keep the v6.9 anti-retreat guard, but remove the two positive fixed-distance
    # translation endpoints.  v10.1 continues to ask both legs to pass the stance frontier
    # and asks confirmed support exchanges to transport through WORLD space at
    # a meaningful fraction of commanded planar speed.
    post_touchdown_support_retreat = RewTerm(
        func=mdp.post_touchdown_support_retreat_penalty_v69,
        params=dict(_V69_TRANSLATION),
        weight=-0.30,
    )
    # v10.2 preserves v10.1 locomotion mechanics through the validated v6.3 confirmed-step ->
    # mechanical support-takeover handoff stream.  This stream is fall-neutral and
    # occurs at actual gait frequency; passage arithmetic is otherwise unchanged.
    # with a 50% coupling to the previous opposite-leg quality so a strong lead leg
    # cannot hide a short-stepping/catch-up leg.
    bilateral_passage_dense_step_support = RewTerm(
        func=mdp.dense_step_support_bilateral_passage_v98,
        params={
            **_GAIT_ROUTING,
            "command_threshold": 0.08,
            "quality_softness_m": 0.025,
            "bilateral_mix": 0.50,
            "event_hold_s": 0.06,
            "command_change_linear_mps": 0.015,
            "command_change_yaw_rps": 0.020,
            **_VIABILITY,
        },
        weight=1.10,
    )
    # Every consecutive v6.3 mechanical support-takeover exchange immediately answers
    # to WORLD support-frontier rate / commanded planar speed.  The signed curve,
    # neutral point, softness, and event hold are unchanged; v10.0 raises only the weight.
    commanded_world_rate_dense_step_support = RewTerm(
        func=mdp.dense_step_support_bilateral_commanded_world_rate_v101,
        params={
            **_GAIT_ROUTING,
            "command_threshold": 0.08,
            "neutral_rate_ratio": 0.20,
            "rate_softness_ratio": 0.12,
            "bilateral_mix": 0.50,
            "event_hold_s": 0.06,
            "command_change_linear_mps": 0.015,
            "command_change_yaw_rps": 0.020,
            **_VIABILITY,
        },
        weight=2.00,
    )

    opposite_support_release = RewTerm(
        func=mdp.opposite_support_release_event_v66,
        params=dict(_GAIT_FLAT),
        # Keep release strongly rewarded so the proven whole-foot alternating gait
        # does not collapse, but reduce its terminal income modestly.
        weight=1.80,
    )
    completed_gait_step = RewTerm(
        func=mdp.completed_gait_step_retention_v95,
        params={**_GAIT_ROUTING, "v68_params": dict(_V68_STATE), "minimum_multiplier": 0.55},
        # Completion remains positive, but it is no longer the dominant terminal
        # reward when the new support fails to advance the world frontier.
        weight=1.85,
    )
    takeover_failure = RewTerm(
        func=mdp.takeover_failure_event_v66,
        params=dict(_GAIT_FLAT),
        weight=-0.35,
    )

    whole_foot_air_time_exploration = RewTerm(
        func=mdp.whole_foot_air_time_viability_gated_v63,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "asset_cfg": _whole("robot"), "threshold": 0.18, "full_clearance_m": 0.040,
            "command_threshold": 0.08, "full_credit_until_s": 0.25, "fade_out_end_s": 0.42,
            "bodies_per_foot": 2, **_VIABILITY,
        },
        weight=0.45,
    )

    prolonged_single_support = RewTerm(
        func=mdp.moving_prolonged_whole_foot_single_support_v30,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "grace_s": 0.70, "ramp_s": 0.35, "command_threshold": 0.08,
            "bodies_per_foot": 2,
        },
        weight=-0.45,
    )
    prolonged_same_stance = RewTerm(
        func=mdp.moving_prolonged_whole_foot_stance_v30,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "grace_s": 1.00, "ramp_s": 0.40, "command_threshold": 0.08,
            "bodies_per_foot": 2,
        },
        weight=-0.25,
    )
    moving_long_double_support = RewTerm(
        func=mdp.moving_long_double_support_penalty,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "command_threshold": 0.08, "max_double_support_s": 0.45,
            "force_threshold": 1.0,
        },
        weight=-0.20,
    )

    # v6.5 zero-command stability cleanup remains unchanged.
    standing_both_whole_feet = RewTerm(
        func=mdp.standing_both_whole_feet_contact_v250,
        params={"command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"), "force_threshold": 1.0, "command_threshold": 0.05},
        weight=0.30,
    )
    standing_bilateral_full_plant = RewTerm(
        func=mdp.standing_bilateral_full_plant_v65,
        params={"command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"), "command_threshold": 0.05, "force_threshold": 1.0, "min_load_share": 0.20},
        weight=0.45,
    )
    standing_whole_foot_load_imbalance = RewTerm(
        func=mdp.standing_whole_foot_load_imbalance_v65,
        params={"command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"), "command_threshold": 0.05, "deadband": 0.12, "scale": 0.35},
        weight=-0.35,
    )
    standing_actionable_joint_velocity = RewTerm(
        func=mdp.standing_actionable_joint_velocity_l2_v65,
        params={"command_name": "base_velocity", "command_threshold": 0.05, "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
        weight=-0.012,
    )
    standing_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.20,
    )
    standing_yaw_rate = RewTerm(
        func=mdp.standing_yaw_rate_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.10,
    )
    standing_lateral_com_balance = RewTerm(
        func=mdp.standing_com_lateral_over_feet_l2,
        params={"command_name": "base_velocity", "asset_cfg": _whole("robot"), "command_threshold": 0.05, "deadband_m": 0.006, "scale_m": 0.045, "max_base_xy_speed_mps": 10.0},
        weight=-0.10,
    )

    feet_slide = RewTerm(
        func=mdp.whole_foot_slide_v39,
        params={"sensor_cfg": _whole("contact_forces"), "asset_cfg": _whole("robot"), "force_threshold": 1.0, "bodies_per_foot": 2},
        weight=-0.30,
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]), "threshold": 1.0},
        weight=-1.0,
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2.50)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
        weight=-1.0e-7,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True)},
        weight=-0.05,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.015)
    sustained_max_joint_torque_barrier = RewTerm(
        func=mdp.sustained_max_torque_barrier_l2_v60,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM, "soft_ratio": 0.92,
            "free_duration_s": 0.25, "full_penalty_duration_s": 0.75,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True),
        },
        weight=-0.08,
    )

    # v11.1 R3: only main+toe fully airborne quality-triggered swing bouts; repeated same-side bouts have diminishing
    # value except for recovery and lateral/strafe commands. No prescribed gait phase.
    balanced_swing_discovery = RewTerm(
        func=mdp.bilateral_swing_discovery_v111,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "asset_cfg": _whole("robot"),
            "command_threshold": 0.08, "clearance_full_m": 0.022,
            "rise_full_mps": 0.100, "forward_full_mps": 0.22,
            "max_stance_speed_mps": 0.12,
            "attempt_threshold": 0.065, "rearm_threshold": 0.025,
            "rearm_steps": 4, "attempt_cooldown_s": 0.25,
            "individual_bootstrap": 0.55, "opposite_attempt_bonus": 0.45,
            "repeat_decay": 0.68, "repeat_floor": 0.20,
            "recency_decay": 0.985,
            "recovery_window_s": 0.30, "prep_window_s": 0.75,
            **_VIABILITY,
        },
        weight=1.20,
    )

    # Only pays for increasing load/COM acceptance after qualified airborne
    # directional swing and full-foot replant; cannot farm static leaning.
    receiving_foot_load_acceptance = RewTerm(
        func=mdp.receiving_foot_load_acceptance_v111,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "asset_cfg": _whole("robot"),
            "command_threshold": 0.08, "clearance_full_m": 0.022,
            "rise_full_mps": 0.100, "forward_full_mps": 0.22,
            "max_stance_speed_mps": 0.12,
            "attempt_threshold": 0.065, "rearm_threshold": 0.025,
            "rearm_steps": 4, "attempt_cooldown_s": 0.25,
            "individual_bootstrap": 0.55, "opposite_attempt_bonus": 0.45,
            "repeat_decay": 0.68, "repeat_floor": 0.20,
            "recency_decay": 0.985,
            "recovery_window_s": 0.30, "prep_window_s": 0.75,
            **_VIABILITY,
        },
        weight=0.85,
    )

    # R3: bounded load + COM preparation, from full-foot airborne to replant;
    # can reward early supported transition without requiring full takeover.
    # Monotone per-command-epoch credit prevents repetitive rocking income.
    receiving_foot_preparation = RewTerm(
        func=mdp.receiving_foot_preparation_v111,
        params={
            "command_name": "base_velocity", "sensor_cfg": _whole("contact_forces"),
            "asset_cfg": _whole("robot"),
            "command_threshold": 0.08, "clearance_full_m": 0.022,
            "rise_full_mps": 0.100, "forward_full_mps": 0.22,
            "max_stance_speed_mps": 0.12,
            "attempt_threshold": 0.065, "rearm_threshold": 0.025,
            "rearm_steps": 4, "attempt_cooldown_s": 0.25,
            "individual_bootstrap": 0.55, "opposite_attempt_bonus": 0.45,
            "repeat_decay": 0.68, "repeat_floor": 0.20,
            "recency_decay": 0.985,
            "recovery_window_s": 0.30, "prep_window_s": 0.75,
            **_VIABILITY,
        },
        weight=0.75,
    )
