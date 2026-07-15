"""Lightweight training diagnostics for Berkeley Humanoid Lite v1.2.2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class PolicyDiagnostics(ManagerTermBase):
    """Track action health, torque use, base-COM height, and standing gate quality."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_standing_age = torch.zeros(self.num_envs, device=self.device)
        self._posture_conforming_age = torch.zeros(self.num_envs, device=self.device)
        self._cached: dict[str, float | torch.Tensor] = {}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stable_standing_age[env_ids] = 0.0
        self._posture_conforming_age[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        action_name: str,
        asset_cfg: SceneEntityCfg,
        foot_asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        joint_velocity_limit_rad_s: float,
        torque_limit_nm: float,
        torque_soft_ratio: float = 0.70,
        desired_base_com_height_m: float = 0.656,
        update_interval_steps: int = 25,
        standing_command_threshold: float = 0.05,
    ) -> dict[str, float | torch.Tensor]:
        del env_ids

        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        action_term = env.action_manager.get_term(action_name)
        command = env.command_manager.get_command(command_name)

        standing = torch.linalg.vector_norm(command[:, :3], dim=1) < standing_command_threshold
        upright = asset.data.projected_gravity_b[:, 2] < -0.97
        quiet_xy = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1) < 0.15
        quiet_yaw = torch.abs(asset.data.root_ang_vel_b[:, 2]) < 0.20
        joint_error = torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        )
        near_default = torch.amax(joint_error, dim=1) < 0.20

        forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        foot_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0
        both_feet = torch.all(foot_contact, dim=1)

        # Stability and posture conformity are intentionally separate. A policy can
        # stand robustly in a non-default crouch; that must not be reported as a
        # standing failure merely because posture shaping is not yet satisfied.
        stable_standing = standing & upright & quiet_xy & quiet_yaw & both_feet
        posture_conforming = stable_standing & near_default

        self._stable_standing_age = torch.where(
            stable_standing,
            self._stable_standing_age + env.step_dt,
            torch.zeros_like(self._stable_standing_age),
        )
        self._posture_conforming_age = torch.where(
            posture_conforming,
            self._posture_conforming_age + env.step_dt,
            torch.zeros_like(self._posture_conforming_age),
        )

        if self._cached and int(env.common_step_counter) % max(1, update_interval_steps) != 0:
            return self._cached.copy()

        raw = action_term.raw_actions
        bounded = action_term.bounded_actions
        targets = action_term.processed_actions
        lower = action_term.lower_limits
        upper = action_term.upper_limits

        raw_abs = torch.abs(raw)
        target_at_limit = (targets <= lower + 1.0e-3) | (targets >= upper - 1.0e-3)
        vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        vel_fraction = torch.mean((vel >= 0.95 * joint_velocity_limit_rad_s).float())

        torque_abs = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
        torque_utilization = torque_abs / torque_limit_nm
        torque_soft_fraction = torch.mean((torque_utilization > torque_soft_ratio).float())
        torque_limit_fraction = torch.mean((torque_utilization >= 0.95).float())

        standing_count = torch.clamp(torch.sum(standing), min=1)
        stable_success_10 = torch.sum((self._stable_standing_age >= 10.0) & standing) / standing_count
        stable_success_20 = torch.sum((self._stable_standing_age >= 20.0) & standing) / standing_count
        posture_success_10 = torch.sum((self._posture_conforming_age >= 10.0) & standing) / standing_count
        posture_success_20 = torch.sum((self._posture_conforming_age >= 20.0) & standing) / standing_count
        zero_cmd_drift = torch.sum(
            torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1) * standing
        ) / standing_count

        body_vel = asset.data.body_lin_vel_w[:, foot_asset_cfg.body_ids, :2]
        slip = torch.sum(torch.linalg.vector_norm(body_vel, dim=-1) * foot_contact, dim=1)
        standing_slip = torch.sum(slip * standing) / standing_count

        # Root frame height and root-body COM height are intentionally both reported.
        root_frame_height = asset.data.root_pos_w[:, 2]
        base_com_height = asset.data.root_com_pos_w[:, 2]
        base_com_height_error = torch.abs(base_com_height - desired_base_com_height_m)
        posture_rms = torch.sqrt(torch.mean(torch.square(joint_error), dim=1))
        posture_max = torch.amax(joint_error, dim=1)

        vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
        lin_tracking_error = command[:, :2] - vel_yaw[:, :2]
        yaw_tracking_error = command[:, 2] - asset.data.root_ang_vel_w[:, 2]

        # v1.4.3 gait/leg-lift diagnostics. These are numeric episode extras, so
        # they are safe for RSL-RL logging and useful even before offline analysis.
        moving = torch.linalg.vector_norm(command[:, :2], dim=1) > 0.12
        moving_count = torch.clamp(torch.sum(moving), min=1)
        foot_contact_count = torch.sum(foot_contact.int(), dim=1)
        single_support = foot_contact_count == 1
        double_support = foot_contact_count == 2
        no_support = foot_contact_count == 0
        current_air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
        air_fraction_per_foot = torch.mean((current_air_time > 0.0).float(), dim=0)
        max_air_time = torch.max(current_air_time, dim=1)[0]
        if current_air_time.shape[1] > 1:
            air_time_imbalance = torch.abs(current_air_time[:, 0] - current_air_time[:, 1])
        else:
            air_time_imbalance = torch.zeros_like(max_air_time)
        long_air_time = max_air_time > 0.28
        foot_z = asset.data.body_pos_w[:, foot_asset_cfg.body_ids, 2]
        stance_z = torch.sum(foot_z * foot_contact.float(), dim=1) / torch.clamp(
            torch.sum(foot_contact.float(), dim=1), min=1.0
        )
        swing_clearance = torch.where(
            ~foot_contact, torch.relu(foot_z - stance_z.unsqueeze(-1)), torch.zeros_like(foot_z)
        )
        max_swing_clearance = torch.max(swing_clearance, dim=1)[0]

        command_xy = command[:, :2]
        command_norm = torch.linalg.vector_norm(command_xy, dim=1)
        command_dir = command_xy / torch.clamp(command_norm.unsqueeze(-1), min=1.0e-6)
        body_vel_yaw = quat_apply_inverse(
            yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(foot_asset_cfg.body_ids), -1),
            asset.data.body_lin_vel_w[:, foot_asset_cfg.body_ids, :3],
        )[:, :, :2]
        swing_foot_velocity_projection = torch.sum(
            body_vel_yaw * command_dir.unsqueeze(1), dim=-1
        )
        swing_forward_velocity = torch.sum(
            torch.clamp(swing_foot_velocity_projection, min=0.0) * (~foot_contact).float(), dim=1
        )

        def standing_fraction(condition: torch.Tensor) -> torch.Tensor:
            return torch.sum((condition & standing).float()) / standing_count

        def moving_fraction(condition: torch.Tensor) -> torch.Tensor:
            return torch.sum((condition & moving).float()) / moving_count

        cached: dict[str, float | torch.Tensor] = {
            "raw_action_mean_abs": torch.mean(raw_abs),
            "raw_action_std": torch.std(raw),
            "raw_action_min": torch.min(raw),
            "raw_action_max": torch.max(raw),
            "raw_action_excess_fraction": torch.mean((raw_abs > 1.0).float()),
            "bounded_saturation_fraction": torch.mean((torch.abs(bounded) >= 0.999).float()),
            "target_limit_fraction": torch.mean(target_at_limit.float()),
            "joint_velocity_limit_fraction": vel_fraction,
            "torque_over_soft_fraction": torque_soft_fraction,
            "torque_limit_fraction": torque_limit_fraction,
            "torque_abs_mean_nm": torch.mean(torque_abs),
            "standing_env_fraction": torch.mean(standing.float()),
            # Backward-compatible keys retain the old posture-conforming meaning.
            "standing_success_10s_fraction": posture_success_10,
            "standing_success_20s_fraction": posture_success_20,
            "standing_current_max_continuous_success_s": torch.max(self._posture_conforming_age),
            # Explicit v1.2.3 split metrics.
            "standing_stable_success_10s_fraction": stable_success_10,
            "standing_stable_success_20s_fraction": stable_success_20,
            "standing_current_max_continuous_stable_s": torch.max(self._stable_standing_age),
            "standing_posture_success_10s_fraction": posture_success_10,
            "standing_posture_success_20s_fraction": posture_success_20,
            "standing_current_max_continuous_posture_s": torch.max(self._posture_conforming_age),
            "standing_upright_fraction": standing_fraction(upright),
            "standing_quiet_xy_fraction": standing_fraction(quiet_xy),
            "standing_quiet_yaw_fraction": standing_fraction(quiet_yaw),
            "standing_near_default_fraction": standing_fraction(near_default),
            "standing_both_feet_fraction": standing_fraction(both_feet),
            "standing_stable_all_conditions_fraction": standing_fraction(stable_standing),
            "standing_posture_all_conditions_fraction": standing_fraction(posture_conforming),
            # Backward-compatible alias for v1.2.2 dashboards.
            "standing_all_conditions_fraction": standing_fraction(posture_conforming),
            "standing_joint_posture_rms_mean_rad": torch.sum(posture_rms * standing) / standing_count,
            "standing_joint_posture_max_mean_rad": torch.sum(posture_max * standing) / standing_count,
            "zero_command_xy_drift_mean_mps": zero_cmd_drift,
            "standing_foot_slip_mean_mps": standing_slip,
            "root_frame_height_mean_m": torch.mean(root_frame_height),
            "base_com_height_mean_m": torch.mean(base_com_height),
            "standing_base_com_height_mean_m": torch.sum(base_com_height * standing) / standing_count,
            "standing_base_com_height_error_mean_m": torch.sum(base_com_height_error * standing) / standing_count,
            "linear_velocity_tracking_rmse_mps": torch.sqrt(torch.mean(torch.square(lin_tracking_error))),
            "yaw_rate_tracking_rmse_rad_s": torch.sqrt(torch.mean(torch.square(yaw_tracking_error))),
            "one_step_delay_env_fraction": torch.mean(action_term.delay_one_step.float()),
            "moving_command_fraction": torch.mean(moving.float()),
            "single_support_fraction": torch.mean(single_support.float()),
            "double_support_fraction": torch.mean(double_support.float()),
            "no_support_fraction": torch.mean(no_support.float()),
            "moving_single_support_fraction": moving_fraction(single_support),
            "moving_double_support_fraction": moving_fraction(double_support),
            "moving_no_support_fraction": moving_fraction(no_support),
            "foot_air_fraction_left": air_fraction_per_foot[0],
            "foot_air_fraction_right": air_fraction_per_foot[1] if air_fraction_per_foot.numel() > 1 else air_fraction_per_foot[0],
            "moving_max_air_time_mean_s": torch.sum(max_air_time * moving.float()) / moving_count,
            "moving_air_time_imbalance_mean_s": torch.sum(air_time_imbalance * moving.float()) / moving_count,
            "moving_long_air_time_fraction": torch.sum((long_air_time & moving).float()) / moving_count,
            "moving_swing_clearance_mean_m": torch.sum(max_swing_clearance * moving.float()) / moving_count,
            "moving_swing_clearance_pseudo_max_m": torch.max(max_swing_clearance * moving.float()),
            "moving_swing_foot_forward_velocity_mean_mps": torch.sum(swing_forward_velocity * moving.float()) / moving_count,
        }

        if hasattr(action_term, "drive_targets"):
            drive_targets = action_term.drive_targets
            delayed_targets = action_term.delayed_policy_targets
            cached.update({
                "actuator_policy_to_drive_lag_abs_mean_rad": torch.mean(torch.abs(targets - drive_targets)),
                "actuator_policy_to_delayed_abs_mean_rad": torch.mean(torch.abs(targets - delayed_targets)),
                "actuator_delay_mean_ms": 1000.0 * torch.mean(action_term.sampled_total_delay_s),
                "actuator_tau_mean_ms": 1000.0 * torch.mean(action_term.sampled_tau_s),
                "actuator_velocity_scale_mean": torch.mean(action_term.sampled_velocity_scale),
            })

            if (
                hasattr(action_term, "loaded_envelope_enabled")
                and action_term.loaded_envelope_enabled
            ):
                cached.update({
                    "actuator_loaded_velocity_cap_mean_rad_s": torch.mean(
                        action_term.loaded_velocity_cap_rad_s
                    ),
                    "actuator_final_velocity_cap_mean_rad_s": torch.mean(
                        action_term.velocity_cap_rad_s
                    ),
                    "actuator_loaded_envelope_active_fraction": torch.mean(
                        action_term.loaded_envelope_active.float()
                    ),
                    "actuator_reference_speed_mean_rad_s": torch.mean(
                        action_term.reference_speed_rad_s
                    ),
                    "actuator_loaded_velocity_scale_mean": torch.mean(
                        action_term.sampled_loaded_velocity_scale
                    ),
                })

        # Per-joint action and torque health metrics. Global metrics are preserved
        # for continuity; standing-only metrics isolate the behavior that matters
        # for the residual Stand-v0 experiment.
        standing_weight = standing.float()
        default = action_term.default_positions
        target_residual_abs = torch.abs(targets - default)
        bounded_abs = torch.abs(bounded)
        joint_pos_error_abs = joint_error

        for index, name in enumerate(action_term.joint_names):
            short_name = name.replace("leg_left_", "L_").replace("leg_right_", "R_").replace("_joint", "")
            cached[f"raw_abs_mean/{short_name}"] = torch.mean(raw_abs[:, index])
            cached[f"raw_excess_fraction/{short_name}"] = torch.mean((raw_abs[:, index] > 1.0).float())
            cached[f"torque_over_soft_fraction/{short_name}"] = torch.mean(
                (torque_utilization[:, index] > torque_soft_ratio).float()
            )
            cached[f"torque_limit_fraction/{short_name}"] = torch.mean(
                (torque_utilization[:, index] >= 0.95).float()
            )

            cached[f"standing/raw_abs_mean/{short_name}"] = torch.sum(
                raw_abs[:, index] * standing_weight
            ) / standing_count
            cached[f"standing/bounded_abs_mean/{short_name}"] = torch.sum(
                bounded_abs[:, index] * standing_weight
            ) / standing_count
            cached[f"standing/raw_excess_fraction/{short_name}"] = torch.sum(
                (raw_abs[:, index] > 1.0).float() * standing_weight
            ) / standing_count
            cached[f"standing/target_residual_abs_mean_rad/{short_name}"] = torch.sum(
                target_residual_abs[:, index] * standing_weight
            ) / standing_count
            cached[f"standing/target_limit_fraction/{short_name}"] = torch.sum(
                target_at_limit[:, index].float() * standing_weight
            ) / standing_count
            cached[f"standing/joint_pos_error_abs_mean_rad/{short_name}"] = torch.sum(
                joint_pos_error_abs[:, index] * standing_weight
            ) / standing_count
            cached[f"standing/torque_abs_mean_nm/{short_name}"] = torch.sum(
                torque_abs[:, index] * standing_weight
            ) / standing_count
            cached[f"standing/torque_over_soft_fraction/{short_name}"] = torch.sum(
                (torque_utilization[:, index] > torque_soft_ratio).float() * standing_weight
            ) / standing_count
            cached[f"standing/torque_limit_fraction/{short_name}"] = torch.sum(
                (torque_utilization[:, index] >= 0.95).float() * standing_weight
            ) / standing_count

            if hasattr(action_term, "drive_targets"):
                drive_lag = torch.abs(targets[:, index] - action_term.drive_targets[:, index])
                cached[f"standing/actuator_drive_lag_abs_mean_rad/{short_name}"] = torch.sum(
                    drive_lag * standing_weight
                ) / standing_count

                if (
                    hasattr(action_term, "loaded_envelope_enabled")
                    and action_term.loaded_envelope_enabled
                ):
                    cached[f"standing/actuator_loaded_active_fraction/{short_name}"] = torch.sum(
                        action_term.loaded_envelope_active[:, index].float() * standing_weight
                    ) / standing_count
                    cached[f"standing/actuator_final_velocity_cap_mean_rad_s/{short_name}"] = torch.sum(
                        action_term.velocity_cap_rad_s[:, index] * standing_weight
                    ) / standing_count
                    cached[f"standing/actuator_loaded_velocity_cap_mean_rad_s/{short_name}"] = torch.sum(
                        action_term.loaded_velocity_cap_rad_s[:, index] * standing_weight
                    ) / standing_count
                    cached[f"standing/actuator_reference_speed_mean_rad_s/{short_name}"] = torch.sum(
                        action_term.reference_speed_rad_s[:, index] * standing_weight
                    ) / standing_count

        self._cached = cached
        return cached.copy()
