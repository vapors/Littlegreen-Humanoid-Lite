"""Measured-response ST3215 action term for Berkeley Humanoid Lite v1.3/v1.4.

Policy action contract v3 is unchanged. This term inserts an empirical
command-to-drive-target response model after residual target mapping and before the
PhysX implicit joint drive.

Stage-A pipeline (v1.3.x):

    raw policy action -> clamp -> q_default +/- 0.20 rad residual -> physical clip
      -> bus-phase delay -> measured response-delay proxy
      -> per-joint static gain -> small-signal hysteresis/error floor
      -> first-order-equivalent lag -> amplitude-conditioned unloaded velocity envelope
      -> PhysX joint position target

Stage-B loaded extension (v1.4.0) preserves every Stage-A block and additionally
applies a conservative whole-body loaded trajectory envelope:

    v_loaded = min(gain * abs(v_ref), vmax)
    v_cap = min(v_stage_a, v_loaded)

Only knee pitch uses full crouch/stand-return fit conditioning by default. A modest
knee stand-return tau multiplier is supported as a response proxy. Loaded envelope
fits and tau multipliers are not physical torque-speed or damping constants.

Track 2 ``tau`` values are response proxies, not physical damping coefficients.
Special overshoot/tail behavior remains deferred so the monotonic model can be
validated independently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .hardware_actions import BoundedDefaultResidualJointPositionAction


class ST3215MeasuredResidualJointPositionAction(BoundedDefaultResidualJointPositionAction):
    """Action contract v3 followed by an empirical ST3215 response model."""

    cfg: "ST3215MeasuredResidualJointPositionActionCfg"

    def __init__(self, cfg: "ST3215MeasuredResidualJointPositionActionCfg", env):
        super().__init__(cfg, env)

        n = self._num_joints
        self._validate_len("tau_median_s", cfg.tau_median_s, n)
        self._validate_len("tau_p10_s", cfg.tau_p10_s, n)
        self._validate_len("tau_p90_s", cfg.tau_p90_s, n)
        self._validate_len("static_gain", cfg.static_gain, n)
        self._validate_len("small_signal_error_floor_rad", cfg.small_signal_error_floor_rad, n)
        self._validate_len("center_hysteresis_span_rad", cfg.center_hysteresis_span_rad, n)
        self._validate_len("velocity_curves_rad_s", cfg.velocity_curves_rad_s, n)

        knot_count = len(cfg.velocity_amplitude_knots_rad)
        if knot_count < 2:
            raise ValueError("velocity_amplitude_knots_rad requires at least two points")
        for index, curve in enumerate(cfg.velocity_curves_rad_s):
            if len(curve) != knot_count:
                raise ValueError(
                    f"velocity curve {index} has {len(curve)} points; expected {knot_count}"
                )

        self._physics_dt = float(env.physics_dt)
        if self._physics_dt <= 0.0:
            raise ValueError("Environment physics_dt must be positive")
        self._policy_dt = float(getattr(env, "step_dt", cfg.policy_dt_s_nominal))
        if self._policy_dt <= 0.0:
            raise ValueError("Environment step_dt/policy_dt must be positive")

        device = self.device
        dtype = torch.float32
        self._tau_median = torch.tensor(cfg.tau_median_s, device=device, dtype=dtype).unsqueeze(0)
        self._tau_p10 = torch.tensor(cfg.tau_p10_s, device=device, dtype=dtype).unsqueeze(0)
        self._tau_p90 = torch.tensor(cfg.tau_p90_s, device=device, dtype=dtype).unsqueeze(0)
        self._static_gain = torch.tensor(cfg.static_gain, device=device, dtype=dtype).unsqueeze(0)
        self._error_floor = torch.tensor(
            cfg.small_signal_error_floor_rad, device=device, dtype=dtype
        ).unsqueeze(0)
        self._hysteresis_halfspan = 0.5 * torch.tensor(
            cfg.center_hysteresis_span_rad, device=device, dtype=dtype
        ).unsqueeze(0)
        self._deadzone = torch.maximum(self._error_floor, self._hysteresis_halfspan)
        self._velocity_knots = torch.tensor(
            cfg.velocity_amplitude_knots_rad, device=device, dtype=dtype
        )
        self._velocity_curves = torch.tensor(
            cfg.velocity_curves_rad_s, device=device, dtype=dtype
        ).unsqueeze(0)

        self._loaded_envelope_enabled = bool(cfg.loaded_envelope_enabled)
        if self._loaded_envelope_enabled:
            self._validate_loaded_cfg(cfg, n)
            self._loaded_crouch_sign = torch.tensor(
                cfg.loaded_crouch_direction_sign, device=device, dtype=dtype
            ).unsqueeze(0)
            self._loaded_crouch_gain = torch.tensor(
                cfg.loaded_crouch_low_demand_gain, device=device, dtype=dtype
            ).unsqueeze(0)
            self._loaded_crouch_vmax = torch.tensor(
                cfg.loaded_crouch_vmax_rad_s, device=device, dtype=dtype
            ).unsqueeze(0)
            self._loaded_return_gain = torch.tensor(
                cfg.loaded_return_low_demand_gain, device=device, dtype=dtype
            ).unsqueeze(0)
            self._loaded_return_vmax = torch.tensor(
                cfg.loaded_return_vmax_rad_s, device=device, dtype=dtype
            ).unsqueeze(0)
            self._loaded_direction_weight = torch.tensor(
                cfg.loaded_direction_conditioning_weight, device=device, dtype=dtype
            ).unsqueeze(0)
            self._loaded_return_tau_scale = torch.tensor(
                cfg.loaded_return_tau_scale, device=device, dtype=dtype
            ).unsqueeze(0)
        else:
            ones = torch.ones((1, n), device=device, dtype=dtype)
            self._loaded_crouch_sign = ones
            self._loaded_crouch_gain = ones
            self._loaded_crouch_vmax = torch.full_like(ones, float("inf"))
            self._loaded_return_gain = ones
            self._loaded_return_vmax = torch.full_like(ones, float("inf"))
            self._loaded_direction_weight = torch.zeros_like(ones)
            self._loaded_return_tau_scale = ones

        self._drive_targets = self._default.clone()
        self._delayed_policy_targets = self._default.clone()
        self._previous_delayed_policy_targets = self._default.clone()
        self._velocity_cap_rad_s = torch.zeros_like(self._default)
        self._stage_a_velocity_candidate_rad_s = torch.zeros_like(self._default)
        self._loaded_velocity_cap_rad_s = torch.zeros_like(self._default)
        self._reference_speed_rad_s = torch.zeros_like(self._default)
        self._loaded_envelope_active = torch.zeros_like(self._default, dtype=torch.bool)
        self._loaded_direction_return = torch.zeros_like(self._default, dtype=torch.bool)

        self._sampled_tau_s = self._tau_median.expand(self.num_envs, -1).clone()
        self._sampled_velocity_scale = torch.ones_like(self._default)
        self._sampled_loaded_velocity_scale = torch.ones_like(self._default)
        self._sampled_response_delay_s = torch.zeros_like(self._default)
        self._sampled_total_delay_s = torch.zeros_like(self._default)
        self._sampled_delay_steps = torch.zeros_like(self._default, dtype=torch.long)

        max_delay_s = (
            float(cfg.bus_phase_delay_s_range[1])
            + float(cfg.response_delay_s_range[1]) * float(cfg.response_delay_scale)
        )
        self._history_len = max(3, int(math.ceil(max_delay_s / self._physics_dt)) + 3)
        self._target_history = self._default.unsqueeze(-1).repeat(1, 1, self._history_len)
        self._history_cursor = 0

        self._sample_response_parameters(slice(None))

    @staticmethod
    def _validate_len(name: str, values, expected: int) -> None:
        if len(values) != expected:
            raise ValueError(f"{name} length {len(values)} does not match joint count {expected}")

    @classmethod
    def _validate_loaded_cfg(cls, cfg, expected: int) -> None:
        names = (
            "loaded_crouch_direction_sign",
            "loaded_crouch_low_demand_gain",
            "loaded_crouch_vmax_rad_s",
            "loaded_return_low_demand_gain",
            "loaded_return_vmax_rad_s",
            "loaded_direction_conditioning_weight",
            "loaded_return_tau_scale",
        )
        for name in names:
            values = getattr(cfg, name)
            if values is None:
                raise ValueError(f"{name} must be provided when loaded_envelope_enabled=True")
            cls._validate_len(name, values, expected)
        if any(sign not in (-1, 1) for sign in cfg.loaded_crouch_direction_sign):
            raise ValueError("loaded_crouch_direction_sign values must be +/-1")
        if any(not 0.0 <= weight <= 1.0 for weight in cfg.loaded_direction_conditioning_weight):
            raise ValueError("loaded_direction_conditioning_weight values must be in [0, 1]")
        if any(value < 1.0 for value in cfg.loaded_return_tau_scale):
            raise ValueError("loaded_return_tau_scale values must be >= 1.0")

    @property
    def drive_targets(self) -> torch.Tensor:
        """Filtered position targets actually sent to the PhysX joint drives."""
        return self._drive_targets

    @property
    def delayed_policy_targets(self) -> torch.Tensor:
        """Residual policy targets after the sampled command-history delay."""
        return self._delayed_policy_targets

    @property
    def sampled_tau_s(self) -> torch.Tensor:
        return self._sampled_tau_s

    @property
    def sampled_total_delay_s(self) -> torch.Tensor:
        return self._sampled_total_delay_s

    @property
    def sampled_velocity_scale(self) -> torch.Tensor:
        return self._sampled_velocity_scale

    @property
    def sampled_loaded_velocity_scale(self) -> torch.Tensor:
        return self._sampled_loaded_velocity_scale

    @property
    def velocity_cap_rad_s(self) -> torch.Tensor:
        return self._velocity_cap_rad_s

    @property
    def stage_a_velocity_candidate_rad_s(self) -> torch.Tensor:
        return self._stage_a_velocity_candidate_rad_s

    @property
    def loaded_velocity_cap_rad_s(self) -> torch.Tensor:
        return self._loaded_velocity_cap_rad_s

    @property
    def reference_speed_rad_s(self) -> torch.Tensor:
        return self._reference_speed_rad_s

    @property
    def loaded_envelope_active(self) -> torch.Tensor:
        return self._loaded_envelope_active

    @property
    def loaded_direction_return(self) -> torch.Tensor:
        return self._loaded_direction_return

    @property
    def loaded_envelope_enabled(self) -> bool:
        return self._loaded_envelope_enabled

    @property
    def actuator_model_name(self) -> str:
        return self.cfg.actuator_model_name

    @property
    def actuator_model_stage(self) -> str:
        return self.cfg.actuator_model_stage

    def process_actions(self, actions: torch.Tensor):
        # Preserve v1.2.3 network semantics exactly. Delay is modeled in physical
        # target history, not with the legacy one-step action switch.
        self._raw_actions[:] = actions
        self._previous_bounded_actions[:] = self._bounded_actions
        self._bounded_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._applied_bounded_actions[:] = self._bounded_actions
        self._processed_actions[:] = self._default + self._bounded_actions * self._residual_scale
        self._processed_actions[:] = torch.clamp(self._processed_actions, self._lower, self._upper)

    def apply_actions(self):
        # Isaac Lab processes policy actions once per env step and calls apply_action
        # once per physics decimation step. The history therefore advances at sim dt.
        cursor = self._history_cursor
        self._target_history[:, :, cursor] = self._processed_actions
        gather_index = torch.remainder(cursor - self._sampled_delay_steps, self._history_len)
        self._delayed_policy_targets[:] = torch.gather(
            self._target_history, 2, gather_index.unsqueeze(-1)
        ).squeeze(-1)
        self._history_cursor = (cursor + 1) % self._history_len

        # Static-gain proxy is applied to the residual about q_default. This is not
        # interpreted as physical stiffness.
        servo_goal = self._default + self._static_gain * (
            self._delayed_policy_targets - self._default
        )
        servo_goal = torch.clamp(servo_goal, self._lower, self._upper)

        error = servo_goal - self._drive_targets
        abs_error = torch.abs(error)
        effective_error = torch.sign(error) * torch.relu(abs_error - self._deadzone)

        # Stage A: derive a velocity cap from measured unloaded step amplitude.
        command_delta_signed = (
            self._delayed_policy_targets - self._previous_delayed_policy_targets
        )
        command_delta = torch.abs(command_delta_signed)
        changed = command_delta > float(self.cfg.command_change_epsilon_rad)
        response_amplitude = torch.maximum(command_delta, abs_error)
        stage_a_candidate = self._interpolate_velocity(response_amplitude)
        stage_a_candidate *= self._sampled_velocity_scale
        self._stage_a_velocity_candidate_rad_s[:] = stage_a_candidate
        candidate_velocity = stage_a_candidate

        if self._loaded_envelope_enabled:
            # Stage B: use target-to-target reference rate at the policy cadence.
            reference_speed = command_delta / self._policy_dt
            is_crouch = command_delta_signed * self._loaded_crouch_sign > 0.0
            is_return = command_delta_signed * self._loaded_crouch_sign < 0.0

            avg_gain = 0.5 * (self._loaded_crouch_gain + self._loaded_return_gain)
            selected_gain = torch.where(
                is_crouch, self._loaded_crouch_gain, self._loaded_return_gain
            )
            effective_gain = avg_gain + self._loaded_direction_weight * (
                selected_gain - avg_gain
            )

            avg_vmax = 0.5 * (self._loaded_crouch_vmax + self._loaded_return_vmax)
            selected_vmax = torch.where(
                is_crouch, self._loaded_crouch_vmax, self._loaded_return_vmax
            )
            effective_vmax = avg_vmax + self._loaded_direction_weight * (
                selected_vmax - avg_vmax
            )

            loaded_candidate = torch.minimum(
                effective_gain * reference_speed,
                effective_vmax,
            )
            loaded_candidate *= self._sampled_loaded_velocity_scale

            # The loaded fit is defined for trajectory motion. If there is no fresh
            # target change (for example immediately after reset), do not create a
            # zero cap; fall back to the Stage-A response candidate.
            loaded_candidate = torch.where(changed, loaded_candidate, stage_a_candidate)
            combined_candidate = torch.minimum(stage_a_candidate, loaded_candidate)
            candidate_velocity = combined_candidate

            self._loaded_velocity_cap_rad_s[:] = torch.where(
                changed, loaded_candidate, self._loaded_velocity_cap_rad_s
            )
            self._reference_speed_rad_s[:] = torch.where(
                changed, reference_speed, self._reference_speed_rad_s
            )
            self._loaded_envelope_active[:] = torch.where(
                changed,
                loaded_candidate < (stage_a_candidate - 1.0e-6),
                self._loaded_envelope_active,
            )
            self._loaded_direction_return[:] = torch.where(
                changed, is_return, self._loaded_direction_return
            )

        self._velocity_cap_rad_s[:] = torch.where(
            changed, candidate_velocity, self._velocity_cap_rad_s
        )
        needs_cap = (self._velocity_cap_rad_s <= 0.0) & (abs_error > self._deadzone)
        self._velocity_cap_rad_s[:] = torch.where(
            needs_cap, candidate_velocity, self._velocity_cap_rad_s
        )
        self._previous_delayed_policy_targets[:] = self._delayed_policy_targets

        effective_tau = self._sampled_tau_s
        if self._loaded_envelope_enabled:
            return_tau_scale = torch.where(
                self._loaded_direction_return,
                self._loaded_return_tau_scale,
                torch.ones_like(self._loaded_return_tau_scale),
            )
            effective_tau = effective_tau * return_tau_scale

        alpha = 1.0 - torch.exp(
            -self._physics_dt / torch.clamp(effective_tau, min=1.0e-4)
        )
        delta = alpha * effective_error
        max_delta = self._velocity_cap_rad_s * self._physics_dt
        delta = torch.clamp(delta, min=-max_delta, max=max_delta)
        self._drive_targets[:] = torch.clamp(
            self._drive_targets + delta, self._lower, self._upper
        )

        self._asset.set_joint_position_target(self._drive_targets, joint_ids=self._joint_ids)

    def _interpolate_velocity(self, amplitude_rad: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(
            amplitude_rad,
            min=float(self._velocity_knots[0]),
            max=float(self._velocity_knots[-1]),
        )
        # Bucketize against interior knots -> segment index in [0, K-2].
        idx = torch.bucketize(x, self._velocity_knots[1:-1])
        curves = self._velocity_curves.expand(self.num_envs, -1, -1)
        y0 = torch.gather(curves, 2, idx.unsqueeze(-1)).squeeze(-1)
        y1 = torch.gather(curves, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
        x0 = self._velocity_knots[idx]
        x1 = self._velocity_knots[idx + 1]
        fraction = (x - x0) / torch.clamp(x1 - x0, min=1.0e-8)
        return y0 + fraction * (y1 - y0)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)

        current = self._asset.data.joint_pos[:, self._joint_ids][env_ids]
        self._drive_targets[env_ids] = current
        self._delayed_policy_targets[env_ids] = current
        self._previous_delayed_policy_targets[env_ids] = current
        self._velocity_cap_rad_s[env_ids] = 0.0
        self._stage_a_velocity_candidate_rad_s[env_ids] = 0.0
        self._loaded_velocity_cap_rad_s[env_ids] = 0.0
        self._reference_speed_rad_s[env_ids] = 0.0
        self._loaded_envelope_active[env_ids] = False
        self._loaded_direction_return[env_ids] = False
        self._target_history[env_ids] = current.unsqueeze(-1).expand(
            *current.shape, self._history_len
        )
        self._sample_response_parameters(env_ids)

    def _sample_response_parameters(self, env_ids) -> None:
        current_tau = self._sampled_tau_s[env_ids]
        if self.cfg.randomize_tau:
            tau_low = self._tau_p10.expand(self.num_envs, -1)[env_ids]
            tau_high = self._tau_p90.expand(self.num_envs, -1)[env_ids]
            sampled_tau = tau_low + torch.rand_like(current_tau) * (tau_high - tau_low)
        else:
            sampled_tau = self._tau_median.expand(self.num_envs, -1)[env_ids]
        self._sampled_tau_s[env_ids] = sampled_tau

        current_velocity_scale = self._sampled_velocity_scale[env_ids]
        if self.cfg.randomize_velocity_scale:
            low, high = self.cfg.velocity_scale_range
            sampled_velocity_scale = float(low) + torch.rand_like(current_velocity_scale) * (
                float(high) - float(low)
            )
        else:
            sampled_velocity_scale = torch.ones_like(current_velocity_scale)
        self._sampled_velocity_scale[env_ids] = sampled_velocity_scale

        current_loaded_scale = self._sampled_loaded_velocity_scale[env_ids]
        if self._loaded_envelope_enabled and self.cfg.randomize_loaded_velocity_scale:
            low, high = self.cfg.loaded_velocity_scale_range
            sampled_loaded_scale = float(low) + torch.rand_like(current_loaded_scale) * (
                float(high) - float(low)
            )
        else:
            sampled_loaded_scale = torch.ones_like(current_loaded_scale)
        self._sampled_loaded_velocity_scale[env_ids] = sampled_loaded_scale

        # Bus phase is shared across one robot/environment; response-delay proxy is
        # sampled per joint. response_delay_scale is explicit because Track 2 may
        # later separate encoder-read observability from true mechanical onset.
        current_response_delay = self._sampled_response_delay_s[env_ids]
        if self.cfg.randomize_response_delay:
            r0, r1 = self.cfg.response_delay_s_range
            response_delay = float(r0) + torch.rand_like(current_response_delay) * (
                float(r1) - float(r0)
            )
        else:
            response_delay = torch.full_like(
                current_response_delay, float(self.cfg.response_delay_s_nominal)
            )
        self._sampled_response_delay_s[env_ids] = response_delay

        env_count = response_delay.shape[0]
        b0, b1 = self.cfg.bus_phase_delay_s_range
        if self.cfg.randomize_bus_phase:
            bus = float(b0) + torch.rand((env_count, 1), device=self.device) * (
                float(b1) - float(b0)
            )
        else:
            bus = torch.full(
                (env_count, 1),
                float(self.cfg.bus_phase_delay_s_nominal),
                device=self.device,
            )

        total = bus + response_delay * float(self.cfg.response_delay_scale)
        self._sampled_total_delay_s[env_ids] = total
        self._sampled_delay_steps[env_ids] = torch.clamp(
            torch.round(total / self._physics_dt).to(torch.long),
            min=0,
            max=self._history_len - 2,
        )


@configclass
class ST3215MeasuredResidualJointPositionActionCfg(ActionTermCfg):
    """Configuration for Stage-A ST3215 response and optional v1.4 loaded extension."""

    class_type: type[ActionTerm] = ST3215MeasuredResidualJointPositionAction

    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    lower_limits: list[float] = MISSING
    upper_limits: list[float] = MISSING
    preserve_order: bool = True
    residual_scale_rad: float | list[float] = 0.20
    # Interface compatibility with v1.2.3 diagnostics. The measured history model
    # replaces the old 0/1-step latency switch.
    one_step_delay_probability: float = 0.0

    actuator_model_name: str = "track1_actuator_identification_v2_stage_a"
    actuator_model_stage: str = "stage_a"
    velocity_amplitude_knots_rad: list[float] = MISSING
    velocity_curves_rad_s: list[list[float]] = MISSING
    tau_median_s: list[float] = MISSING
    tau_p10_s: list[float] = MISSING
    tau_p90_s: list[float] = MISSING
    static_gain: list[float] = MISSING
    small_signal_error_floor_rad: list[float] = MISSING
    center_hysteresis_span_rad: list[float] = MISSING

    bus_phase_delay_s_range: tuple[float, float] = (0.0, 0.020)
    bus_phase_delay_s_nominal: float = 0.010
    response_delay_s_range: tuple[float, float] = (0.02344220505, 0.0489510989)
    response_delay_s_nominal: float = 0.032590821
    response_delay_scale: float = 1.0
    velocity_scale_range: tuple[float, float] = (0.95, 1.05)
    policy_dt_s_nominal: float = 0.020

    randomize_tau: bool = True
    randomize_velocity_scale: bool = True
    randomize_response_delay: bool = True
    randomize_bus_phase: bool = True
    command_change_epsilon_rad: float = 1.0e-5

    # Optional Stage-B loaded trajectory calibration layer.
    loaded_envelope_enabled: bool = False
    loaded_dataset_name: str = ""
    loaded_envelope_combination_rule: str = ""
    loaded_crouch_direction_sign: list[int] | None = None
    loaded_crouch_low_demand_gain: list[float] | None = None
    loaded_crouch_vmax_rad_s: list[float] | None = None
    loaded_return_low_demand_gain: list[float] | None = None
    loaded_return_vmax_rad_s: list[float] | None = None
    loaded_direction_conditioning_weight: list[float] | None = None
    loaded_return_tau_scale: list[float] | None = None
    loaded_velocity_scale_range: tuple[float, float] = (0.95, 1.05)
    randomize_loaded_velocity_scale: bool = True
