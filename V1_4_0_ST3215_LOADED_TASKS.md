# Berkeley Humanoid Lite v1.4.0 Loaded ST3215 Task Family

## Versioning strategy

v1.4.0 is additive. It starts from the supplied v1.3.1 source tree, including the
updated `hardware_contract.py` physical joint limits, and preserves the existing
v1.2.3 and v1.3.x task registrations.

Existing reference tasks:

- `Velocity-Lilgreen-Stand-v0`
- `Velocity-Lilgreen-Hardware-v0`
- `Velocity-Lilgreen-Stand-ST3215-v0`
- `Velocity-Lilgreen-Hardware-ST3215-v0`

New v1.4.0 tasks:

- `Velocity-Lilgreen-Stand-ST3215-Loaded-v0`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v0`

The v1.4 task family uses experiment root:

```text
lilgreen_v1_4_0_st3215_loaded
```

## Policy and deployment contract

v1.4.0 does **not** change the actor/deployment interface:

- 45-D actor observation
- 12-D policy action in canonical hardware order
- 50 Hz policy rate
- action contract v3
- bounded normalized action in `[-1, +1]`
- `q_target = q_default + 0.20 * bounded_action`
- final target clipped to the updated physical joint limits
- nominal q_default root-body COM target `0.4899105727672577 m`

The loaded actuator model is a training-environment layer after the policy target
mapping. It does not change ONNX action semantics.

## Model stack

v1.4.0 preserves the v1.3.1 Stage-A suspended single-joint model:

1. bus-phase wait;
2. configurable command-to-observed-motion response-delay proxy;
3. per-joint tau-equivalent response randomization;
4. per-joint amplitude-to-velocity curves;
5. static-gain response proxy;
6. small-signal residual-error and center-hysteresis proxies;
7. narrow velocity-response scale randomization.

It then adds Stage B using the updated loaded standing-transition synthesis:

```text
v_loaded = min(gain * abs(v_ref), vmax)
v_final_cap = min(v_stage_a_cap, v_loaded)
```

The empirical loaded fits are trajectory envelopes, not intrinsic ST3215 free-speed
limits or torque-speed constants.

## Conservative direction conditioning

The v2 standing-load synthesis found the clearest persistent direction-conditioned
behavior in knee pitch. Therefore the first v1.4 baseline uses:

- knee pitch: crouch/stand-return fitted envelopes selected by motion direction;
- hip pitch: symmetric average of crouch/return fits, emphasizing speed dependence;
- ankle pitch: symmetric average, because the older broad return penalty did not
  persist across the updated sweep;
- roll families: conservative symmetric loaded envelope;
- hip yaw: cautious symmetric treatment because the pose ladder excites it less cleanly.

A 1.10x tau-response proxy is applied only to knee stand-return motion as a modest
first-pass representation of the persistent loaded return asymmetry. It is explicitly
not a physical damping coefficient.

The loaded envelope receives independent episode-level scale randomization in
`[0.95, 1.05]`.

## What v1.4.0 deliberately does not infer

The loaded standing campaign does not identify:

- physical stiffness in N*m/rad;
- physical damping in N*m*s/rad;
- joint torque from `load_ratio`;
- balance stability from the transition archive, because synchronized IMU data is absent.

The loaded dataset is used as a trajectory-response calibration and validation layer.

## New diagnostics

In addition to the v1.3.x action, torque, standing, posture, COM-height, and actuator-lag
metrics, v1.4.0 reports:

- `actuator_loaded_velocity_cap_mean_rad_s`
- `actuator_final_velocity_cap_mean_rad_s`
- `actuator_loaded_envelope_active_fraction`
- `actuator_reference_speed_mean_rad_s`
- `actuator_loaded_velocity_scale_mean`

Standing-only per-joint metrics include loaded-envelope active fraction, final velocity
cap, loaded velocity cap, and reference speed.

Use `scripts/rsl_rl/analyze_policy_v1_4_0.py` for offline rollout analysis.

## First smoke run

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v0 \
  --num_envs 4096 \
  --max_iterations 100 \
  --seed 42 \
  --run_name stand_st3215_loaded_v140_smoke_seed42 \
  --headless
```

## Fresh 5000-iteration baseline

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name stand_st3215_loaded_v140_fresh_seed42 \
  --headless
```

## Analyze a checkpoint

```bash
python scripts/rsl_rl/analyze_policy_v1_4_0.py \
  --task Velocity-Lilgreen-Stand-ST3215-Loaded-v0 \
  --load_run <run-name> \
  --checkpoint model_499.pt \
  --num_envs 256 \
  --steps 1500 \
  --headless
```

Recommended analysis checkpoints for a full baseline remain approximately 500, 1000,
2500, and 4999 iterations.
