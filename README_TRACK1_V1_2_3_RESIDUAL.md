# Berkeley Humanoid Lite Track 1 — v1.2.3 Residual Action Patch

This patch is based on `Berkeley-Humanoid-Lite_v1_2_2.zip` and introduces the first v1.2.3 standing experiment without overwriting the reproducible v1.2.2 task semantics.

## New experiment task

```text
Velocity-Lilgreen-Stand-Residual-v0
```

Action contract v3:

```text
raw network action
    -> raw-excess penalty remains unchanged
    -> clamp to [-1, +1]
    -> optional one-policy-step delay in normalized action space
    -> q_target = q_default + 0.20 * bounded_action
    -> hard clip to calibrated physical joint limits
```

The existing `Velocity-Lilgreen-Stand-v0` and `Velocity-Lilgreen-Hardware-v0` registrations remain unchanged.

## Diagnostics changes

Standing success is split into:

```text
stable standing:
  standing command
  + upright
  + quiet XY motion
  + quiet yaw motion
  + both feet in contact

posture conformity:
  stable standing
  + max actionable-joint error from q_default < 0.20 rad
```

New standing-only per-joint diagnostics include raw action magnitude/excess, bounded action magnitude, target residual, physical target-limit occupancy, posture error, torque mean, soft-torque occupancy, and hard torque-limit occupancy.

Backward-compatible diagnostic keys are retained for existing dashboards; their historical posture-conforming meaning is preserved.

## Base COM measurement

Run a deterministic q_default hold measurement before changing the current 0.656 m reward target:

```bash
python scripts/rsl_rl/measure_nominal_pose.py \
  --task Velocity-Lilgreen-Stand-Residual-v0 \
  --num_envs 64 \
  --settle_seconds 5 \
  --sample_seconds 2 \
  --output logs/nominal_pose_v123.json \
  --headless
```

The script disables command motion and reset/randomization terms that would contaminate the measurement, holds zero normalized action, and reports root-frame height, root-body COM height, joint posture RMS, and max joint error over the sampling window.

## Export contract changes

`export_policy.py` now emits explicit action contract v3 metadata:

```text
action_contract_version: 3
action_contract_name: bounded_default_centered_symmetric_residual
action_transform: bounded_default_centered_symmetric_residual
action_residual_scale_rad: [0.20, ... x12]
action_default_rad: [...]
action_target_lower_rad: calibrated physical lower limits
action_target_upper_rad: calibrated physical upper limits
action_nominal_residual_lower_rad: clipped q_default - scale
action_nominal_residual_upper_rad: clipped q_default + scale
previous_action_observation: bounded_normalized_action
```

The old full-range asymmetric mapping remains explicitly identified as contract v2. ROS 2 deployment copy remains blocked for nonlegacy contracts unless the generic compatibility override is deliberately supplied after the deployment node supports the exported transform.

## Validation performed here

- Python syntax compilation passed for all changed/new Python files.
- Scalar residual reference mapping tests passed for center, endpoints, raw-action clamp, and physical-limit clip.
- Isaac Lab runtime validation still needs to be run on the user's Isaac Lab machine.
