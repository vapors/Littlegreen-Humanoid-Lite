# Berkeley Humanoid Lite v1.2.2 changes

v1.2.2 is a focused Stand-v0 refinement based on the first 500-iteration run and deterministic rollout analysis.

## Action regularization

Stand-v0:
- `bounded_action_l2`: `-0.005 -> -0.0075`
- `raw_action_excess_l2`: `-0.050 -> -0.100`
- `bounded_action_rate_l2`: unchanged at `-0.020`

Hardware-v0 receives a gentler matching update for later use:
- `bounded_action_l2`: `-0.003 -> -0.005`
- `raw_action_excess_l2`: `-0.050 -> -0.080`

## Base-height framing

The old reward used `root_pos_w[:, 2]`, but the Lilgreen root/link frame is near ground level. v1.2.2 uses `root_com_pos_w[:, 2]`, which tracks the physical root-body center of mass.

Initial reference:
- root-frame standing height measured in the first run: about `0.0183 m`
- base inertial COM Z offset in the robot URDF: `0.637874 m`
- first-pass COM target: `0.656 m`

Stand-v0 uses:
- desired COM height: `0.656 m`
- Gaussian std: `0.05 m`
- weight: `+1.0`

Hardware-v0 uses:
- desired COM height: `0.656 m`
- Gaussian std: `0.06 m`
- weight: `+0.75`

Live diagnostics now report both root-frame and COM height so the target can be confirmed empirically.

## Soft torque utilization penalty

A new reward term penalizes only torque utilization above 70% of the modeled ST3215 peak torque.

With `tau_max = 2.94 N m`:

```
utilization = abs(tau) / tau_max
excess = relu(utilization - 0.70)
penalty = sum(excess^2)
```

Weights:
- Stand-v0: `-0.020`
- Hardware-v0: `-0.015`

The existing small `joint_torques_l2` penalty remains in place.

## Expanded policy diagnostics

Added:
- torque over soft threshold fraction
- torque limit fraction
- mean absolute torque
- per-joint soft torque fraction
- per-joint torque-limit fraction
- root-frame height mean
- base-COM height mean
- standing base-COM height mean
- standing base-COM height error
- standing upright gate fraction
- standing quiet-XY gate fraction
- standing quiet-yaw gate fraction
- standing near-default gate fraction
- standing both-feet gate fraction
- standing all-conditions gate fraction
- current max continuous standing success duration

## Offline analyzer additions

`analyze_policy_v1_2.py` now reports:
- soft torque utilization fraction globally and per joint
- standing gate fractions
- maximum continuous strict standing-success duration
- standing base-COM height and error

## Curriculum field fix included

The environment configuration field is `curriculum` (singular) for both Stand-v0 and Hardware-v0.
