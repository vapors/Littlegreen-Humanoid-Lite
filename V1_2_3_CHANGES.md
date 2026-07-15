# Berkeley Humanoid Lite v1.2.3 Changes

## Training semantics

- `Velocity-Lilgreen-Stand-v0` now uses action contract v3: a symmetric ±0.20 rad residual around `q_default`, followed by calibrated physical-limit clipping.
- `Velocity-Lilgreen-Hardware-v0` uses the same v3 action semantics so Stand-to-Hardware checkpoint continuation does not change action interpretation.
- The duplicate experimental `Velocity-Lilgreen-Stand-Residual-v0` task is no longer needed; the canonical Stand-v0 task is the residual baseline.
- The shared RSL-RL experiment namespace is `lilgreen_v1_2_3`.

## Nominal COM height

The deterministic reset-state measurement for q_default is:

```text
root-body COM height = 0.4899105727672577 m
```

This value is stored as `NOMINAL_QDEFAULT_BASE_COM_HEIGHT_M` and used by both Stand-v0 and Hardware-v0 standing-height reward/diagnostic terms.

## Diagnostics

- Stable standing is separated from posture conformity.
- 10 s and 20 s success metrics are reported separately for stability and posture-conforming stability.
- Standing-only per-joint diagnostics include raw and bounded action magnitude, raw excess, target residual magnitude, target-limit occupancy, joint-position error, mean absolute torque, soft-torque occupancy, and hard torque-limit occupancy.
- The offline analyzer is `scripts/rsl_rl/analyze_policy_v1_2_3.py`.

## Export metadata

`export_policy.py` emits an explicit action-contract v3 description and prevents the residual mapping from being confused with the old full-range asymmetric action contract.

## Preserved v1.2.2 shaping

The successful Stand-v0 action/torque health terms are preserved:

```text
bounded_action_rate_l2    -0.020
bounded_action_l2         -0.0075
raw_action_excess_l2      -0.100
dof_torques_l2            -1.0e-4
soft_torque_utilization   -0.020
```
