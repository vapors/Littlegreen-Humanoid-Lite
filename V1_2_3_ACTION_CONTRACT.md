# Berkeley Humanoid Lite v1.2.3 Action Contract

## Canonical contract

`Velocity-Lilgreen-Stand-v0` and `Velocity-Lilgreen-Hardware-v0` use action contract v3.
For each of the 12 actionable joints:

```text
raw policy output a_raw
    -> a_bounded = clip(a_raw, -1, +1)
    -> q_nominal = q_default + 0.20 rad * a_bounded
    -> q_target = clip(q_nominal, physical_lower, physical_upper)
```

The residual scale is `0.20 rad` per joint. The final physical clip remains necessary because the calibrated hardware envelope is asymmetric around `q_default` for some joints.

## Observation semantics

The policy observation keeps the previous **bounded normalized action**, not the raw network output and not the physical position target. Raw output is retained separately for diagnostics and the raw-action-excess penalty.

## Canonical joint order

1. `leg_left_hip_roll_joint`
2. `leg_left_hip_yaw_joint`
3. `leg_left_hip_pitch_joint`
4. `leg_left_knee_pitch_joint`
5. `leg_left_ankle_pitch_joint`
6. `leg_left_ankle_roll_joint`
7. `leg_right_hip_roll_joint`
8. `leg_right_hip_yaw_joint`
9. `leg_right_hip_pitch_joint`
10. `leg_right_knee_pitch_joint`
11. `leg_right_ankle_pitch_joint`
12. `leg_right_ankle_roll_joint`

## Export/deployment contract

v1.2.3 exports explicit metadata including:

- `action_contract_version: 3`
- `action_contract_name: bounded_default_centered_symmetric_residual`
- `action_residual_scale_rad`
- `action_default_rad`
- calibrated physical target lower/upper arrays
- effective nominal residual lower/upper arrays after physical clipping
- `previous_action_observation: bounded_normalized_action`

The exporter blocks automatic ROS 2 deployment copy for nonlegacy contracts unless compatibility is explicitly acknowledged. A deployment node must implement the v3 transform before executing a v1.2.3 policy.
