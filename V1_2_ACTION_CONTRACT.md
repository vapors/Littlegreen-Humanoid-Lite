# Berkeley Humanoid Lite v1.2 Action Contract

## Purpose

v1.2 replaces the legacy unbounded action path used by `Velocity-Lilgreen-Humanoid-v0` with an explicit hardware-aligned contract shared by:

- `Velocity-Lilgreen-Stand-v0`
- `Velocity-Lilgreen-Hardware-v0`

The legacy task is left unchanged for reproducibility.

## Network action semantics

The actor still emits 12 floating-point values in canonical physical-servo order:

1. left hip roll
2. left hip yaw
3. left hip pitch
4. left knee pitch
5. left ankle pitch
6. left ankle roll
7. right hip roll
8. right hip yaw
9. right hip pitch
10. right knee pitch
11. right ankle pitch
12. right ankle roll

Each raw network value is clamped to `[-1, +1]` before the physical target is formed.

For each joint:

```text
raw network output
        |
        v
clamp [-1, +1]
        |
        v
bounded normalized action
        |
        +---- a < 0 ---> interpolate default -> hardware lower limit
        |
        +---- a >= 0 --> interpolate default -> hardware upper limit
        |
        v
physical target radians
```

Thus:

```text
a = -1  -> physical lower limit
a =  0  -> training default pose
a = +1  -> physical upper limit
```

This is asymmetric around the training default because the real joint travel is asymmetric.

## Previous-action observation

The last 12 values of the 45-D actor observation are now the **bounded normalized action**, not the unbounded raw network output.

The actor observation remains:

```text
0:3    commanded velocity
3:6    base angular velocity
6:9    projected gravity
9:21   relative joint position
21:33  joint velocity
33:45  previous bounded normalized action
```

This keeps the observation dimension at 45 while preventing the raw-action feedback loop observed in the v1.1 deployment tests.

## OLD Physical joint envelope

The action term uses the physical limits already used by the Orange Pi ROS 2 joint map:

```text
joint                 lower rad   default rad   upper rad
left hip roll          -0.698       0.000         0.524
left hip yaw           -0.087       0.000         1.221
left hip pitch         -1.483      -0.100         0.785
left knee               0.000       0.400         1.483
left ankle pitch       -0.384      -0.300         1.134
left ankle roll        -0.524       0.000         0.785
right hip roll         -0.698       0.000         0.524
right hip yaw          -0.087       0.000         1.221
right hip pitch        -1.483      -0.100         0.785
right knee              0.000       0.400         1.483
right ankle pitch      -0.384      -0.300         1.134
right ankle roll       -0.524       0.000         0.785
```


# Berkeley ROS 2 Workspace v2.5 — Measured Hardware Joint Contract

## Purpose

v2.5 propagates the measured 12-joint hardware envelope into the active ROS 2 control path without adding a new runtime adapter. The source measurement used a physical endpoint shim and a fixed **10-step inward margin per endpoint**.

## Runtime authority by layer

1. `bhl_st3215_driver/config/servo_map.yaml`
   - native ST3215 IDs, signs, calibrated centers, safe radian limits, safe raw-step limits
   - final native driver conversion and clamping
2. `berkeley_biped_pkg/src/configs/joint_map.yaml`
   - canonical action/joint ordering and default pose
   - policy-node target clipping
   - `pd_controller_pkg` safety/reference clipping
3. `bhl_st3215_driver/config/track1_action_contract_v3.yaml`
   - ROS-side mirror used only by the standing-load runner contract audit
4. `berkeley_biped_pkg/src/configs/joint_limits.yaml`
   - compatibility/documentation mirror; not loaded by current runtime nodes

No additional runtime adapter or limit service was added.

## Authoritative safe limits

| Joint | Lower rad | Upper rad |
|---|---:|---:|
| `leg_left_hip_roll_joint` | -0.694893297 | 0.780796221 |
| `leg_left_hip_yaw_joint` | -0.088970886 | 0.644271931 |
| `leg_left_hip_pitch_joint` | -1.922077927 | 0.681087470 |
| `leg_left_knee_pitch_joint` | 0.134990309 | 2.235010008 |
| `leg_left_ankle_pitch_joint` | -0.809941856 | 0.710233105 |
| `leg_left_ankle_roll_joint` | -0.513883564 | 0.912718569 |
| `leg_right_hip_roll_joint` | -0.874369049 | 0.642737950 |
| `leg_right_hip_yaw_joint` | -0.056757289 | 0.701029220 |
| `leg_right_hip_pitch_joint` | -1.991107063 | 0.546097160 |
| `leg_right_knee_pitch_joint` | 0.171805848 | 2.241145931 |
| `leg_right_ankle_pitch_joint` | -0.845223414 | 0.819145741 |
| `leg_right_ankle_roll_joint` | -0.443320448 | 1.061514705 |

## Track 1 propagation

For Berkeley-Humanoid-Lite, update only `HARDWARE_LOWER_LIMIT_RAD` and `HARDWARE_UPPER_LIMIT_RAD` in the existing `hardware_contract.py`, preserving the rest of that file's API and constants. Hardware-aligned tasks consume those arrays for action processing and for startup simulation joint-limit writes.

## Deployment caution

Changing the hardware limit contract can change command clipping for a previously trained policy. Keep old policies paired with their original contract for reproducibility; use the v2.5 measured contract for new hardware-aligned training and validation before deploying a new policy.


## Raw-action health penalties

v1.2 keeps the raw actor output available internally for diagnostics and applies three different regularizers:

- bounded action-rate penalty;
- bounded action-magnitude penalty;
- raw action excess penalty, active only beyond `abs(raw) > 1`.

This lets the actor use the full normalized action range while explicitly discouraging the large `-20`, `-40`, and `-60` raw outputs observed in v1.1.

## Optional one-policy-step latency

The action term can assign a one-policy-step delay to a fraction of environments. The Hardware-v0 curriculum increases this probability from 0 to 25% over its four stages.

At the v1.2 50 Hz policy rate, one step is 20 ms.

## Deployment warning

The v1.2 exported policy contract is **not compatible with the legacy ROS 2 action transform**:

```text
q_target = q_default + 0.25 * raw_action
```

`export_policy.py` now records `action_contract_version: 2` and the physical lower/default/upper arrays in `policy.yaml`. It also blocks the automatic copy into the ROS 2 deployment directory unless the explicit override flag is supplied.

The Orange Pi policy node must implement the same bounded asymmetric transform before a v1.2 policy is physically released.
