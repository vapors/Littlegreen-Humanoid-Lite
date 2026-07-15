# Berkeley Humanoid Lite v1.2 Changes

## Summary

v1.2 is a sim-to-real training revision built around two new tasks while preserving the original v1.1 task as a baseline.

### New registered tasks

```text
Velocity-Lilgreen-Stand-v0
Velocity-Lilgreen-Hardware-v0
```

### Preserved baseline

```text
Velocity-Lilgreen-Humanoid-v0
```

The baseline config and runner remain registered under their original names.

## Major changes

### 1. Bounded hardware action contract

New custom action term:

```text
BoundedDefaultCenteredJointPositionAction
```

Behavior:

- retains raw actor output for diagnostics;
- clamps normalized action to `[-1, +1]`;
- maps zero exactly to the training default pose;
- maps negative action asymmetrically toward the physical lower limit;
- maps positive action asymmetrically toward the physical upper limit;
- exposes bounded actions for the previous-action observation;
- supports optional per-environment one-policy-step latency.

### 2. Previous action repaired

The actor observation still has 45 values, but observation elements 33:45 are now the previous/current bounded normalized action rather than the unbounded raw network output.

### 3. 50 Hz policy rate

The two new tasks use:

```text
physics: 200 Hz
policy/action: 50 Hz
```

The legacy task remains at 25 Hz.

### 4. Hardware joint envelope

Startup events apply the same physical 12-joint radian limits used by the ROS 2 hardware map.

### 5. ST3215 first-pass actuator limits

The 12 V / 30 kg.cm servo model starts with:

```text
2.94 N.m peak effort limit
4.72 rad/s no-load velocity limit
```

### 6. Stand-v0

Standing-first task with:

- 75% exact standing environments;
- small command envelope;
- gentle reset distribution;
- narrow material and actuator randomization;
- no periodic pushes;
- lower observation noise;
- explicit stand-gated posture/contact/slip rewards;
- bounded-action health penalties.

### 7. Hardware-v0

Four-stage curriculum controlling:

- standing fraction;
- linear/yaw command range;
- root reset velocity;
- joint reset offset;
- push severity;
- one-policy-step command latency probability.

### 8. Live diagnostics

A `PolicyDiagnostics` curriculum term logs:

- raw action mean abs/std/min/max;
- raw action excess fraction;
- bounded saturation fraction;
- physical target-limit fraction;
- qdot limit fraction;
- standing environment fraction;
- 10 s and 20 s standing success;
- zero-command XY drift;
- standing foot slip;
- linear-velocity tracking RMSE;
- yaw-rate tracking RMSE;
- latency-randomized environment fraction;
- per-joint raw magnitude and excess fraction.

### 9. Offline rollout analyzer

New script:

```text
scripts/rsl_rl/analyze_policy_v1_2.py
```

It writes per-joint JSON/CSV diagnostics including raw p95/p99, saturation, target limits, measured q/qdot, and torque usage.

### 10. PPO configs

New runner configs:

```text
LilgreenStandPPORunnerCfg
LilgreenHardwarePPORunnerCfg
```

Both share experiment root:

```text
logs/rsl_rl/lilgreen_v1_2
```

and identical network architecture for checkpoint transfer.

### 11. Export metadata and deployment guard

`export_policy.py` now understands both:

- legacy action contract v1;
- bounded asymmetric action contract v2.

Contract-v2 export records target lower/default/upper arrays and blocks automatic ROS 2 deployment copying unless explicitly overridden.

## New source files

```text
.../velocity/mdp/hardware_contract.py
.../velocity/mdp/hardware_actions.py
.../velocity/mdp/hardware_rewards.py
.../velocity/mdp/diagnostics.py

.../config/lilgreen_humanoid/v1_2_common.py
.../config/lilgreen_humanoid/env_cfg_stand.py
.../config/lilgreen_humanoid/env_cfg_hardware.py

scripts/rsl_rl/analyze_policy_v1_2.py

V1_2_ACTION_CONTRACT.md
V1_2_TRAINING_GUIDE.md
V1_2_CHANGES.md
```

## Validation performed outside Isaac Lab runtime

- all modified/new Python files compile with `compileall`;
- pure action-contract endpoint/default/clamping checks pass;
- task registration names and runner entry points were statically checked;
- exporter syntax and v1/v2 action metadata paths were statically checked.

A real Isaac Lab 2.1.0 smoke build/run is still required on the GPU host because this packaging environment does not contain Isaac Sim/Isaac Lab runtime.

### 12. Checkpoint video playback rate

`train_eval.py` now derives MP4 FPS from `sim.dt * decimation`, so legacy 25 Hz runs still save at 25 FPS while the v1.2 50 Hz tasks save at 50 FPS instead of appearing in slow motion.
