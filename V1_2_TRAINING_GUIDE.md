# Berkeley Humanoid Lite v1.2 Training Guide

## Task layout

v1.2 preserves the original baseline task and adds two new tasks:

```text
Velocity-Lilgreen-Humanoid-v0    legacy v1.1 baseline, unchanged
Velocity-Lilgreen-Stand-v0       standing-first pretraining
Velocity-Lilgreen-Hardware-v0    final staged hardware-oriented locomotion
```

The intended path is:

```text
Stand-v0
   |
   | checkpoint transfer
   v
Hardware-v0
   |
   | policy diagnostics + simulation validation
   v
export contract-v2 ONNX
   |
   | only after ROS 2 deployment transform is updated
   v
supported physical testing
```

Stand-v0 and Hardware-v0 use the same actor observation size, action size, and neural-network architecture, so checkpoints are structurally compatible.

## Policy rate

The two new v1.2 tasks use:

```text
physics dt      0.005 s  = 200 Hz
policy/action   decimation 4
policy dt       0.020 s  = 50 Hz
```

The legacy baseline remains at 25 Hz.

The 50 Hz rate was chosen to align one policy/action update with the validated 50 Hz Orange Pi command and feedback loop. This removes a deliberate 40 ms policy hold from the new sim-to-real path while preserving 200 Hz physics integration.

## 1. Smoke test task registration

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 64 \
  --max_iterations 5 \
  --run_name stand_smoke \
  --headless
```

Then smoke-test Hardware-v0:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-v0 \
  --num_envs 64 \
  --max_iterations 5 \
  --run_name hardware_smoke \
  --headless
```

Before a long run, verify that TensorBoard contains:

```text
Curriculum/policy_diagnostics/raw_action_mean_abs
Curriculum/policy_diagnostics/raw_action_excess_fraction
Curriculum/policy_diagnostics/bounded_saturation_fraction
Curriculum/policy_diagnostics/target_limit_fraction
Curriculum/policy_diagnostics/standing_success_10s_fraction
Curriculum/policy_diagnostics/standing_success_20s_fraction
Curriculum/policy_diagnostics/zero_command_xy_drift_mean_mps
Curriculum/policy_diagnostics/standing_foot_slip_mean_mps
Curriculum/policy_diagnostics/linear_velocity_tracking_rmse_mps
Curriculum/policy_diagnostics/yaw_rate_tracking_rmse_rad_s
```

## 2. Stand-v0 training

Suggested first serious run:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 4096 \
  --seed 42 \
  --run_name stand_seed42 \
  --headless
```

The runner default is 10,000 iterations with a checkpoint every 250 iterations.

Stand-v0 distribution:

```text
75% exact standing environments
x command       +/-0.15 m/s
y command       +/-0.08 m/s
yaw command     +/-0.25 rad/s
```

Stand-v0 deliberately uses:

- lower observation noise;
- narrow reset perturbations;
- no periodic push event;
- narrow friction randomization;
- +/-5% base mass and actuator-gain randomization;
- explicit standing rewards for low drift, low yaw rate, default pose, base height, both-feet contact, and low foot slip.

Do not select the best checkpoint only from episodic reward. Prefer a checkpoint with:

- low raw-action excess fraction;
- low bounded saturation fraction;
- low physical target-limit fraction;
- high 10 s and 20 s standing success;
- low zero-command drift;
- low standing foot slip;
- visually calm response to small velocity commands.

## 3. Analyze a Stand-v0 checkpoint

```bash
python scripts/rsl_rl/analyze_policy_v1_2.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --load_run <stand_run_directory_name> \
  --checkpoint model_5000.pt \
  --num_envs 256 \
  --steps 1500 \
  --headless
```

Outputs are written under:

```text
logs/rsl_rl/lilgreen_v1_2/<run>/analysis/<timestamp>/
  policy_analysis.json
  joint_action_stats.csv
```

The CSV includes per-joint:

- raw mean/std/min/max;
- raw p95/p99 and absolute p95/p99;
- fraction with `abs(raw) > 1`;
- bounded saturation fraction;
- lower/upper physical target saturation;
- actual position range;
- qdot p95/p99;
- velocity-limit fraction;
- torque p95/p99;
- torque-limit fraction.

## 4. Start Hardware-v0 from the selected Stand-v0 checkpoint

Both v1.2 runner configs use the same experiment root:

```text
logs/rsl_rl/lilgreen_v1_2/
```

Resume with:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-v0 \
  --resume \
  --load_run <stand_run_directory_name> \
  --checkpoint model_5000.pt \
  --num_envs 4096 \
  --seed 42 \
  --run_name hardware_from_stand_seed42 \
  --headless
```

Hardware-v0 starts its environment curriculum from Stage 0 even though the policy weights come from Stand-v0.

## Hardware-v0 curriculum

With the default 64 policy steps per PPO rollout, the stage boundaries are approximately 500, 1500, and 3500 Hardware-v0 PPO iterations.

| Stage | PPO range | Standing | X m/s | Y m/s | Yaw rad/s | Reset vel | Joint reset | Mass/gain scale | Push XY | 20 ms delay envs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 stand-first | 0–499 | 75% | +/-0.15 | +/-0.08 | +/-0.25 | +/-0.10 | +/-0.03 rad | 0.95–1.05 | 0 | 0% |
| 1 small motion | 500–1499 | 50% | +/-0.40 | +/-0.20 | +/-0.60 | +/-0.20 | +/-0.04 rad | 0.92–1.08 | 0 | 5% |
| 2 locomotion | 1500–3499 | 30% | +/-0.70 | +/-0.35 | +/-1.00 | +/-0.35 | +/-0.06 rad | 0.90–1.10 | +/-0.25 m/s | 15% |
| 3 deployment | 3500+ | 25% | +/-1.00 | +/-0.50 | +/-1.50 | +/-0.50 | +/-0.08 rad | 0.88–1.12 | +/-0.50 m/s | 25% |

Hardware-v0 keeps a fixed material-bucket envelope while the curriculum stages the command envelope, reset severity, base-mass scale, actuator-gain scale, pushes, and one-policy-step latency.

## 5. PPO defaults for the new tasks

Stand-v0:

```text
steps/env          64
max iterations     10000
save interval      250
init action std    0.6
learning rate      5e-4
entropy            0.006
normalization      disabled
```

Hardware-v0:

```text
steps/env          64
max iterations     50000
save interval      500
init action std    0.5
learning rate      5e-4
entropy            0.005
normalization      disabled
```

Both use the same `[256, 128, 128]` actor and critic hidden layers.

## 6. Hardware model used in v1.2

For the 12 V / 30 kg.cm ST3215 variant, v1.2 starts with:

```text
peak effort limit       2.94 N.m
no-load speed limit     4.72 rad/s
```

These are first-pass limits, not a complete servo dynamic model. Track 2 Orange Pi step-response work should eventually update:

- effective position-loop stiffness;
- effective damping;
- rise time;
- velocity saturation under load;
- command latency;
- deadband;
- overshoot and settling;
- sustained-current/thermal behavior.

## 7. Export

Export normally:

```bash
python scripts/rsl_rl/export_policy.py \
  --task Velocity-Lilgreen-Hardware-v0 \
  --load_run <hardware_run_directory_name> \
  --checkpoint model_10000.pt \
  --num_envs 1 \
  --headless
```

For v1.2, `policy.yaml` includes:

```text
action_contract_version: 2
action_transform: bounded_default_centered_asymmetric
action_limit_lower: -1.0
action_limit_upper: 1.0
action_target_lower_rad: [...]
action_target_upper_rad: [...]
action_default_rad: [...]
previous_action_observation: bounded_normalized_action
```

The exporter does not automatically copy a contract-v2 policy into the ROS 2 deployment directory unless explicitly overridden. This is intentional: the current Orange Pi policy node must first implement the same bounded asymmetric transform.

## 8. Release gate before physical policy control

Do not release the learned policy to physical writes based only on a good video.

Require at minimum:

1. raw action excess is low and stable;
2. bounded-action saturation is not persistent;
3. target-limit saturation is not persistent;
4. 10 s and 20 s standing success are strong;
5. zero-command drift and foot slip are low;
6. command tracking is good across the final envelope;
7. simulated qdot and torque limit fractions are reasonable;
8. the ONNX contract-v2 transform is reproduced exactly in ROS 2;
9. Track 2 outer-PD behavior is stable on controlled manual references;
10. first policy tests remain supported/guarded with a hardware power cutoff available.
