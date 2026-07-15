# Berkeley Humanoid Lite v1.2.3 Baseline Guide

## 1. Validate the nominal pose tool

The canonical task is now Stand-v0:

```bash
python scripts/rsl_rl/measure_nominal_pose.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 64 \
  --settle_seconds 5 \
  --sample_seconds 2 \
  --output logs/nominal_pose_v123.json \
  --headless
```

The deterministic reset-state q_default root-body COM height should be approximately `0.4899106 m` for the current asset.

## 2. Short smoke training

Run a short fresh training job before committing to the 5000-iteration baseline:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 4096 \
  --max_iterations 100 \
  --seed 42 \
  --run_name stand_v123_residual020_smoke_seed42 \
  --headless
```

Confirm that reward terms, diagnostics, checkpoint saving, and TensorBoard logging are healthy.

## 3. Establish the new 5000-iteration baseline

Use a fresh run, not a v1.2.2 resume, because the physical meaning of the policy action changed:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name stand_v123_residual020_seed42 \
  --headless
```

## 4. Offline analysis

```bash
python scripts/rsl_rl/analyze_policy_v1_2_3.py \
  --task Velocity-Lilgreen-Stand-v0 \
  --load_run <run-directory-name> \
  --checkpoint model_5000.pt \
  --num_envs 256 \
  --steps 1500 \
  --headless
```

Primary comparison targets:

- stable-standing 10 s / 20 s success;
- posture-conforming 10 s / 20 s success;
- fall and timeout counts;
- standing COM height distribution and error to the 0.4899106 m nominal target;
- global and standing-only torque-limit occupancy;
- global and standing-only raw action excess;
- standing-only target residual magnitude and physical target-limit occupancy;
- zero-command XY drift and standing foot slip.

## 5. Export

Export metadata must report action contract v3. Do not copy a policy into the ROS 2 deployment path until the policy node implements the exported residual transform.
