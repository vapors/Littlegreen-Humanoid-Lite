# Berkeley Humanoid Lite v1.3.0 ST3215 Task Family

## Frozen v1.2.3 baseline

The existing task registrations remain available and unchanged:

- `Velocity-Lilgreen-Stand-v0`
- `Velocity-Lilgreen-Hardware-v0`

## New actuator-aware tasks

- `Velocity-Lilgreen-Stand-ST3215-v0`
- `Velocity-Lilgreen-Hardware-ST3215-v0`

Both preserve the v1.2.3 45-D actor observation, 12-D action order, action contract v3, q_default reference, ±0.20 rad residual mapping, physical joint clipping, rewards, and measured nominal COM height. The ST3215 response layer is inserted after policy target mapping and before the PhysX joint position drive.

## Stage-A actuator model

The first v1.3.0 baseline models:

1. 0–20 ms bus-phase delay.
2. Configurable response-delay proxy based on write-to-first-encoder timing.
3. Per-joint tau-equivalent response randomization over measured p10–p90 ranges.
4. Per-joint amplitude-to-peak-velocity curves measured from 0.02–0.20 rad steps.
5. Per-joint large-signal static gain.
6. Small-signal residual-error and center-hysteresis proxies.
7. Approximately ±5% velocity-response scale randomization.

`response_delay_scale` is explicit because the write-to-first-encoder interval is treated as a command-to-observed-motion proxy, not a proven pure mechanical delay. Encoder read cadence and the current raw-position/raw-speed/telemetry workload may contribute to that interval. Track 2 can revise the delay scale or range after driver streamlining without changing the policy action contract or the rest of the ST3215 response model.

Special overshoot and long-settling-tail data is preserved as metadata but not injected in Stage A. The first baseline intentionally validates a monotonic measured-response model before adding family-specific transient branches.

## Not inferred from unloaded tests

The suspended tests do not identify physical stiffness in N·m/rad or physical damping in N·m·s/rad. Generic actuator stiffness/damping randomization is removed from the new v1.3.0 ST3215 tasks. The frozen v1.2.3 tasks retain their historical settings.

## Fresh Stand-ST3215 baseline

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Stand-ST3215-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name stand_st3215_v130_fresh_seed42 \
  --headless
```

Analyze checkpoints with:

```bash
python scripts/rsl_rl/analyze_policy_v1_3_0.py \
  --task Velocity-Lilgreen-Stand-ST3215-v0 \
  --load_run <run-name> \
  --checkpoint model_2500.pt \
  --num_envs 256 \
  --steps 1500 \
  --headless
```
