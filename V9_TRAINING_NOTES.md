# Littlegreen Hardware ST3215 Loaded v9

## Purpose

`Velocity-Lilgreen-Hardware-ST3215-Loaded-v9` is a clean gait-acquisition branch from the successful v5s3 Stand checkpoint at iteration 3500. It does not change the robot/deployment contract.

Preserved exactly:

- canonical 12-joint order;
- v5s3 `q_default`;
- action contract v4 and vector residual scales;
- physical joint limits;
- ST3215 Stage-A constants;
- loaded Stage-B envelope;
- 47-D observation order and phase semantics.

## Key corrections

- Stateful no-progress and no-support buffers clear at every episode reset.
- No-support monitoring normally arms only after valid foot support has been observed.
- The 45-D Stand actor expands to 47-D with the two new input columns initialized to zero.
- v9 uses an explicit 22-term reward configuration rather than inheriting the v5→v8 reward stack.
- External pushes remain disabled for the complete acquisition run.

## Command stages

With 64 policy steps per PPO iteration:

| Approx. iterations | Standing | Forward vx | Lateral vy | Yaw rate |
|---:|---:|---:|---:|---:|
| 0–2,000 | 0.25 | 0.25–0.36 | ±0.04 | ±0.08 |
| 2,000–5,000 | 0.23 | 0.25–0.42 | ±0.05 | ±0.12 |
| 5,000–8,000 | 0.21 | 0.25–0.50 | ±0.065 | ±0.17 |
| 8,000+ | 0.20 | 0.25–0.55 | ±0.08 | ±0.22 |

Only the command, reset velocity, joint reset offset, and base-mass envelope broaden. The task does not add pushes during this run.

## Start the 10k run

```bash
./scripts/rsl_rl/train_v9_from_stand3500.sh
```

Equivalent explicit command:

```bash
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v9 \
  --num_envs 4096 \
  --max_iterations 10000 \
  --seed 42 \
  --run_name hardware_st3215_loaded_v9_from_stand3500_seed42 \
  --resume \
  --policy_only_warm_start \
  --resume_log_root logs/rsl_rl/lilgreen_v1_4_5_st3215_stabilized_forward_s3 \
  --load_run 2026-07-13_12-10-53_stand_st3215_loaded_v145s3_forward_com_height460_seed42 \
  --checkpoint model_3500.pt \
  --headless
```

## Recommended review checkpoints

Run forced-command analysis around iterations 250, 500, 750, 1,000, 2,000, 3,500, 5,000, 8,000, and 10,000. The first acceptance gate is not reward magnitude; it is:

- no instant reset loop;
- both feet participate;
- real swing clearance rather than unweighting;
- commanded-direction step placement;
- torque occupancy does not climb into the old bracing regime.

Example:

```bash
python scripts/rsl_rl/analyze_policy_v9.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v9 \
  --load_run <v9-run-folder> \
  --checkpoint model_1000.pt \
  --num_envs 256 \
  --steps 2000 \
  --eval_force_command 0.30 0.0 0.0 \
  --headless
```

## Runtime validation still required

Static validation cannot prove Isaac Lab registration, PhysX contact behavior, PPO stability, or actual checkpoint compatibility. Run a short 32–64 environment smoke test before committing the full GPU session.
