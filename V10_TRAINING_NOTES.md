# Littlegreen-Humanoid-Lite v2.0.0 — Hardware ST3215 Loaded v10

## Purpose

`Velocity-Lilgreen-Hardware-ST3215-Loaded-v10` is a condensed 5,000-iteration gait-acquisition task intended to answer one question before another long run:

> Can the mature v5s3 Stand policy learn alternating load transfer, real bilateral foot clearance, and clearance-gated placement without falling back to planted double support or one-leg bracing?

The task starts policy-only from:

```text
logs/rsl_rl/lilgreen_v1_4_5_st3215_stabilized_forward_s3/
  2026-07-13_12-10-53_stand_st3215_loaded_v145s3_forward_com_height460_seed42/
    model_3500.pt
```

The critic and optimizer start fresh. The 45-D Stand actor is expanded to 47-D with the two phase-input columns initialized to zero.

## Preserved contracts

v10 does not change:

- action contract v4;
- canonical 12-joint order;
- v5s3 `q_default`;
- vector residual action scales;
- physical joint limits;
- ST3215 Stage-A actuator model;
- loaded Stage-B actuator envelope;
- 12-D action size;
- 47-D observation size and ordering.

The meaning of observations 45–46 is intentionally revised. The phase is now command-synchronized rather than derived only from episode time.

## Main v10 changes

### Command-synchronized phase

- Standing freezes phase at a double-support boundary.
- A standing-to-moving transition resets phase.
- The first swing side is balanced across environments and alternates on later movement onsets.
- Phase advances only while a movement command is active.

### Short transfer windows

The old v9 transition width effectively made double support attractive for a large portion of the cycle. v10 uses one short transfer interval at the start of each half-cycle.

| Stage | Period | Transfer per boundary | Total scheduled double support |
|---:|---:|---:|---:|
| A | 0.90 s | 8% of cycle / 72 ms | 16% |
| B | 0.86 s | 7% / 60 ms | 14% |
| C | 0.82 s | 6% / 49 ms | 12% |
| D | 0.78 s | 5% / 39 ms | 10% |

### Explicit load and COM transfer

During transfer and early swing, v10 rewards:

- vertical support-force ratio moving toward the expected stance foot;
- COM moving forward and laterally toward the expected stance leg;
- alternating the target side every half-cycle.

Equal left/right loading is zero reward. Loading the wrong side is negative.

### Zero-baseline swing trajectory

The swing-height reference follows:

```text
h_ref = H * sin(pi * swing_progress)
```

The reward is baseline-corrected so zero foot lift produces exactly zero reward, unlike the v9 Gaussian target that still paid a positive baseline near the floor.

### Clearance-gated placement

Forward placement is disabled until the selected swing foot achieves real clearance. Zero step displacement is also normalized to zero reward.

### Taller, more forward moving posture

- Moving COM-height target: `0.455 m`
- Forward COM target progresses from `0.080 m` to `0.085 m` relative to the foot midpoint.
- Standing remains at the successful v5s3 `0.460 m` / `0.070 m` target.

### No no-progress termination during the decision run

The reset-safe no-support and orientation terminations remain. The no-progress termination is intentionally absent so the early policy can spend time learning transfer and lift before translation is mandatory.

## Condensed 5,000-iteration curriculum

With 64 policy steps per PPO iteration:

| Iterations | Focus | Stand fraction | `vx` | `vy` | yaw | Clearance target | Placement scale |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0–1,500 | Transfer + first lift | 0.28 | 0.25–0.34 | 0 | 0 | 18 mm | 0.15 |
| 1,500–3,000 | Lift + place | 0.26 | 0.25–0.40 | ±0.02 | ±0.04 | 24 mm | 0.70 |
| 3,000–4,200 | Translate | 0.23 | 0.25–0.48 | ±0.05 | ±0.12 | 28 mm | 1.00 |
| 4,200–5,000 | Broaden | 0.20 | 0.25–0.55 | ±0.08 | ±0.22 | 30 mm | 1.00 |

Velocity-tracking pressure increases gradually across the same stages.

## Training commands

### Short smoke test

```bash
NUM_ENVS=64 MAX_ITERATIONS=20 ./scripts/rsl_rl/train_v10_from_stand3500.sh
```

Confirm:

- the v10 task registers;
- the 45→47 policy-only warm start reports copied first 45 columns and zero new columns;
- no immediate reset loop appears;
- reward and curriculum logging begins normally.

### Full go/no-go run

```bash
./scripts/rsl_rl/train_v10_from_stand3500.sh
```

Defaults:

```text
4096 environments
5000 iterations
seed 42
save every 250 iterations
```

## Recommended analysis checkpoints

Analyze at approximately:

```text
250, 500, 1000, 1500, 2250, 3000, 3600, 4200, 4750, 4999
```

Example forced-forward analysis:

```bash
python scripts/rsl_rl/analyze_policy_v10.py \
  --task Velocity-Lilgreen-Hardware-ST3215-Loaded-v10 \
  --load_run <V10_RUN_DIRECTORY_NAME> \
  --checkpoint model_1500.pt \
  --num_envs 256 \
  --steps 2000 \
  --eval_force_command 0.30 0.00 0.00 \
  --headless
```

## Go/no-go guide

By model 1500, continue only if there is evidence of bilateral acquisition rather than the old right-leg brace:

- both left and right lift counts are nontrivial;
- lift imbalance is trending below about 85%;
- moving swing-clearance p95 reaches roughly 8–10 mm or more;
- moving single-support fraction is clearly above noise;
- moving no-support remains low;
- non-timeout terminations remain rare.

By model 3000, a promising run should show:

- repeated lift on both sides;
- clearance p95 moving toward 15–20 mm;
- improving alternation fraction;
- less persistent right ankle/knee torque-limit occupancy than v9;
- measurable command-aligned translation under `vx=0.30`.

A run should be stopped rather than extended if it still shows either of these v9 local optima:

```text
nearly permanent double support with negligible clearance
or
one permanent stance leg with >85–90% lift imbalance
```

## Deployment warning

Although the shape remains 47-D, v10 changes the phase-generator semantics. A v10 policy must not be deployed with the older episode-time phase oscillator. The exporter now writes:

```yaml
deployment_requires_command_synchronized_phase_v10: true
```

Track 2 must reproduce the command-onset reset, standing freeze, alternating first swing, stage-selected period, and phase progression before a v10 policy is eligible for live deployment.
