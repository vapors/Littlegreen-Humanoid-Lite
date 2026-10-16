# LittleGreen Humanoid Lite — Track 1 v11.1.0 Bilateral Gait Acquisition

**Purpose:** an experimental **reward-model revision** built on the host-qualified execution path of v11.0.0 HF1. The new release does **not** claim that the robot walks; we are specifically addressing the v11.0 unilateral left-foot lift/right-stance exploit. It is NOT a behavior-preserving refactor; new training must start fresh. Track 2 ROS 2 hardware is untouched and autonomous hardware walking is not authorized.

## What is and isn't changed

The mass-corrected USD (provisional 3.852 kg), 12 hardware joint order/limits/residual scales, actuator Stage-A and loaded response profiles, free root, k=18.0/d=1.5 implicit PD, 2.94 Nm effort cap, physics `dt=0.005`, decimation 4, actor 47-D/critic 50-D, previous bounded normalized action observation, neutral-static phase, and PPO numerical recipe are unchanged from v11.0 HF1. The PPO experiment identity changes to prevent accidental cross-version resume. Fresh start: **do not** load v10.x or v11.0 checkpoints/normalizers into v11.1. Same-run v11.1 resume remains guarded by the run source/contract hashes.

**Five deliberate reward-config changes** (all other v11.0 reward terms and numerical parameters unchanged):

| Reward | v11.0 HF1 | v11.1.0 | Reason |
|---|---:|---:|---|
| `track_lin_vel_xy_acquisition` | +0.80 | **+0.50** | Less income for non-step forward rocking |
| `coherent_swing_rise_velocity` | +1.00 | **+0.65** | Repeated partial foot lifts less dominant |
| `whole_foot_air_time_exploration` | +0.50 | **+0.35** | Reduce same-side hover/air-time farming |
| `balanced_swing_discovery` | +0.85 | **+1.20** | New side-aware qualified-bout reward, with distinct attempts rather than per-frame repetition |
| `receiving_foot_load_acceptance` | absent | **+0.85** | New progress-after-replant reward toward the next support handoff |

**Bilateral exploration logic:** same physical-foot kinematics and viability gates as v10.3.1; a new distinct attempt requires quality >0.065, whole-foot clearance >4 mm, 4 below-threshold rearm frames and at least 0.25 s between credited onsets. Consecutive attempts on the *same* side receive discovery multipliers 1.0, 0.68, 0.462, then down to a floor of 0.20; a qualified opposite-side attempt after two or more repeated bouts may receive up to a +45% bounded discovery multiplier. **This only attenuates the new bootstrap exploration term; it does not penalize actual physical recovery steps, nor alter the existing completed-step and WORLD-rate terms.** Recovery (tilt plus angular motion or significant tilt) and clearly commanded lateral/strafe steps bypass repetition attenuation; they do not create free reward income. No imposed gait phase or scripted target joint motion.

**Receiving-foot acceptance:** when a swing foot has been completely airborne (main + toe), moved meaningfully along the commanded WORLD direction, and the other foot was fully planted, a short (<=0.90 s) reward window is armed. Only later replant with main + toe contact and viable anchored support earns *incremental increases* in the receiving foot's normalized load share and COM closeness. Each increment is measured above its prior maximum. Static holding or repeatedly unloading/reloading to the same peak cannot farm this term. Confirmed support handoffs, bilateral passage, and signed WORLD-rate rewards remain unchanged.

## Installation — unpack as a new source checkout

Use the same already-working Isaac Lab / Isaac Sim Python 3.10 environment that passed v11.0 HF1; do not overwrite v11.0 without backup. The root pyproject retains the earlier Isaac stack pins. The archive is standalone, not an overlay.

```bash
cd ~/Littlegreen-Humanoid-Lite   # your newly extracted v11.1 tree
conda activate littlegreen-humanoid-lite
python -m pip install -e source/littlegreen_humanoid_lite
python -m pip install -e source/littlegreen_humanoid_lite_assets
python -m pip install -e source/littlegreen_humanoid_lite_lowlevel
python -m pytest tests -q
python scripts/rsl_rl/check_contract.py
```

## Isaac Lab preflight, smoke, acquisition

```bash
cd ~/Littlegreen-Humanoid-Lite
python scripts/rsl_rl/preflight.py \
  --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111 \
  --output logs/v111_preflight.json --headless

python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111 \
  --num_envs 64 --max_iterations 1 --seed 113 \
  --run_name v111_smoke --headless

# First baseline qualification, fresh from random policy weights
python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111 \
  --num_envs 2048 --max_iterations 250 --seed 113 \
  --run_name v111_bilateral_seed113 --headless
```

The PPO config has `save_interval=250`, maximum 4,000 iterations by default, exploration noise std=0.80, 48 steps/environment, lr=5e-4 and 256/128/128 MLP. Inspect checkpoint filenames in the actual run directory (the previous 250-iteration run produced `model_249.pt`). If 250-step bilateral diagnostics improve, use a **verified same-run resume** to 500–1,000 total iterations or start a separately named longer fresh run. RSL-RL's `--max_iterations` resume meaning may depend on the installed version; check your runner logs before scheduling continuation.

## Analyzer, interactive playback and exporter

```bash
python scripts/rsl_rl/analyze_policy.py \
  --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111 \
  --num_envs 128 --steps 1200 \
  --load_run RUN_DIRECTORY_NAME --checkpoint model_249.pt \
  --eval_force_command 0.20 0.0 0.0 --headless

python scripts/rsl_rl/play_joystick.py \
  --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111 \
  --num_envs 1 --load_run RUN_DIRECTORY_NAME --checkpoint model_249.pt

python scripts/rsl_rl/export_policy_littlegreen.py \
  --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111 \
  --num_envs 1 --load_run RUN_DIRECTORY_NAME --checkpoint model_249.pt --headless
```

Run `analyze_policy.py` with `--eval_force_command 0.30 0.0 0.0` for the second trained-speed probe. `0.40` and `0.60` are **out of the trained forward range** and should be labeled stress probes, never interpreted as trained velocity tracking. Keep the analyzer's standing probe enabled and force-command results separate.

**New analyzer block `bilateral_acquisition_v111`:** qualified attempt counts per side (event onsets, not airborne frame count), mean per-side swing quality, mean credit multiplier, bilateral recent quality, repeated-attempt streak maximum, recovery/strafe exception occupancy, receiving-foot load/COM increments and completion events by side, actual new-term per-step weighted contributions before time-step integration. Historical counters for main/toe release, load share, coherent rise, landing progress, confirmed steps, support baton, WORLD-rate events and joint torque are retained. Full measured contribution accounting for **every** existing reward remains a future host test; do not infer those values from configured weights alone.

**Zero-command pose:** the analyzer continues to derive late viable full-plant median pose and previous bounded action, but marks any pose with fewer than 100 selected samples or <50% full-plant fraction as `DIAGNOSTIC_ONLY` (conservative screening threshold, not a guarantee). A stronger sample can only become `CANDIDATE_REQUIRES_TRACK2_REVIEW`, never automatically deployed. Keep `q_default` unchanged.

## Zero-command deployment contract correction

Training command distribution remains **12% full zero-command standing** and otherwise vx in `[0.12,0.30]` m/s, vy=0, yaw=0. Earlier v11 export metadata incorrectly suggested clamping zero to +0.12. v11.1 exports command-contract version 2:

- `[0,0,0]` must remain **exactly zero** and preserve a zero policy observation command (and the neutral-static phase tail `[0,1]`).
- A requested forward command in `(0,0.08)` m/s maps to zero/hold; `[0.08,0.12)` clips up to +0.12; `[0.12,0.30]` is supported; above +0.30 clips to +0.30.
- Negative forward, nonzero lateral or nonzero yaw are unsupported in this acquisition training profile and must be **rejected or switched to safe zero** (never converted to positive walking).
- `--deploy_ros2` remains forbidden until Track 2 explicitly validates and implements this new command behavior and the rest of the policy bundle. This metadata change does NOT modify Track 2 itself.

These semantics are encoded in `policy.yaml` and `deployment_contract.yaml`, not hard-coded by a ROS 2 update. If the runtime cannot honor them, decline deployment.

## What success looks like

| Gate | Requested evidence |
|---|---|
| 250 iterations | Right-foot attempts begin, swing reward not monopolized by the left, no fall/slip loophole |
| 500 iterations | Receiving-foot loading and COM-closeness growth on both sides; first truly alternating handoffs |
| 1,000–2,000 | Repeatable both-leg whole-foot lift/replant, positive stance-relative forward landing, bilateral WORLD-frame progress |
| All checkpoints | 47-D obs, 12-D action contract parity; corrected mass/PD/torque; keep right-ankle saturation visible |

Do **not** claim walking acquisition from total reward, isolated left-foot lift, or an unqualified pose. If 250 iterations show no right-foot exploration, stop to examine the new reward gates before a long run.

## Release provenance and limitations

- `manifests/CONTRACT_V11.json`: original v11.0 HF1 frozen baseline.
- `manifests/CONTRACT_V111.json`: current v11.1 source-resolved configuration AST with 45 reward terms.
- `manifests/REWARD_V111_DIFF.json`: allowlisted five-term changes and weights.
- `manifests/REWARD_PARITY.json`: **historical** v11.0 parity, not a claim that v11.1 rewards are equal.
- `manifests/OFFLINE_VALIDATION_V111.json`: actual offline tests and reference asset checks.
- `CHECKSUMS.sha256`: packaged file hashes.

Validation here is offline only: the new reward state has been exercised with deterministic synthetic tensors, not live PhysX contacts. Gym instantiation, Isaac Lab training and reward magnitude observations, ONNX export, parity with hardware limitations, and physical stability must still be checked on the Isaac Lab workstation. Other movement skills remain planned only, with no fictional task registrations. The source supports config-based ManagerBasedRLEnv skill composition once those individual skills are implemented and validated.

## Full active reward inventory (configuration weights, not measured reward income)

| Term | Weight | Function |
|---|---:|---|
| `action_rate_l2` | -0.0005 | `mdp.action_rate_l2` |
| `ang_vel_xy_l2` | -0.05 | `mdp.ang_vel_xy_l2` |
| `balanced_swing_discovery` | 1.2 | `mdp.bilateral_swing_discovery_v111` |
| `base_com_height_deficit` | -0.4 | `mdp.base_com_height_deficit_v63` |
| `bilateral_passage_dense_step_support` | 1.1 | `mdp.dense_step_support_bilateral_passage_v98` |
| `coherent_swing_rise_velocity` | 0.65 | `mdp.coherent_swing_rise_velocity_v67` |
| `commanded_world_rate_dense_step_support` | 2 | `mdp.dense_step_support_bilateral_commanded_world_rate_v101` |
| `completed_gait_step` | 1.85 | `mdp.completed_gait_step_retention_v95` |
| `dof_acc_l2` | -1e-07 | `mdp.joint_acc_l2` |
| `dof_pos_limits` | -0.05 | `mdp.joint_pos_limits` |
| `feet_slide` | -0.3 | `mdp.whole_foot_slide_v39` |
| `finite_directional_swing_progress` | 1.2 | `mdp.command_conditioned_directional_swing_progress_v95` |
| `finite_swing_initiation` | 1.9 | `mdp.finite_whole_foot_swing_initiation_v66` |
| `flat_orientation_l2` | -0.18 | `mdp.flat_orientation_l2` |
| `lin_vel_z_l2` | -0.05 | `mdp.lin_vel_z_l2` |
| `loaded_replant` | 0.45 | `mdp.loaded_replant_retention_v95` |
| `moving_long_double_support` | -0.2 | `mdp.moving_long_double_support_penalty` |
| `opposite_support_release` | 1.8 | `mdp.opposite_support_release_event_v66` |
| `post_touchdown_support_retreat` | -0.3 | `mdp.post_touchdown_support_retreat_penalty_v69` |
| `prolonged_same_stance` | -0.25 | `mdp.moving_prolonged_whole_foot_stance_v30` |
| `prolonged_single_support` | -0.45 | `mdp.moving_prolonged_whole_foot_single_support_v30` |
| `raw_action_excess_l2` | -0.015 | `mdp.raw_action_excess_l2` |
| `ready_no_liftoff` | -0.45 | `mdp.ready_no_liftoff_penalty_v67` |
| `receiving_foot_load_acceptance` | 0.85 | `mdp.receiving_foot_load_acceptance_v111` |
| `stance_relative_com_drive` | 0.18 | `mdp.body_yaw_lateral_support_transfer_v99` |
| `standing_actionable_joint_velocity` | -0.012 | `mdp.standing_actionable_joint_velocity_l2_v65` |
| `standing_base_xy_speed` | -0.2 | `mdp.standing_base_xy_speed_l2` |
| `standing_bilateral_full_plant` | 0.45 | `mdp.standing_bilateral_full_plant_v65` |
| `standing_both_whole_feet` | 0.3 | `mdp.standing_both_whole_feet_contact_v250` |
| `standing_lateral_com_balance` | -0.1 | `mdp.standing_com_lateral_over_feet_l2` |
| `standing_whole_foot_load_imbalance` | -0.35 | `mdp.standing_whole_foot_load_imbalance_v65` |
| `standing_yaw_rate` | -0.1 | `mdp.standing_yaw_rate_l2` |
| `support_handoff` | 1.85 | `mdp.support_handoff_event_v66` |
| `support_readiness_progress` | 0.35 | `mdp.support_readiness_progress_v66` |
| `support_takeover_progress` | 1.85 | `mdp.support_takeover_progress_v66` |
| `sustained_max_joint_torque_barrier` | -0.08 | `mdp.sustained_max_torque_barrier_l2_v60` |
| `swing_backtrack` | -0.3 | `mdp.swing_backtrack_penalty_v95` |
| `swing_behind_liftoff` | -0.18 | `mdp.swing_behind_liftoff_penalty_v95` |
| `swing_foot_command_velocity` | 1.85 | `mdp.swing_foot_command_velocity_v95` |
| `takeover_failure` | -0.35 | `mdp.takeover_failure_event_v66` |
| `termination_penalty` | -2.5 | `mdp.is_terminated` |
| `track_ang_vel_z_acquisition` | 0.55 | `mdp.track_ang_vel_z_acquisition_v63` |
| `track_lin_vel_xy_acquisition` | 0.5 | `mdp.gait_conditioned_track_lin_vel_xy_v67` |
| `undesired_contacts` | -1 | `mdp.undesired_contacts` |
| `whole_foot_air_time_exploration` | 0.35 | `mdp.whole_foot_air_time_viability_gated_v63` |

## Reference background

The early v6.7–v6.8 experiments established real whole-foot alternation and COM transfer, demonstrating the mechanical possibility of bilateral steps. Later v9.7–v10.2 WORLD-rate/bilateral events were valuable but often gated behind completed motion; v10.3.1 exposed how unilateral partial swing could dominate early reward income. Those prior experiments inform this hypothesis but do not prove that v11.1 will train successfully. Do not use old v10.x checkpoints to resume this changed recipe. Historical source and analyses remain in the separately backed-up archive, not in this clean release.
