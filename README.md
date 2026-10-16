# LittleGreen Humanoid Lite — Track 1 v11.0.0 Clean Core

**A clean source consolidation, not a new gait experiment.** Track 1 is Isaac Lab + RSL-RL PPO training, simulation analysis, and export. The active task preserves the v10.3.1 bilateral-acquisition reward semantics and ST3215 model. Track 2 controls the physical Orange Pi + ST3215 hardware separately; this release neither modifies nor authorizes hardware commands.

**Validation boundary:** the package has source-level and offline verification. It has **not** been installed in a running Isaac Lab/PhysX host in this environment; no gait, installation with CUDA, real servo equivalence, or ONNX runtime compatibility is claimed without that test. Physical mass 3.852 kg is provisional.

## Quick start

Target Isaac Lab environment is the one already used by the project; the archive's root pyproject retains Isaac Lab/Isaac Sim version pins (`isaaclab[isaacsim,all]==2.1.0`, `isaacsim==4.5.0`) and Python **3.10** requirements. Match the simulator stack already verified on your host before changing any pins. This workspace requires NVIDIA Isaac Sim and an appropriate GPU for physics.

```bash
# Unpack the complete v11 ZIP and work in its root. Do NOT overlay onto v10.3.1.
cd ~/Littlegreen-Humanoid-Lite
conda activate littlegreen-humanoid-lite
python -m pip install -e source/littlegreen_humanoid_lite
python -m pip install -e source/littlegreen_humanoid_lite_assets
python -m pip install -e source/littlegreen_humanoid_lite_lowlevel

# Offline verification (works without Isaac Lab if pytest and Python deps are installed)
python -m pytest tests -q
python scripts/rsl_rl/check_contract.py

# Host-specific active configuration / physical-plant preflight, no policy load
python scripts/rsl_rl/preflight.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --headless --output logs/v11_preflight.json

# One-iteration fresh smoke before a long run
python scripts/rsl_rl/train_eval.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --num_envs 64 --max_iterations 1 --seed 113 --run_name v11_smoke --headless

# Short dynamics qualification, then full fresh run AFTER successful preflight
python scripts/rsl_rl/train_eval.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --num_envs 2048 --max_iterations 250 --seed 113 --run_name v11_250_qualification --headless
python scripts/rsl_rl/train_eval.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --num_envs 2048 --max_iterations 4000 --seed 113 --run_name v11_acquisition_fresh --headless
```

Run checkpoints appear at `logs/rsl_rl/littlegreen_v11_0_0_clean_core_bilateral_acquisition_fresh/<timestamp>_<run_name>/model_{iteration}.pt`. The canonical PPO configuration saves every 250 iterations, and train_eval preserves the final small-smoke-save behavior when video is requested. Logs are outside the release.

Resume **only within the exact verified v11 experiment** (no policy-only v10.x warm-start), using your actual run directory and checkpoint name:

```bash
python scripts/rsl_rl/train_eval.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --num_envs 2048 --resume --load_run RUN_DIRECTORY_NAME --checkpoint model_250.pt --max_iterations 250 --headless
```

The earlier run must contain `run_contract_v11.json`; the trainer compares current source-code/asset hashes, task identity, architecture, simulator timing and normalization settings. Changing recipe/code triggers refusal. `--max_iterations` is the number of additional learning iterations requested by the runner; inspect your RSL-RL installed version before a long continuation. New v11 training starts from randomized network weights; reusing an old v10.3.x checkpoint is prohibited on this canonical task.

## Analyze, play, export

```bash
# Match seed, env count, steps and force command across baseline comparisons
python scripts/rsl_rl/analyze_policy.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --num_envs 128 --steps 1200 --load_run 2026-10-16_12-25-41_v11_250_qualification --checkpoint model_249.pt --eval_force_command 0.30 0.0 0.0 --headless

# Windowed / visible joystick replay: file contains "vx vy yaw" floats
printf '0.15 0 0\n' > /tmp/joystick_cmd.txt
python scripts/rsl_rl/play_joystick.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --load_run 2026-10-16_12-25-41_v11_250_qualification --checkpoint model_249.pt --num_envs 1


printf '0.15 0 0\n' > /tmp/joystick_cmd.txt
python scripts/rsl_rl/play_joystick.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --load_run 2026-10-15_21-15-05_v11_bilateral_acquisition_seed113_10000 --checkpoint model_249.pt --num_envs 1


# Export ONNX, TorchScript and deployment contract. Does NOT auto-publish to physical ROS.
python scripts/rsl_rl/export_policy_littlegreen.py --task Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110 --load_run 2026-10-16_12-25-41_v11_250_qualification --checkpoint model_249.pt --num_envs 1 --headless
```

The analyzer carries v10.3.1 diagnostic field names for provenance. It reports bilateral foot events, commanded WORLD-frame progress, stance slip, action/torque statistics, mass readback, the zero-command full-plant handoff-pose sample and previous bounded-action parity. A qualified handoff pose is **not** automatically a safe deployment pose. The current analyzer's reward contribution instrumentation is limited (e.g. `balanced_swing_discovery`); **complete per-term measured reward-contribution accounting remains a host-validation/development gate**, not an accomplished capability. Analyzer does not assert compatibility with arbitrary historical checkpoints. Replay of a v10.3.1 checkpoint must remain in the backed-up v10.3.1 environment until explicit full contract comparison proves it compatible.

Export is restricted to the v11 task in this release. The existing schema includes 47-D observation, bounded previous-action, contract-v4 12-D residual, corrected mass USD, neutral-static phase, trained command envelopes and ONNX/SHA manifest metadata. **Do not use `--deploy_ros2` without a Track 2 contract audit and supervised hardware safety preflight.**

## Frozen physical & policy contract

| Property | v11 frozen baseline |
|---|---|
| Corrected simulation USD | `lilgreen_mass_v2926.usd`, provisionally 3.852 kg, free root |
| Simulation and policy timing | 200 Hz physics (0.005 s), decimation 4, 50 Hz policy (0.020 s) |
| Implicit simulated PD | Legs k=18.0, d=1.5; ankles k=18.0, d=1.5 |
| Simulated effort limit | 2.94 Nm per actionable actuator (not a claim about firmware PID) |
| Actor/critic | 47-D / 50-D |
| Actor input order | command3, angular velocity3, gravity3, joint_pos12, joint_vel12, previous bounded normalized action12, phase sin/cos2 |
| Previous action | indices [33,44] inclusive; bounded normalized, **not radians** |
| Active action | 12-D ST3215 measured-vector residual around protected q_default, mirrored sign and physical joint limits |
| Contact | both main foot and passive toe; whole-foot release, support baton, replant and handoff qualification |
| Phase | `neutral_static` (`sin=0,cos=1`) |
| Command training envelope | fixed forward acquisition vx [0.12,0.30] m/s, vy=0, yaw=0, 12% standing; no staged curriculum |
| PPO | 48 steps/env, network [256,128,128], noise 0.80, lr=5e-4 fixed, entropy 0.006, γ=0.99, GAE λ=0.95, 5 epochs, 4 mini-batches, 4,000 maximum iterations default |

The exact **joint order, q_default, physical limits and per-joint residual scales** are owned by `mdp/hardware_contract.py` and `actions_cfg.py`, not duplicated into competing editable documents. `manifests/CONTRACT_BASELINE.json` and `CONTRACT_V11.json` contain source-verified metadata and **the full expressions for all reward terms**. No value in the protected reward recipe was changed.

## Architecture and source authority

```text
source/littlegreen_humanoid_lite/
  littlegreen_humanoid_lite/tasks/locomotion/velocity/
    config/littlegreen_humanoid/
      __init__.py               ONE gym registration: v110
      tasks/locomotion.py       single canonical v10.3.1-derived env config
      rewards/locomotion.py     flattened current reward terms
      agents/rsl_rl_ppo_cfg.py  single PPO runner config
      base_cfg.py / observations_cfg.py / actions_cfg.py
      mdp.py                    MDP facade
      skills.py                 typed planned-capability inventory
    mdp/                        preserved contact/gait/ST3215 function sources
source/littlegreen_humanoid_lite_assets/data/usd/
  lilgreen_mass_v2926.usd       current mass asset
scripts/rsl_rl/
  train_eval.py / analyze_policy.py / play_joystick.py
  export_policy_littlegreen.py / preflight.py / check_contract.py
tests/                         dependency-free static/provenance tests
manifests/                     audit evidence, hashes and future capability metadata
```

Some older-named MDP modules (`commands_v101.py`, `bilateral_acquisition_v1031.py`) remain deliberately because they are **genuinely active functions**, not obsolete task classes. The base/observation/action objects still use some historical symbols as stable API boundaries. No versioned reward inheritance classes or historic task registrations remain on the active release path. Assets for older MuJoCo/URDF operations are retained separately from the canonical Isaac Lab USD.

## Reward inventory — frozen v10.3.1 effective terms (44)

Weights below are *configuration weights*, not measured reward income. Compare raw vs weighted per-step contributions on Isaac Lab before interpreting reward economics.

| Reward term | Weight | Function |
|---|---:|---|
| `action_rate_l2` | -0.0005 | `mdp.action_rate_l2` |
| `ang_vel_xy_l2` | -0.05 | `mdp.ang_vel_xy_l2` |
| `balanced_swing_discovery` | 0.85 | `mdp.balanced_swing_discovery_v1031` |
| `base_com_height_deficit` | -0.4 | `mdp.base_com_height_deficit_v63` |
| `bilateral_passage_dense_step_support` | 1.1 | `mdp.dense_step_support_bilateral_passage_v98` |
| `coherent_swing_rise_velocity` | 1 | `mdp.coherent_swing_rise_velocity_v67` |
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
| `track_lin_vel_xy_acquisition` | 0.8 | `mdp.gait_conditioned_track_lin_vel_xy_v67` |
| `undesired_contacts` | -1 | `mdp.undesired_contacts` |
| `whole_foot_air_time_exploration` | 0.5 | `mdp.whole_foot_air_time_viability_gated_v63` |

Acquisition scaffold explicitly retains the 3/10/22/40 mm main/toe release ladder, bilateral passage, signed WORLD frame progress, confirmed support takeover and fall-neutral rewards. Avoid credit for partial heel lift, slipping support, permanent unilateral loading or forward falls. Six rebalance coefficients in v10.3.1 remain frozen, and `balanced_swing_discovery_v1031` retains **0.85**. Skills such as stairs and recovery should compose shared observation/action hardware definitions, not inherit one earlier experiment's reward class. Existing ManagerBasedRLEnv managers and RSL-RL PPO are preserved.

## Future skills — supported architecture, not claimed training

See `skills.py` and `manifests/PLANNED_SKILLS.json`: stand, full directional walking, side stepping, turning, crouch walk, stairs, obstacles, jumping and recovery are **planned**. Implement one configuration and (if needed) a small transparent manager curriculum for each, then test and register it; there are **no fake placeholder Gym registrations** or scripted foot trajectories. No automatic Stand→Transfer→Walk warm-start chain is introduced in v11. PPO-based generalizable skill learning remains the intent.

## Historical lessons retained

- v6.7–6.8 provided credible contact-/clearance-gated bilateral liftoff and support mechanics. Those events remain active; a positive reward alone does not prove gait.
- v9.7–v10.2 linked confirmed support events, bilateral passage and signed commanded WORLD-rate so static gait/rocking does not earn true travel credit; the richer movement distribution later remained exploratory.
- v10.3 made the provisional mass-corrected USD plus lower simulated PD explicit. v10.3.1 added balanced pre-handoff acquisition and six economic changes to address uneven leg liftoff. **This release has no verified new walking result.**
- Prior hardware tests identified ST3215 trajectory/feedback envelopes, not independent physical torque/stiffness equivalents. The simulated k/d is a working model only.

## Acceptance gates & unresolved items

| Check | Status |
|---|---|
| Baseline v10.3.1 reward AST function/params/weights and PPO recipe, except new experiment name | Static parity verified; review `REWARD_PARITY.json` |
| Version, copied source modules, finite USD and archive inventory | Audited offline |
| Test Python source compile / fresh pip editable installation without Isaac dependencies | See `OFFLINE_VALIDATION.json` for exact results |
| Clean Isaac Lab runtime `gym.make`, joint shape, k/d/torque, mass and USD references | **Required on host** |
| 64-env one-iteration smoke and 2048-env 250-iteration qualification | **Required on host** |
| Measured reward per-term contributions and deterministic baseline-v11 trajectory parity | **Required, analyzer full contribution feature pending** |
| Resume/compatibility positive and negative cases in live RSL-RL | **Required on host** |
| ONNX export validation and Track 2 runtime equivalence | **Required on host; no hardware authorization** |

The verified and unverified details are in `manifests/OFFLINE_VALIDATION.json`. `REMOVED_FROM_V11.md` and `manifests/LEGACY_FILE_MANIFEST.json` enumerate excluded historical files in the source backup; the original user archive remains intact. The corrected USD contains binary USD content; **its complete internal composition and references must still be checked with Omniverse/USD tools on the Isaac host**. Other USDs are retained pending that check. A total source-file reduction does not by itself establish behavioral parity.
