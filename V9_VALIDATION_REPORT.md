# Littlegreen-Humanoid-Lite v2.0.0 — Hardware Loaded v9 Validation Report

Date: 2026-07-14

## Scope

This report covers the additive task:

`Velocity-Lilgreen-Hardware-ST3215-Loaded-v9`

The implementation is intended to warm-start actor-side weights from:

`logs/rsl_rl/lilgreen_v1_4_5_st3215_stabilized_forward_s3/2026-07-13_12-10-53_stand_st3215_loaded_v145s3_forward_com_height460_seed42/model_3500.pt`

## Completed offline validation

- Python syntax/AST parsing passed for the active Track 1 package and RSL-RL scripts.
- Existing static test suite plus v9 regression checks: **53 passed**.
- Stateful termination behavior was exercised with offline tensor-backed environment stubs:
  - stale no-support timer is cleared at reset;
  - no-support arms after valid support and terminates only after sustained flight;
  - stale no-progress timers are cleared at reset.
- Original 24 task IDs remain registered and v9 is the only new task ID.
- v9 reward configuration is standalone, does not inherit the v5→v8 stack, and contains exactly **22 reward terms**.
- v9 staged command envelope was statically verified:
  - forward `vx`: 0.25–0.55 m/s maximum final range;
  - lateral `vy`: up to ±0.08 m/s;
  - yaw rate: up to ±0.22 rad/s;
  - external push envelope: 0.0 throughout.
- The included Stand-v5s3 `model_3500.pt` is byte-identical to the v2.0.0 source archive.
- The Stand-v5s3 checkpoint actor first layer is `(256, 45)`.
- The revised warm-start path copies all 45 source columns exactly and zero-initializes the two new phase columns.
- The following contract/model files are byte-identical to v2.0.0:
  - `hardware_contract.py`;
  - `st3215_actuator_model.py`;
  - `st3215_loaded_actuator_model.py`;
  - `env_cfg_hardware_st3215_loaded_v145.py`.
- Training helper shell syntax passed.
- Final archive ZIP integrity passed.

## Deliberate source changes

- Added v9 task, runner, analyzer, training helper, training notes, and static tests.
- Added a gentle four-stage command/reset/mass curriculum.
- Corrected reset handling in shared stateful no-progress and no-support terminations.
- Added support-aware arming to no-support termination.
- Changed policy-only expanded actor initialization from random new columns to zero new columns.

## Not run in this environment

The following require the Isaac Lab training host and are **not claimed as validated**:

- editable package installation inside the Isaac Lab environment;
- Gym task instantiation;
- confirmation of live 47-D actor and critic observation tensors;
- PhysX contact-sensor behavior across resets;
- PPO warm-start execution against `model_3500.pt`;
- 32–64 environment smoke training;
- 4,096 environment / 10,000 iteration training;
- forced-command rollout analysis;
- policy export or ONNX deployment;
- real-robot behavior.

## Required first runtime gate

Before the full training session, run a short smoke test using 32–64 environments and approximately 20–50 iterations. Confirm:

1. the task registers and creates successfully;
2. the warm-start log reports two zero-initialized new actor input columns;
3. observation shape is 47 and action shape is 12;
4. resets do not enter a repeated no-support loop;
5. reward terms initialize without signature errors;
6. checkpoint saving works.
