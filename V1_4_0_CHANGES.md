# Berkeley Humanoid Lite v1.4.0 Changes

v1.4.0 starts from `Berkeley-Humanoid-Lite_v1_3_1_ST3215` and preserves the
updated physical joint limits in `hardware_contract.py`.

## Added

- `Velocity-Lilgreen-Stand-ST3215-Loaded-v0`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v0`
- Stage-B loaded whole-body trajectory envelope layered after the v1.3.1 Stage-A
  suspended single-joint ST3215 response model.
- Conservative knee-only crouch/stand-return direction conditioning.
- Conservative knee stand-return tau-response proxy multiplier.
- Independent loaded-envelope scale randomization.
- Loaded-envelope training diagnostics and offline analysis output.
- Export metadata that records the loaded dataset, fit arrays, combination rule,
  direction-conditioning weights, tau-response scales, and randomization range.
- Track 2 loaded standing-transition handoff, model JSON/YAML, synthesis archive,
  and the 500-iteration v1.3.1 updated-limit reference analysis.

## Preserved

- action contract v3;
- ±0.20 rad residual mapping around q_default;
- 45-D actor observation and 12-D action contract;
- 50 Hz policy rate;
- measured nominal q_default base-COM target;
- v1.3.1 response-delay and Stage-A actuator response configuration;
- v1.2.3 and v1.3.x task registrations.

## Modeling boundary

The v2 loaded standing sweep is used as an empirical trajectory-response calibration
layer. v1.4.0 does not infer physical stiffness, physical damping, joint torque from
`load_ratio`, or balance stability from the loaded archive.

See `V1_4_0_ST3215_LOADED_TASKS.md` for task names, model structure, diagnostics,
and training commands.
