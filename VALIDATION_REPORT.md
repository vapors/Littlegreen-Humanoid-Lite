# Littlegreen-Humanoid-Lite v2.0.0 Validation Report

## Completed offline validation

| Validation | Result |
|---|---|
| Source ZIP integrity | PASS |
| Python syntax compilation (`compileall`) | PASS |
| Offline first-party tests | PASS — 47 passed |
| Velocity-task source equivalence after name-only normalization | PASS — 43/43 files |
| Task registry ID comparison | PASS — 24/24 unchanged |
| Task entry-point comparison after package normalization | PASS |
| `hardware_contract.py` literal snapshot | PASS — 23/23 unchanged |
| ST3215 Stage-A literal snapshot | PASS — 20/20 unchanged |
| loaded Stage-B literal snapshot | PASS — 18/18 unchanged |
| Active stale old-package import scan | PASS — none found |
| Active stale package metadata scan | PASS — none found |
| YAML parsing | PASS — 7 files |
| XML/URDF parsing | PASS — 9 files |
| JSON parsing | PASS — 3 files |
| Exporter/analyzer syntax | PASS through compileall |
| Required renamed file layout | PASS |
| Removed old exporter/scene paths | PASS |
| Latest ONNX exact-copy comparison | PASS |
| Latest policy YAML/ONNX SHA pairing | PASS |
| Latest policy semantic comparison after identity/path normalization | PASS |
| Frozen `logs/`, `checkpoints/`, `data/`, and USD hashes | PASS |
| Final release archive integrity | reported in release manifest after packaging |

### Test result

```text
...............................................                          [100%]
47 passed in 0.16s
```

### Policy contract checked

```text
task: Velocity-Lilgreen-Hardware-ST3215-Loaded-v7
num_observations: 47
num_actions: 12
action_contract_version: 4
previous_action_observation: bounded_normalized_action
policy_sha256: 3e663fbde931d8056fe8461a19a6d7bc759329d9c36b3418e4bb3a95196bf627
```

The uploaded policy is a valid v7-labelled 47-D policy pair. Its metadata was not changed to v8.

### Packaging metadata check

All three package metadata builds completed successfully and reported the new names/version. Release wheels are intentionally not shipped: the inherited package projects use explicit top-level package declarations, and changing their package-discovery design would be a separate packaging refactor rather than a rename-only change.

### UV lock check

`uv lock --check --offline` was attempted but could not execute because the validation container has no Python 3.10 interpreter and offline mode prevents `uv` from acquiring one. The first-party package names, workspace members, source mappings, and version records in `uv.lock` were inspected and updated statically.

## Not run in this environment

The following require the configured Isaac Lab/Isaac Sim training host or the real deployment environment and are **not claimed as validated**:

- Isaac Lab extension import under the supported Python 3.10 environment;
- Gym environment registration at runtime;
- PhysX environment creation or stepping;
- v0, v5s3, v7, or v8 simulation behavior;
- PPO training, resume, or checkpoint compatibility;
- end-to-end policy export from a `.pt` checkpoint;
- ONNX Runtime inference or graph-shape inspection (`onnxruntime` unavailable here);
- Isaac Sim video generation;
- ROS 2 deployment and Orange Pi execution;
- physical robot standing, walking, calibration, or servo safety validation.

## Required training-host acceptance checks

Run these before designating v2.0.0 as the active training workspace:

1. Create or activate the project’s supported Python 3.10 / Isaac Lab environment.
2. Install the three renamed packages in editable mode.
3. Run `python scripts/list_envs.py` and confirm all 24 IDs register.
4. Instantiate at minimum:
   - `Velocity-Lilgreen-Stand-v0`
   - `Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3`
   - `Velocity-Lilgreen-Hardware-ST3215-Loaded-v7`
   - `Velocity-Lilgreen-Hardware-ST3215-Loaded-v8`
5. Confirm v7/v8 expose 47 actor observations and 12 actions.
6. Step each selected environment with deterministic zero actions and check for finite observations/rewards.
7. Run one short training smoke test without resuming a historical checkpoint.
8. Export a known renamed-workspace checkpoint with `export_policy_littlegreen.py` and inspect the YAML/ONNX pair.
9. Run the v1.4.8 analyzer against a fresh rollout and compare key fields to the pre-rename baseline.
10. Deploy only after the Track 2 policy-contract preflight confirms the 47-D/v4 interface.

## Compatibility caution

Historical Python pickle files can embed their original module path. Because v2.0.0 deliberately provides no `berkeley_humanoid_lite` alias package, a historical `.pkl` may fail to unpickle under the new namespace. The artifacts are preserved byte-for-byte; any required pickle migration should be a controlled, one-off operation on a copy. PyTorch checkpoint compatibility must likewise be verified on the training host rather than assumed.
