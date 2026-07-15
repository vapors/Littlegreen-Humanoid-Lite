# Littlegreen-Humanoid-Lite v2.0.0 Rename Audit

Prepared from the authoritative Track 1 archive `Berkeley-Humanoid-Lite_v1_4_8.zip`, with `littlegreen_ros2_ws_v2_8_0` used only as a naming and deployment-contract reference. The supplied `policy_policy_latest.zip` was integrated as the current deployable policy pair.

## Source integrity

| Input | SHA-256 |
|---|---|
| `Berkeley-Humanoid-Lite_v1_4_8.zip` | `2dadc98398e226967bafad168fc63d205e9483ae9515f13fb6281bcc7918b05d` |
| `littlegreen_ros2_ws_v2_8_0(1).zip` | `5442c53f695d96daa02ed6c48894fc09c9cc175785e23bad637c1ebc30a853c8` |
| `policy_policy_latest.zip` | `b91cb2504c4d9092586d23c63cc486754d92e8db5fc455118a84b2abd008229a` |

Both authoritative project archives passed ZIP integrity checks before modification. The original Track 1 extraction remained untouched during the refactor.

## Implemented package rename

| v1.4.8 name | v2.0.0 name |
|---|---|
| `Berkeley-Humanoid-Lite_v1_4_8` | `Littlegreen-Humanoid-Lite_v2_0_0` |
| `berkeley_humanoid_lite` | `littlegreen_humanoid_lite` |
| `berkeley_humanoid_lite_assets` | `littlegreen_humanoid_lite_assets` |
| `berkeley_humanoid_lite_lowlevel` | `littlegreen_humanoid_lite_lowlevel` |
| `export_policy_bhl.py` / duplicate exporters | `export_policy_littlegreen.py` canonical implementation |
| `export_policy.py` | retained as a small compatibility entry-point wrapper |
| `bhl_biped_scene.xml` | `littlegreen_biped_scene.xml` |
| `bhl_scene.xml` | `littlegreen_scene.xml` |

No compatibility alias packages using the former Python import names were added.

## Project and extension metadata

- Root workspace name: `littlegreen-humanoid-lite-workspace`
- Root version: `2.0.0`
- All three Python distribution versions: `2.0.0`
- Root `VERSION`: `2.0.0`
- UV workspace members and first-party dependency mappings updated to the three Littlegreen package names.
- Environment, Docker, script, test, and active documentation paths updated.
- Export default deployment directory updated to `~/littlegreen_ros2_ws/src/littlegreen_biped_pkg/src/configs`.
- Active `_bhl_*` private runtime identifiers were renamed to `_littlegreen_*` without changing their calculations.

## Policy integration

The supplied policy pair was installed as:

- `configs/policy.onnx`
- `configs/policy_latest.yaml`

The ONNX file is an exact byte-for-byte copy of the uploaded file. Its SHA-256 is:

`3e663fbde931d8056fe8461a19a6d7bc759329d9c36b3418e4bb3a95196bf627`

The paired metadata remains truthful to the supplied artifact:

- Task: `Velocity-Lilgreen-Hardware-ST3215-Loaded-v7`
- Observations: `47`
- Actions: `12`
- Action contract: v4

The policy was **not** relabeled as the v8 task. Absolute source checkpoint/export paths were retained only as provenance fields; the active policy path is portable and relative to `configs/`.

## Preserved task registry

All 24 task IDs are unchanged:

- `Velocity-Berkeley-Humanoid-Lite-Biped-v0`
- `Velocity-Berkeley-Humanoid-Lite-v0`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v0`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v1`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v2`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v3`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v4`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v5`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s2`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s3`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v6`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v7`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v8`
- `Velocity-Lilgreen-Hardware-ST3215-v0`
- `Velocity-Lilgreen-Hardware-v0`
- `Velocity-Lilgreen-Humanoid-v0`
- `Velocity-Lilgreen-Stand-ST3215-Loaded-v0`
- `Velocity-Lilgreen-Stand-ST3215-Loaded-v5`
- `Velocity-Lilgreen-Stand-ST3215-Loaded-v5s`
- `Velocity-Lilgreen-Stand-ST3215-Loaded-v5s2`
- `Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3`
- `Velocity-Lilgreen-Stand-ST3215-v0`
- `Velocity-Lilgreen-Stand-v0`

The two Berkeley task IDs remain as deliberate legacy public identifiers. The `Velocity-Lilgreen-*` family was already the established public task namespace and was also preserved.

## Behavior-preservation boundary

The following were held invariant:

- canonical 12-action joint order;
- `q_default` values;
- calibrated physical joint limits;
- vector residual scales;
- action contract v4 math and target clipping order;
- bounded previous-action observation semantics;
- 47-D phase-guided observation layout used by v7/v8;
- gait phase semantics;
- ST3215 Stage-A constants;
- loaded Stage-B envelope arrays;
- reward weights and reward functions;
- curriculum thresholds and stage behavior;
- environment physics and randomization logic;
- policy export calculations;
- rollout analyzer calculations.

The 43 Python files under the velocity task implementation compare exactly to v1.4.8 after package/private-identifier normalization. Literal contract snapshots also compare exactly: 23 values in `hardware_contract.py`, 20 in `st3215_actuator_model.py`, and 18 in `st3215_loaded_actuator_model.py`.

## Historical and upstream allowlist

The following Berkeley/Lilgreen strings intentionally remain:

- copyright and contributor attribution;
- upstream repository, paper, website, and documentation URLs;
- the two legacy Berkeley task IDs;
- historical v1.x documentation, run names, checkpoint metadata, and analyzer descriptions;
- frozen Track 2 dataset provenance identifiers;
- source checkpoint/export paths in the supplied policy metadata;
- legacy Berkeley-named MJCF/URDF/USD robot layers needed by historical task and asset compatibility;
- binary USD crate contents, which were not rewritten.

These are provenance or compatibility records, not stale first-party package imports.

## Additional discrepancies corrected

During implementation, the following issues were corrected without altering training behavior:

1. The previously omitted third distribution, `berkeley_humanoid_lite_lowlevel`, was included in the rename.
2. Package metadata that still reported `1.4.1` was synchronized to `2.0.0`.
3. A root `VERSION` file and package `__version__` values were added.
4. Missing package support files were added: the main-package README and the asset package `__init__.py`.
5. Three byte-identical exporters were consolidated into one implementation plus one stable wrapper.
6. The v1.4.4 “static” test was made genuinely static so it does not import Isaac Lab during offline validation; its assertions still inspect the original contract values.
7. A pre-existing blank line before the XML declaration in `lilgreen_robot.xml` was removed so standard XML parsers accept the file. No XML model values changed.
8. Generated `.pytest_cache`, `__pycache__`, `*.egg-info`, wheel-build directories, stale nested Git metadata, and a stale `test.zip` were excluded from the release.
9. README links that had been mechanically renamed to nonexistent Littlegreen upstream URLs were restored and explicitly labeled as upstream references.
10. The uploaded latest policy was identified as v7/47-D rather than being silently presented as a v8 export.

## Frozen artifacts

The following trees are byte-identical to v1.4.8:

| Tree | Files | Result |
|---|---:|---|
| `logs/` | 38 | exact |
| `checkpoints/` | 16 | exact |
| `data/` | 11 | exact |
| binary USD files | 8 | exact |

Detailed static results are in `VALIDATION_REPORT.md`. Installation and compatibility guidance is in `MIGRATION_NOTES.md`.
