# Berkeley Humanoid Lite v1.4.5s3 — Forward-COM Height Refinement

This branch is a small stand-stabilization refinement on top of v1.4.5s2.

Goals:
- Preserve the v1.4.5/v5s/v5s2 contract-v4 vector residual profile.
- Keep the stabilized athletic q_default and larger hip/knee/ankle pitch authority.
- Raise the stand COM-height target slightly to reduce ankle-pitch bracing while shifting the COM forward.
- Move the stand COM-over-feet target farther forward and nudge the weak forward-lean cue.

New tasks:
- `Velocity-Lilgreen-Stand-ST3215-Loaded-v5s3`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v5s3`

Key constants:
- stand height target: `0.460 m`
- moving height target: `0.430 m`
- COM-over-feet forward target: `0.070 m`
- COM forward band: `0.060–0.080 m`
- projected gravity forward lean target: `0.065`

The analyzer hotfix for forward-COM diagnostics is included.
