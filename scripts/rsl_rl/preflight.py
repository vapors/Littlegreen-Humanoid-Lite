"""v11.1.0 live Isaac Lab articulation / action contract preflight (no policy).

Launches a single free-root physics environment and reads the real spawned
PhysX articulation; it does not infer mass from an on-disk USD audit alone.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from isaaclab.app import AppLauncher

TASK="Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111"
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default=TASK)
p.add_argument("--output", default="logs/v11_preflight.json")
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.task != TASK: p.error("v11.1.0 preflight requires the current v111 task")
a.enable_cameras=False
app=AppLauncher(a).app
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv,multi_agent_to_single_agent
import littlegreen_humanoid_lite.tasks
from littlegreen_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,ST3215_PEAK_TORQUE_NM,HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD)

out=Path(a.output).expanduser();out.parent.mkdir(parents=True,exist_ok=True)
report={"task":TASK,"status":"NOT_RUN","mass_provisional":True}
raw=None; vec=None
try:
    cfg=parse_env_cfg(TASK,device=a.device,num_envs=1)
    assert abs(cfg.sim.dt-0.005)<1e-9
    assert cfg.decimation==4
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is False
    assert cfg.events.actuator_gains is None
    assert cfg.curriculum is None
    assert "lilgreen_mass_v2926.usd" in cfg.scene.robot.spawn.usd_path
    usd=Path(cfg.scene.robot.spawn.usd_path)
    if not usd.is_file():raise RuntimeError(f"Missing corrected USD: {usd}")
    with usd.open('rb') as fh: usd_sha=hashlib.file_digest(fh,'sha256').hexdigest()
    for group in ("legs","ankles"):
        ac=cfg.scene.robot.actuators[group]
        assert (float(ac.stiffness),float(ac.damping))==(18.0,1.5)
        assert math.isclose(float(ac.effort_limit_sim),ST3215_PEAK_TORQUE_NM,abs_tol=1e-5)
    assert cfg.observations.policy.gait_phase.params["phase_mode"]=="neutral_static"
    assert list(cfg.actions.joint_pos.joint_names)==list(ACTIONABLE_JOINTS_V1_2_3)
    assert list(cfg.actions.joint_pos.lower_limits)==list(HARDWARE_LOWER_LIMIT_RAD)
    assert list(cfg.actions.joint_pos.upper_limits)==list(HARDWARE_UPPER_LIMIT_RAD)
    raw=gym.make(TASK,cfg=cfg,render_mode=None)
    if isinstance(raw.unwrapped,DirectMARLEnv):raw=multi_agent_to_single_agent(raw)
    vec=RslRlVecEnvWrapper(raw)
    obs,_=vec.reset()
    if obs.shape[-1]!=47: raise RuntimeError(f"Actor obs shape mismatch {obs.shape}")
    sim=vec.unwrapped
    if abs(sim.step_dt-0.020)>1e-6 or abs(sim.physics_dt-0.005)>1e-6:
        raise RuntimeError(f"Bad runtime time steps: {sim.step_dt}/{sim.physics_dt}")
    robot=sim.scene['robot']
    body_names=list(robot.body_names)
    link_masses=[float(x) for x in robot.root_physx_view.get_masses()[0].detach().cpu().tolist()]
    if len(body_names)!=len(link_masses):raise RuntimeError('Rigid-body name/mass count mismatch')
    link_map=dict(zip(body_names,link_masses))
    total=sum(link_masses)
    audit=json.loads(usd.with_suffix('.mass_audit.json').read_text())
    expected={key:float(v['new_mass_kg']) for key,v in audit['links'].items()}
    missing=sorted(set(expected)-set(link_map))
    differences={name:{'expected':value,'observed':link_map.get(name)} for name,value in expected.items()
                 if name not in link_map or abs(value-link_map[name])>0.008}
    if abs(total-3.852)>0.03 or missing or differences:
        raise RuntimeError(f"Runtime mass audit mismatch total={total} missing={missing} link_differences={differences}")
    term=sim.action_manager.get_term('joint_pos')
    joint_ids,names=robot.find_joints(term.joint_names,preserve_order=True)
    if list(names)!=list(ACTIONABLE_JOINTS_V1_2_3) or len(names)!=12:
        raise RuntimeError(f"Actuated joint order mismatch: {names}")
    q_default=[float(x) for x in term._default[0].detach().cpu().tolist()]
    action_scale=[float(x) for x in term._residual_scale[0].detach().cpu().tolist()]
    report.update({"status":"PASS_CONFIG_AND_RUNTIME","runtime_mass_kg":total,
        "expected_provisional_mass_kg":float(audit['total_kg']),
        "usd_path":str(usd),"usd_sha256":usd_sha,
        "body_mass_kg":link_map,"root_free":True,"physics_dt_s":sim.physics_dt,
        "policy_dt_s":sim.step_dt,"actor_obs_count":obs.shape[-1],
        "action_count":len(names),"canonical_joint_order":names,
        "q_default_rad":q_default,"residual_scale_rad":action_scale,
        "phase_mode":"neutral_static","contact_sensor_bodies":list(sim.scene['contact_forces'].body_names),
        "legs_gain":{"stiffness":18.,"damping":1.5},
        "ankles_gain":{"stiffness":18.,"damping":1.5},
        "torque_limit_nm":float(ST3215_PEAK_TORQUE_NM),
        "warnings":["Provisional link masses; weigh actual hardware", 
                    "This preflight checks articulation setup, not free-standing stability or foot penetration", 
                    "No PPO checkpoint was loaded or trained"]})
    print('[PREFLIGHT] PASS: runtime articulated mass and v11.1.0 k150 contract',flush=True)
except Exception as exc:
    report['status']='FAIL'; report['error']=repr(exc)
    raise
finally:
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    if vec is not None:vec.close()
    elif raw is not None:raw.close()
    app.close()
