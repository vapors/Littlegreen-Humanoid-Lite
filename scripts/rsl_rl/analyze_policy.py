"""LittleGreen v11 canonical acquisition checkpoint analyzer."""
from __future__ import annotations

import argparse, csv, json, os, traceback, math
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
import cli_args  # isort: skip
from command_override import apply_velocity_command_override  # isort: skip

parser=argparse.ArgumentParser(description="Analyze LittleGreen v11.1 bilateral-gait acquisition.")
parser.add_argument("--task",default="Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111")
parser.add_argument("--num_envs",type=int,default=256)
parser.add_argument("--steps",type=int,default=2000)
parser.add_argument("--output_dir",type=str,default=None)
parser.add_argument("--log_root",type=str,default=None)
parser.add_argument("--disable_fabric",action="store_true",default=False)
parser.add_argument("--eval_force_command",nargs=3,type=float,default=None,metavar=("VX","VY","YAW"))
parser.add_argument("--eval_force_command_name",type=str,default="base_velocity")
parser.add_argument("--progress_interval",type=int,default=100)
parser.add_argument("--standing_settle_steps",type=int,default=100)
parser.add_argument("--standing_steps",type=int,default=500)
parser.add_argument(
    "--handoff_pose_late_fraction",
    type=float,
    default=0.50,
    help="Fraction of the post-settle zero-command standing probe considered 'late' for handoff-pose extraction (default: last 50%%).",
)
parser.add_argument(
    "--handoff_pose_min_viability",
    type=float,
    default=0.80,
    help="Minimum v10.3 viability factor for zero-command handoff-pose samples (default: 0.80).",
)
cli_args.add_rsl_rl_args(parser); AppLauncher.add_app_launcher_args(parser)
args_cli=parser.parse_args()
if not (0.0 < args_cli.handoff_pose_late_fraction <= 1.0):
    parser.error("--handoff_pose_late_fraction must be in (0, 1].")
if not (0.0 <= args_cli.handoff_pose_min_viability <= 1.0):
    parser.error("--handoff_pose_min_viability must be in [0, 1].")
TASK="Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111"
if args_cli.task!=TASK: parser.error(f"analyze_policy.py accepts only {TASK}")
args_cli.enable_cameras=False
app_launcher=AppLauncher(args_cli); simulation_app=app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import littlegreen_humanoid_lite.tasks  # noqa:F401,E402

WHOLE_FEET=["leg_left_ankle_roll","leg_left_toe_pitch","leg_right_ankle_roll","leg_right_toe_pitch"]

def _write_json(path:Path,payload:Any): path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _f(x): return float(x.detach().cpu().item()) if isinstance(x,torch.Tensor) else float(x)
def _smooth01(t): return t*t*(3.0-2.0*t)
def _viability(robot):
    upright=torch.clamp(-robot.data.projected_gravity_b[:,2],-1.0,1.0)
    cf=math.cos(math.radians(25.0)); cz=math.cos(math.radians(55.0))
    ot=torch.clamp((upright-cz)/(cf-cz),0.0,1.0); orient=_smooth01(ot)
    ht=torch.clamp((robot.data.root_com_pos_w[:,2]-0.340)/(0.400-0.340),0.0,1.0); height=_smooth01(ht)
    return orient*height

def main():
    env_cfg=parse_env_cfg(args_cli.task,device=args_cli.device,num_envs=args_cli.num_envs,use_fabric=not args_cli.disable_fabric)
    agent_cfg:RslRlOnPolicyRunnerCfg=cli_args.parse_rsl_rl_cfg(args_cli.task,args_cli)
    log_root=os.path.abspath(os.path.expanduser(args_cli.log_root)) if args_cli.log_root else os.path.abspath(os.path.join("logs","rsl_rl",agent_cfg.experiment_name))
    checkpoint=get_checkpoint_path(log_root,agent_cfg.load_run,agent_cfg.load_checkpoint)
    run_dir=Path(checkpoint).parent; out=Path(args_cli.output_dir) if args_cli.output_dir else run_dir/"analysis"/datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True,exist_ok=True); status=out/"analysis_status.json"
    _write_json(status,{"status":"starting","task":args_cli.task,"checkpoint":checkpoint,"steps":args_cli.steps})
    print(f"[ANALYZE] Task:       {args_cli.task}",flush=True); print(f"[ANALYZE] Checkpoint: {checkpoint}",flush=True); print(f"[ANALYZE] Output:     {out}",flush=True)
    base_env=vec_env=None
    try:
        base_env=gym.make(args_cli.task,cfg=env_cfg,render_mode=None)
        if isinstance(base_env.unwrapped,DirectMARLEnv): base_env=multi_agent_to_single_agent(base_env)
        vec_env=RslRlVecEnvWrapper(base_env); runner=OnPolicyRunner(vec_env,agent_cfg.to_dict(),log_dir=None,device=agent_cfg.device); runner.load(checkpoint)
        env=vec_env.unwrapped; policy=runner.get_inference_policy(device=env.device); robot=env.scene["robot"]
        action_term=env.action_manager.get_term("joint_pos"); joint_ids,joint_names=robot.find_joints(action_term.joint_names,preserve_order=True)
        sensor=env.scene.sensors["contact_forces"]; contact_ids,contact_names=sensor.find_bodies(WHOLE_FEET,preserve_order=True); body_ids,body_names=robot.find_bodies(WHOLE_FEET,preserve_order=True)
        if list(contact_names)!=WHOLE_FEET or list(body_names)!=WHOLE_FEET: raise RuntimeError("v6.7 analyzer requires ordered whole-foot bodies")
        obs,_=vec_env.get_observations();
        if obs.shape[-1]!=47: raise RuntimeError(f"Expected 47-D actor obs, got {tuple(obs.shape)}")
        print(f"[ANALYZE] observations={tuple(obs.shape)} phase_tail_mean={obs[:,45:47].mean(0).tolist()}",flush=True)
        force=None
        if args_cli.eval_force_command is not None:
            force=torch.tensor(args_cli.eval_force_command,device=env.device,dtype=obs.dtype).repeat(env.num_envs,1); apply_velocity_command_override(env,force,command_name=args_cli.eval_force_command_name,warn=True)

        dev=env.device; nenv=env.num_envs; n=float(nenv); dt=float(env.step_dt); eps=1e-9
        z=lambda: torch.zeros((),device=dev,dtype=torch.float64)
        samples=z(); moving_samples=z(); falls=z(); both=z(); single=z(); none=z(); partial=z(); viability_sum=z(); low_viability=z()
        raw_vel_along=z(); viable_vel_along=z(); forward_low_viability=z(); base_h=z(); com_lat=z(); com_fwd=z(); slide_sum=z(); loaded_count=z()
        cmd_sum=torch.zeros(3,device=dev,dtype=torch.float64); actual_sum=torch.zeros(3,device=dev,dtype=torch.float64); track_sq=torch.zeros(3,device=dev,dtype=torch.float64)
        raw_release=torch.zeros(2,device=dev,dtype=torch.float64); low_v_release=torch.zeros(2,device=dev,dtype=torch.float64); qual=torch.zeros(2,device=dev,dtype=torch.float64)
        land=torch.zeros(2,device=dev,dtype=torch.float64); confirmed_count=torch.zeros(2,device=dev,dtype=torch.float64); firsts=z(); alternating=z(); repeats=z()
        progress_sum=torch.zeros(2,device=dev,dtype=torch.float64); velratio_sum=torch.zeros(2,device=dev,dtype=torch.float64); coherence_sum=torch.zeros(2,device=dev,dtype=torch.float64)
        landing_load_sum=torch.zeros(2,device=dev,dtype=torch.float64); landing_count=torch.zeros(2,device=dev,dtype=torch.float64); takeover_score_sum=z(); takeover_n=z(); takeover_gt50=z()
        handoff_count=torch.zeros(2,device=dev,dtype=torch.float64); opposite_release_count=torch.zeros(2,device=dev,dtype=torch.float64)
        init_score_sum=torch.zeros(2,device=dev,dtype=torch.float64); init_candidate=torch.zeros(2,device=dev,dtype=torch.float64); init_active=torch.zeros(2,device=dev,dtype=torch.float64)
        init_clear_2=torch.zeros(2,device=dev,dtype=torch.float64); init_clear_5=torch.zeros(2,device=dev,dtype=torch.float64); init_clear_10=torch.zeros(2,device=dev,dtype=torch.float64); init_clear_15=torch.zeros(2,device=dev,dtype=torch.float64); init_rising=torch.zeros(2,device=dev,dtype=torch.float64); init_unloaded=torch.zeros(2,device=dev,dtype=torch.float64)
        liftoff_pose_sum=torch.zeros((2,2),device=dev,dtype=torch.float64); liftoff_pose_count=torch.zeros(2,device=dev,dtype=torch.float64)
        v66_start=torch.zeros(2,device=dev,dtype=torch.float64); v66_replant=torch.zeros(2,device=dev,dtype=torch.float64); v66_handoff=torch.zeros(2,device=dev,dtype=torch.float64); v66_release=torch.zeros(2,device=dev,dtype=torch.float64); v66_readiness=z(); v66_traj=z(); v66_takeover=z(); v66_stage=torch.zeros(3,device=dev,dtype=torch.float64); v66_expected=torch.zeros(3,device=dev,dtype=torch.float64); v66_pending=z(); v66_level=z(); v66_takeover_best=z()
        v66_ready_load=z(); v66_ready_com=z(); v66_ready_n=z(); v66_takeover_load=z(); v66_takeover_com=z(); v66_takeover_n=z(); v66_takeover_load50=z(); v66_takeover_com56=z()
        # v6.7 swing-commitment diagnostics: higher clearance ladder, stance-relative
        # swing-foot command velocity, foot-relative COM drive, and pre-swing base motion.
        v67_candidate=z(); v67_clear3=z(); v67_clear10=z(); v67_clear22=z(); v67_clear40=z(); v69_clear50=z(); v69_clear60=z()
        v67_rel_foot_cmd_sum=z(); v67_rel_foot_cmd_pos=z(); v67_rel_foot_cmd_n=z(); v67_rel_foot_cmd_ge20=z()
        v67_com_toward_stance_sum=z(); v67_com_toward_stance_n=z(); v67_ready_n=z(); v67_ready_timer_sum=z(); v67_ready_timer_ge018=z()
        v67_pre_swing_base_along=z(); v67_pre_swing_n=z(); v67_meaningful_swing_base_along=z(); v67_meaningful_swing_n=z()
        v67_coherent_rise_sum=z(); v67_coherent_rise_n=z(); v67_stance_speed_sum=z()
        # v9.8 analyzer-only COM command-frame audit.  The active v6.7 COM reward
        # is intentionally unchanged; these metrics quantify whether its full-planar
        # toward-stance direction contains a backward command component.
        v98_com_gate_n=z(); v98_com_dir_cmd_dot=z(); v98_com_backward_dir=z(); v98_com_rel_cmd_vel=z(); v98_com_lat_toward_vel=z(); v98_com_stance_cmd_offset=z(); v98_com_stance_lat_abs=z()
        v99_com_gate_n=z(); v99_com_lat_toward_vel=z(); v99_com_lat_closeness=z(); v99_com_lat_offset_abs=z(); v99_com_rel_cmd_vel=z()
        # v6.8 clean-swing diagnostics: soft command-conditioned stride/swing targets,
        # remembered frontier/backtracking, and touchdown progress retention.
        v68_swing_n=z(); v68_stride_horizon_sum=z(); v68_target_speed_sum=z(); v68_backtrack_sum=z(); v68_backtrack_gt6=z(); v68_backtrack_gt14=z(); v68_frontier_factor_sum=z()
        v68_replant_n=z(); v68_retention_sum=z(); v68_touchdown_progress_sum=z(); v68_peak_progress_sum=z(); v68_touchdown_horizon_sum=z()
        v69_takeover_n=z(); v69_retreat_sum=z(); v69_retreat_gt5=z(); v69_retreat_gt12=z()
        v69_handoff_n=z(); v69_handoff_retention_sum=z(); v69_support_advance_sum=z(); v69_support_horizon_sum=z(); v69_support_score_sum=z()
        v69_com_n=z(); v69_com_advance_sum=z(); v69_com_horizon_sum=z(); v69_com_score_sum=z()
        hip_knee_idx=torch.tensor([[joint_names.index("leg_left_hip_pitch_joint"),joint_names.index("leg_left_knee_pitch_joint")],[joint_names.index("leg_right_hip_pitch_joint"),joint_names.index("leg_right_knee_pitch_joint")]],device=dev,dtype=torch.long)
        swing_joint_idx=torch.tensor([
            [joint_names.index("leg_left_hip_pitch_joint"), joint_names.index("leg_left_knee_pitch_joint"), joint_names.index("leg_left_ankle_pitch_joint")],
            [joint_names.index("leg_right_hip_pitch_joint"), joint_names.index("leg_right_knee_pitch_joint"), joint_names.index("leg_right_ankle_pitch_joint")],
        ],device=dev,dtype=torch.long)
        # v9.8 reward diagnostics: passage and WORLD-rate are clocked from the
        # validated v6.3 confirmed-step mechanical support-takeover stream.
        v98_pass_n=z(); v98_pass_margin=z(); v98_pass_quality=z(); v98_pass_current_score=z(); v98_pass_pair_valid=z(); v98_pass_pair_quality=z(); v98_pass_pair_score=z(); v98_pass_dense_score=z()
        v98_pass_foot_n=torch.zeros(2,device=dev,dtype=torch.float64); v98_pass_foot_margin=torch.zeros(2,device=dev,dtype=torch.float64); v98_pass_foot_quality=torch.zeros(2,device=dev,dtype=torch.float64); v98_pass_foot_score=torch.zeros(2,device=dev,dtype=torch.float64)
        v98_rate_n=z(); v98_rate_alt=z(); v98_rate_foot_n=torch.zeros(2,device=dev,dtype=torch.float64); v98_rate_advance=torch.zeros(2,device=dev,dtype=torch.float64); v98_rate_elapsed=torch.zeros(2,device=dev,dtype=torch.float64); v98_rate_mps=torch.zeros(2,device=dev,dtype=torch.float64); v98_rate_ratio=torch.zeros(2,device=dev,dtype=torch.float64); v98_rate_score=torch.zeros(2,device=dev,dtype=torch.float64); v98_rate_command=torch.zeros(2,device=dev,dtype=torch.float64); v98_epoch_resets=z()
        # v10.1 bilateral WORLD-rate coupling diagnostics.
        v101_rate_n=z(); v101_rate_current_quality=z(); v101_rate_current_score=z(); v101_rate_pair_valid=z(); v101_rate_pair_quality=z(); v101_rate_pair_score=z(); v101_rate_dense_score=z()
        v101_rate_foot_n=torch.zeros(2,device=dev,dtype=torch.float64); v101_rate_foot_pair_valid=torch.zeros(2,device=dev,dtype=torch.float64); v101_rate_foot_dense_score=torch.zeros(2,device=dev,dtype=torch.float64)
        # Turn-command diagnostics.  These remain useful both for random-command runs and forced yaw probes.
        yaw_active_n=z(); yaw_abs_cmd=z(); yaw_signed_actual=z(); yaw_ratio_sum=z(); yaw_correct=z(); yaw_strong_n=z(); turn_focus_samples=z()
        command_term=env.command_manager.get_term(args_cli.eval_force_command_name)
        # Event-aligned hip/knee/ankle diagnostics.  These do not affect reward.
        swing_start_pose_sum=torch.zeros((2,3),device=dev,dtype=torch.float64); swing_start_n=torch.zeros(2,device=dev,dtype=torch.float64)
        swing_replant_pose_sum=torch.zeros((2,3),device=dev,dtype=torch.float64); swing_replant_n=torch.zeros(2,device=dev,dtype=torch.float64)
        swing_handoff_pose_sum=torch.zeros((2,3),device=dev,dtype=torch.float64); swing_handoff_n=torch.zeros(2,device=dev,dtype=torch.float64)
        swing_peak_pose_sum=torch.zeros((2,3),device=dev,dtype=torch.float64); swing_excursion_sum=torch.zeros((2,3),device=dev,dtype=torch.float64); swing_diag_n=torch.zeros(2,device=dev,dtype=torch.float64)
        swing_diag_active=torch.zeros((nenv,2),device=dev,dtype=torch.bool); swing_diag_replanted=torch.zeros_like(swing_diag_active)
        swing_diag_start=torch.zeros((nenv,2,3),device=dev); swing_diag_min=torch.zeros_like(swing_diag_start); swing_diag_max=torch.zeros_like(swing_diag_start); swing_diag_peak_pose=torch.zeros_like(swing_diag_start); swing_diag_peak_dev=torch.zeros_like(swing_diag_start)
        torque_hard=torch.zeros(len(joint_names),device=dev,dtype=torch.float64); torque_abs=torch.zeros_like(torque_hard); q_sum=torch.zeros_like(torque_hard); raw_excess=torch.zeros_like(torque_hard); bound_sat=torch.zeros_like(torque_hard); joint_samples=0.0
        torque_limit=float(getattr(env_cfg.scene.robot.actuators["legs"],"effort_limit_sim",2.94))

        prev_any=None; prev_center=None; active=torch.zeros((nenv,2),device=dev,dtype=torch.bool); qualified=torch.zeros_like(active); swing_min_v=torch.ones((nenv,2),device=dev)
        liftoff_xy=torch.zeros((nenv,2,2),device=dev); liftoff_dir=torch.zeros_like(liftoff_xy); liftoff_cmd=torch.ones((nenv,2),device=dev); air_quality=torch.ones((nenv,2),device=dev)
        landing=torch.zeros_like(active); main_seen=torch.zeros_like(active); toe_seen=torch.zeros_like(active); landing_timer=torch.zeros((nenv,2),device=dev)
        confirm=torch.zeros_like(active); confirm_timer=torch.zeros((nenv,2),device=dev); confirm_min_v=torch.ones((nenv,2),device=dev); last_foot=torch.full((nenv,),-1,device=dev,dtype=torch.long)
        takeover=torch.zeros_like(active); takeover_timer=torch.zeros((nenv,2),device=dev); takeover_initial_dist=torch.zeros((nenv,2),device=dev)
        handoff_done=torch.zeros_like(active); release_watch_active=torch.zeros((nenv,),device=dev,dtype=torch.bool); release_watch_new_foot=torch.full((nenv,),-1,device=dev,dtype=torch.long); release_watch_timer=torch.zeros((nenv,),device=dev)

        v111_quality_sum=torch.zeros(2, device=dev, dtype=torch.float64)
        v111_pair_sum=z(); v111_reward_sum=z(); v111_active_samples=z()
        v111_attempts=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_credit=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_recovery=z(); v111_strafe=z(); v111_receiver=z()
        v111_receiver_side=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_receiver_load=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_receiver_com=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_receiver_completed=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_max_streak=torch.zeros((),device=dev,dtype=torch.float64)
        # v11.1 R2 exact reward-manager contact gates (current_contact_time based).
        v111_whole_air_samples=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_partial_contact_samples=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_pivot_suppressed_samples=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_ground_reward_leak_samples=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_main_contact=torch.zeros(2,device=dev,dtype=torch.float64)
        v111_toe_contact=torch.zeros(2,device=dev,dtype=torch.float64)
        for step in range(args_cli.steps):
            if force is not None: apply_velocity_command_override(env,force,command_name=args_cli.eval_force_command_name)
            with torch.inference_mode(): act=policy(obs); obs,_,dones,_=vec_env.step(act)[:4]
            dones=dones.to(device=dev,dtype=torch.bool); falls+=dones.sum()
            if hasattr(env, "_lgh111_cached_result"):
                d111=env._lgh111_cached_result
                v111_q=d111['quality']
                airborne111=d111['airborne']
                v111_whole_air_samples+=airborne111.double().sum(0)
                v111_partial_contact_samples+=d111['partial_contact'].double().sum(0)
                v111_pivot_suppressed_samples+=d111['pivot_credit_blocked'].double().sum(0)
                v111_ground_reward_leak_samples+=((d111['quality']>0)&(~airborne111)).double().sum(0)
                # This uses the same ContactSensor.current_contact_time source
                # and L_main,L_toe,R_main,R_toe order as the actual reward.
                time111=sensor.data.current_contact_time[:,contact_ids]
                v111_main_contact+=torch.stack((time111[:,0]>0,time111[:,2]>0),1).double().sum(0)
                v111_toe_contact+=torch.stack((time111[:,1]>0,time111[:,3]>0),1).double().sum(0)
                v111_quality_sum+=v111_q.double().sum(dim=0)
                v111_pair_sum+=d111['paired'].double().sum()
                v111_reward_sum+=d111['swing'].double().sum()
                v111_active_samples+=((v111_q>0.05).any(dim=1)).sum()
                v111_attempts+=d111['events'].double().sum(dim=0)
                v111_credit+=d111['side_credit'].double().sum(dim=0)
                v111_recovery+=d111['recovery_exempt'].double().sum()
                v111_strafe+=d111['strafe_exempt'].double().sum()
                v111_receiver+=d111['receiver'].double().sum()
                v111_receiver_side+=d111['receiver_progress'].double().sum(dim=0)
                v111_receiver_load+=d111['receiver_load_delta'].double().sum(dim=0)
                v111_receiver_com+=d111['receiver_com_delta'].double().sum(dim=0)
                v111_receiver_completed+=d111['receiver_complete'].double().sum(dim=0)
                v111_max_streak=torch.maximum(v111_max_streak,d111['repeat_streak'].double().max())

            if hasattr(env,"_littlegreen_v66_start_event"):
                v66_start+=env._littlegreen_v66_start_event.double().sum(0)
                v66_replant+=env._littlegreen_v66_replant_event.double().sum(0)
                v66_handoff+=env._littlegreen_v66_handoff_event.double().sum(0)
                v66_release+=env._littlegreen_v66_release_event.double().sum(0)
                v66_readiness+=env._littlegreen_v66_support_readiness_delta.double().sum()
                v66_traj+=env._littlegreen_v66_trajectory_delta.double().sum()
                v66_takeover+=env._littlegreen_v66_takeover_delta.double().sum()
                stage_now=env._littlegreen_v66_stage
                for stage_i in range(3): v66_stage[stage_i]+=(stage_now==stage_i).sum()
                exp_now=env._littlegreen_v66_expected_swing
                v66_expected[0]+=(exp_now<0).sum(); v66_expected[1]+=(exp_now==0).sum(); v66_expected[2]+=(exp_now==1).sum()
                v66_pending+=env._littlegreen_v66_pending_release.double().sum()
                v66_level+=env._littlegreen_v66_milestone_level.double().sum()
                v66_takeover_best+=env._littlegreen_v66_takeover_best.double().sum()
            if force is not None: obs=obs.clone(); obs[:,:3]=force
            vfac=_viability(robot); viability_sum+=vfac.double().sum(); low_viability+=(vfac<0.45).sum()
            fs=sensor.data.net_forces_w_history[:,:,contact_ids,:]; vf=torch.clamp(fs[...,2],min=0).mean(1); c=vf>1.0
            mainc=torch.stack((c[:,0],c[:,2]),1); toec=torch.stack((c[:,1],c[:,3]),1); anyc=mainc|toec; count=anyc.sum(1); partial_now=(mainc^toec).any(1)
            pos=robot.data.body_pos_w[:,body_ids,:]; vel=robot.data.body_lin_vel_w[:,body_ids,:]
            centers=torch.stack((0.5*(pos[:,0,:]+pos[:,1,:]),0.5*(pos[:,2,:]+pos[:,3,:])),1); center_xy=centers[:,:,:2]; mid=centers.mean(1)
            rz=yaw_quat(robot.data.root_quat_w); cmd=env.command_manager.get_command(args_cli.eval_force_command_name); actual=quat_apply_inverse(rz,robot.data.root_lin_vel_w)[:,:3]; cmdnorm=torch.linalg.vector_norm(cmd[:,:2],dim=1); moving=cmdnorm>0.08
            yaw_mask=torch.abs(cmd[:,2])>0.08
            if yaw_mask.any():
                yaw_abs=torch.abs(cmd[:,2])
                yaw_signed=actual[:,2]*torch.sign(cmd[:,2])
                yaw_ratio=yaw_signed/torch.clamp(yaw_abs,min=0.08)
                yaw_active_n+=yaw_mask.sum(); yaw_abs_cmd+=(yaw_abs.double()*yaw_mask).sum(); yaw_signed_actual+=(yaw_signed.double()*yaw_mask).sum(); yaw_ratio_sum+=(yaw_ratio.double()*yaw_mask).sum(); yaw_correct+=((yaw_signed>0.0)&yaw_mask).sum(); yaw_strong_n+=((yaw_abs>=0.30)&yaw_mask).sum()
            if force is None and hasattr(command_term,"is_turn_focus_env"):
                turn_focus_samples+=command_term.is_turn_focus_env.sum()
            dir_b=cmd[:,:2]/torch.clamp(cmdnorm.unsqueeze(-1),min=1e-6); dir3=torch.cat((dir_b,torch.zeros_like(dir_b[:,:1])),1); dirw=quat_apply(rz,dir3)[:,:2]
            along=(actual[:,:2]*dir_b).sum(1); raw_vel_along+=(along.double()*moving).sum(); viable_vel_along+=(along.double()*moving*vfac).sum(); forward_low_viability+=((along>0)&moving&(vfac<0.45)).sum()
            comrel=quat_apply_inverse(rz,robot.data.root_com_pos_w-mid)
            zpos=pos[:,:,2]; main_clear=torch.stack((zpos[:,0]-zpos[:,2],zpos[:,2]-zpos[:,0]),1); toe_clear=torch.stack((zpos[:,1]-zpos[:,3],zpos[:,3]-zpos[:,1]),1)
            coh=torch.clamp((0.060-torch.abs(main_clear-toe_clear))/(0.060-0.020),0.0,1.0)
            air=sensor.data.current_air_time[:,contact_ids]; whole_air=torch.stack((torch.min(air[:,:2],1).values,torch.min(air[:,2:],1).values),1)
            foot_load=torch.stack((vf[:,0]+vf[:,1],vf[:,2]+vf[:,3]),1); loadfrac=foot_load/torch.clamp(foot_load.sum(1,keepdim=True),min=1e-6)
            leftrel=quat_apply_inverse(rz,centers[:,0,:]-mid); rightrel=quat_apply_inverse(rz,centers[:,1,:]-mid); footy=torch.stack((leftrel[:,1],rightrel[:,1]),1); comdist=torch.abs(comrel[:,1].unsqueeze(-1)-footy)
            # v6.6 support-baton COM/load diagnostics use the same event-relative
            # foot-closeness metric as the reward state: 0.5 is between the feet,
            # values toward 1.0 mean the COM has moved toward that support foot.
            com_xy=robot.data.root_com_pos_w[:,:2]; dl=torch.linalg.vector_norm(com_xy-center_xy[:,0],dim=1); dr=torch.linalg.vector_norm(com_xy-center_xy[:,1],dim=1); dsum=torch.clamp(dl+dr,min=1e-6); close=torch.stack((dr/dsum,dl/dsum),1)
            if hasattr(env,"_littlegreen_v66_stage"):
                stg=env._littlegreen_v66_stage; exp=env._littlegreen_v66_expected_swing; env_ids=torch.arange(nenv,device=dev)
                free_stance=torch.argmax(loadfrac,dim=1); baton_stance=1-torch.clamp(exp,min=0,max=1); cand_stance=torch.where(exp>=0,baton_stance,free_stance)
                ready_mask=(stg==0)&moving; v66_ready_load+=(loadfrac[env_ids,cand_stance].double()*ready_mask).sum(); v66_ready_com+=(close[env_ids,cand_stance].double()*ready_mask).sum(); v66_ready_n+=ready_mask.sum()
                sf66=torch.clamp(env._littlegreen_v66_swing_foot,min=0,max=1); take_mask=(stg==2)&moving; new_load=loadfrac[env_ids,sf66]; new_close=close[env_ids,sf66]; v66_takeover_load+=(new_load.double()*take_mask).sum(); v66_takeover_com+=(new_close.double()*take_mask).sum(); v66_takeover_n+=take_mask.sum(); v66_takeover_load50+=((new_load>=0.50)&take_mask).sum(); v66_takeover_com56+=((new_close>=0.56)&take_mask).sum()
            # v6.7 swing-commitment diagnostics.  These use the same ordered physical
            # feet as the reward: whichever foot carries less load is the candidate
            # swing foot; command velocity is measured relative to the stance foot.
            env_ids67=torch.arange(nenv,device=dev); swing67=torch.argmin(loadfrac,dim=1); stance67=1-swing67
            whole_clear67=torch.minimum(main_clear,toe_clear); clear67=torch.clamp(whole_clear67[env_ids67,swing67],min=0.0)
            fullplant67=mainc&toec; cand67=moving&(loadfrac[env_ids67,swing67]<=0.45)&(loadfrac[env_ids67,stance67]>=0.55)&fullplant67[env_ids67,stance67]
            v67_candidate+=cand67.sum(); v67_clear3+=((clear67>=0.003)&cand67).sum(); v67_clear10+=((clear67>=0.010)&cand67).sum(); v67_clear22+=((clear67>=0.022)&cand67).sum(); v67_clear40+=((clear67>=0.040)&cand67).sum(); v69_clear50+=((clear67>=0.050)&cand67).sum(); v69_clear60+=((clear67>=0.060)&cand67).sum()
            center_vel67=torch.stack((0.5*(vel[:,0,:2]+vel[:,1,:2]),0.5*(vel[:,2,:2]+vel[:,3,:2])),1)
            rel67=center_vel67[env_ids67,swing67]-center_vel67[env_ids67,stance67]; rel_cmd67=(rel67*dirw).sum(1)
            rel_gate67=cand67&(clear67>=0.003); v67_rel_foot_cmd_sum+=(rel_cmd67.double()*rel_gate67).sum(); v67_rel_foot_cmd_pos+=((rel_cmd67>0)&rel_gate67).sum(); v67_rel_foot_cmd_ge20+=((rel_cmd67>=0.20)&rel_gate67).sum(); v67_rel_foot_cmd_n+=rel_gate67.sum()
            stance_center67=center_xy[env_ids67,stance67]; stance_vel67=center_vel67[env_ids67,stance67]; to_stance67=stance_center67-com_xy; to_stance67=to_stance67/torch.clamp(torch.linalg.vector_norm(to_stance67,dim=1,keepdim=True),min=1e-6)
            base_rel67=robot.data.root_lin_vel_w[:,:2]-stance_vel67; com_toward67=(base_rel67*to_stance67).sum(1); stance_speed67=torch.linalg.vector_norm(stance_vel67,dim=1)
            ready67=moving&fullplant67[:,0]&fullplant67[:,1]&(loadfrac[env_ids67,stance67]>=0.62)&(loadfrac[env_ids67,swing67]<=0.38)&(close[env_ids67,stance67]>=0.58)
            v67_com_toward_stance_sum+=(com_toward67.double()*ready67).sum(); v67_com_toward_stance_n+=ready67.sum(); v67_stance_speed_sum+=(stance_speed67.double()*ready67).sum(); v67_ready_n+=ready67.sum()

            # Exact active stance_relative_com_drive_v67 gate, decomposed in the
            # current WORLD command frame.  This is diagnostic only.  A 3-D vector
            # toward the stance foot would introduce an undesirable downward target;
            # the relevant question is whether the existing planar bridge points
            # backward/forward as well as laterally.
            stance_share98=loadfrac[env_ids67,stance67]; swing_share98=loadfrac[env_ids67,swing67]; stance_close98=close[env_ids67,stance67]
            com_gate98=moving&fullplant67[:,0]&fullplant67[:,1]&(swing_share98<=0.50)&(stance_close98<0.68)
            latw98=torch.stack((-dirw[:,1],dirw[:,0]),dim=1)
            stance_vec98=stance_center67-com_xy
            stance_vec_norm98=stance_vec98/torch.clamp(torch.linalg.vector_norm(stance_vec98,dim=1,keepdim=True),min=1e-6)
            dir_cmd_dot98=(stance_vec_norm98*dirw).sum(1)
            stance_cmd_offset98=(stance_vec98*dirw).sum(1)
            stance_lat_offset98=(stance_vec98*latw98).sum(1)
            rel_cmd_vel98=(base_rel67*dirw).sum(1)
            lat_sign98=torch.sign(stance_lat_offset98); lat_sign98=torch.where(lat_sign98==0,torch.ones_like(lat_sign98),lat_sign98)
            rel_lat_toward98=(base_rel67*latw98).sum(1)*lat_sign98
            v98_com_gate_n+=com_gate98.sum(); v98_com_dir_cmd_dot+=(dir_cmd_dot98.double()*com_gate98).sum(); v98_com_backward_dir+=((dir_cmd_dot98<0)&com_gate98).sum(); v98_com_rel_cmd_vel+=(rel_cmd_vel98.double()*com_gate98).sum(); v98_com_lat_toward_vel+=(rel_lat_toward98.double()*com_gate98).sum(); v98_com_stance_cmd_offset+=(stance_cmd_offset98.double()*com_gate98).sum(); v98_com_stance_lat_abs+=(torch.abs(stance_lat_offset98).double()*com_gate98).sum()
            if hasattr(env, "_littlegreen_v99_com_lateral_gate"):
                gate99=env._littlegreen_v99_com_lateral_gate.bool(); lat_speed99=env._littlegreen_v99_com_lateral_toward_speed_mps; close99=env._littlegreen_v99_com_lateral_closeness; lat_off99=env._littlegreen_v99_com_lateral_offset_m; rel_xy99=env._littlegreen_v99_com_relative_world_xy_mps; rel_cmd99=(rel_xy99*dirw).sum(1)
                v99_com_gate_n+=gate99.sum(); v99_com_lat_toward_vel+=(lat_speed99.double()*gate99).sum(); v99_com_lat_closeness+=(close99.double()*gate99).sum(); v99_com_lat_offset_abs+=(lat_off99.abs().double()*gate99).sum(); v99_com_rel_cmd_vel+=(rel_cmd99.double()*gate99).sum()
            if hasattr(env,"_littlegreen_v67_ready_no_lift_timer"):
                rt67=env._littlegreen_v67_ready_no_lift_timer; v67_ready_timer_sum+=(rt67.double()*ready67).sum(); v67_ready_timer_ge018+=((rt67>=0.18)&ready67).sum()
            vz67=vel[:,:,2]; oppvz67=torch.stack((0.5*(vz67[:,2]+vz67[:,3]),0.5*(vz67[:,0]+vz67[:,1])),1); rise67=torch.minimum(torch.stack((vz67[:,0],vz67[:,2]),1)-oppvz67,torch.stack((vz67[:,1],vz67[:,3]),1)-oppvz67)[env_ids67,swing67]
            v67_coherent_rise_sum+=(rise67.double()*cand67).sum(); v67_coherent_rise_n+=cand67.sum()
            pre67=moving&(clear67<0.010); meaningful67=moving&(clear67>=0.018); v67_pre_swing_base_along+=(along.double()*pre67).sum(); v67_pre_swing_n+=pre67.sum(); v67_meaningful_swing_base_along+=(along.double()*meaningful67).sum(); v67_meaningful_swing_n+=meaningful67.sum()

            # v6.8 swing-frontier diagnostics are exported directly by the reward state.
            if hasattr(env,"_littlegreen_v68_stride_horizon") and hasattr(env,"_littlegreen_v66_stage"):
                swing68=(env._littlegreen_v66_stage==1)&moving
                h68=env._littlegreen_v68_stride_horizon; ts68=env._littlegreen_v68_target_swing_speed; bt68=env._littlegreen_v68_backtrack; ff68=env._littlegreen_v68_frontier_factor
                v68_swing_n+=swing68.sum(); v68_stride_horizon_sum+=(h68.double()*swing68).sum(); v68_target_speed_sum+=(ts68.double()*swing68).sum(); v68_backtrack_sum+=(bt68.double()*swing68).sum(); v68_backtrack_gt6+=((bt68>0.006)&swing68).sum(); v68_backtrack_gt14+=((bt68>0.014)&swing68).sum(); v68_frontier_factor_sum+=(ff68.double()*swing68).sum()
                rep68=getattr(env,"_littlegreen_v66_replant_event",None)
                if isinstance(rep68,torch.Tensor):
                    repmask68=torch.any(rep68,dim=1)
                    if repmask68.any():
                        ret68=env._littlegreen_v68_replant_retention; td68=env._littlegreen_v68_touchdown_progress; pk68=env._littlegreen_v68_touchdown_peak_progress; th68=env._littlegreen_v68_touchdown_stride_horizon
                        v68_replant_n+=repmask68.sum(); v68_retention_sum+=(ret68.double()*repmask68).sum(); v68_touchdown_progress_sum+=(td68.double()*repmask68).sum(); v68_peak_progress_sum+=(pk68.double()*repmask68).sum(); v68_touchdown_horizon_sum+=(th68.double()*repmask68).sum()

            # v6.4 upstream bridge diagnostics mirror whole_foot_swing_initiation_v64.
            opposite_share=torch.flip(loadfrac,dims=[1]); whole_clear=torch.minimum(main_clear,toe_clear)
            vz=vel[:,:,2]; opp_vz=torch.stack((0.5*(vz[:,2]+vz[:,3]),0.5*(vz[:,0]+vz[:,1])),1); main_rel_vz=torch.stack((vz[:,0],vz[:,2]),1)-opp_vz; toe_rel_vz=torch.stack((vz[:,1],vz[:,3]),1)-opp_vz; coherent_rise=torch.minimum(main_rel_vz,toe_rel_vz)
            unload_score=torch.clamp((0.48-loadfrac)/(0.48-0.15),0.0,1.0); support_score=torch.clamp((opposite_share-0.48)/(0.68-0.48),0.0,1.0); clear_score=torch.clamp(whole_clear/0.015,0.0,1.0); rise_score=torch.clamp(coherent_rise/0.12,0.0,1.0); lift_signal=torch.where(anyc,0.50*rise_score,clear_score)
            bridge_eligible=(anyc|(whole_air<=0.12))&(opposite_share>=0.48)&moving.unsqueeze(-1); bridge_score=lift_signal*(0.25+0.75*unload_score)*support_score*bridge_eligible.float()*vfac.unsqueeze(-1)
            candidate=(opposite_share>=0.48)&moving.unsqueeze(-1)&(vfac.unsqueeze(-1)>=0.45); init_candidate+=candidate.double().sum(0); init_score_sum+=(bridge_score.double()*candidate).sum(0); init_active+=(bridge_score>0).double().sum(0); init_clear_2+=((whole_clear>=0.002)&candidate).double().sum(0); init_clear_5+=((whole_clear>=0.005)&candidate).double().sum(0); init_clear_10+=((whole_clear>=0.010)&candidate).double().sum(0); init_clear_15+=((whole_clear>=0.015)&candidate).double().sum(0); init_rising+=((coherent_rise>0.02)&candidate).double().sum(0); init_unloaded+=((loadfrac<=0.45)&candidate).double().sum(0)

            if prev_any is None: prev_any=anyc.clone(); prev_center=center_xy.clone()
            # v6.9 support-translation diagnostics.  Raw event values are sampled only
            # on the one-step handoff event; takeover retreat is sampled continuously.
            if hasattr(env,"_littlegreen_v69_post_touchdown_retreat_m"):
                active69=getattr(env,"_littlegreen_v69_touchdown_active_mask",torch.zeros(nenv,device=dev,dtype=torch.bool)) & (getattr(env,"_littlegreen_v66_stage",torch.zeros(nenv,device=dev,dtype=torch.long))==2) & moving
                ret69=env._littlegreen_v69_post_touchdown_retreat_m
                v69_takeover_n+=active69.sum(); v69_retreat_sum+=(ret69.double()*active69).sum(); v69_retreat_gt5+=((ret69>0.005)&active69).sum(); v69_retreat_gt12+=((ret69>0.012)&active69).sum()
                if hasattr(env,"_littlegreen_v66_handoff_event"):
                    he69=env._littlegreen_v66_handoff_event.any(1)
                    if he69.any():
                        v69_handoff_n+=he69.sum()
                        v69_handoff_retention_sum+=(env._littlegreen_v69_last_handoff_retention.double()*he69).sum()
                        v69_support_advance_sum+=(env._littlegreen_v69_last_support_advance_m.double()*he69).sum()
                        v69_support_horizon_sum+=(env._littlegreen_v69_last_support_horizon_m.double()*he69).sum()
                        v69_support_score_sum+=(env._littlegreen_v69_support_progress_score_value.double()*he69).sum()
                        cv69=env._littlegreen_v69_last_com_progress_valid & he69
                        v69_com_n+=cv69.sum(); v69_com_advance_sum+=(env._littlegreen_v69_last_com_advance_m.double()*cv69).sum(); v69_com_horizon_sum+=(env._littlegreen_v69_last_com_horizon_m.double()*cv69).sum(); v69_com_score_sum+=(env._littlegreen_v69_com_progress_score_value.double()*cv69).sum()

            # Reward-native v9.8 dense per-handoff diagnostics.
            if hasattr(env,"_littlegreen_v98_pass_event"):
                pe=env._littlegreen_v98_pass_event
                if pe.any():
                    v98_pass_n+=pe.sum()
                    v98_pass_margin+=(env._littlegreen_v98_pass_margin_m.double()*pe).sum()
                    v98_pass_quality+=(env._littlegreen_v98_pass_current_quality.double()*pe).sum()
                    v98_pass_current_score+=(env._littlegreen_v98_pass_current_score.double()*pe).sum()
                    pair_mask=pe & env._littlegreen_v98_pass_pair_valid.bool()
                    v98_pass_pair_valid+=pair_mask.double().sum()
                    v98_pass_pair_quality+=(env._littlegreen_v98_pass_pair_quality.double()*pair_mask).sum()
                    v98_pass_pair_score+=(env._littlegreen_v98_pass_pair_score.double()*pair_mask).sum()
                    v98_pass_dense_score+=(env._littlegreen_v98_pass_dense_score.double()*pe).sum()
                    pf=env._littlegreen_v98_pass_foot
                    for fi in range(2):
                        fm=pe&(pf==fi)
                        v98_pass_foot_n[fi]+=fm.sum()
                        v98_pass_foot_margin[fi]+=(env._littlegreen_v98_pass_margin_m.double()*fm).sum()
                        v98_pass_foot_quality[fi]+=(env._littlegreen_v98_pass_current_quality.double()*fm).sum()
                        v98_pass_foot_score[fi]+=(env._littlegreen_v98_pass_current_score.double()*fm).sum()
            if hasattr(env,"_littlegreen_v98_rate_event"):
                re=env._littlegreen_v98_rate_event
                if re.any():
                    v98_rate_n+=re.sum()
                    v98_rate_alt+=(env._littlegreen_v98_rate_alternating.double()*re).sum()
                    rf=env._littlegreen_v98_rate_foot
                    for fi in range(2):
                        rm=re&(rf==fi)
                        v98_rate_foot_n[fi]+=rm.sum()
                        v98_rate_advance[fi]+=(env._littlegreen_v98_rate_advance_m.double()*rm).sum()
                        v98_rate_elapsed[fi]+=(env._littlegreen_v98_rate_elapsed_s.double()*rm).sum()
                        v98_rate_mps[fi]+=(env._littlegreen_v98_rate_mps.double()*rm).sum()
                        v98_rate_ratio[fi]+=(env._littlegreen_v98_rate_ratio.double()*rm).sum()
                        v98_rate_score[fi]+=(env._littlegreen_v98_rate_score.double()*rm).sum()
                        v98_rate_command[fi]+=(env._littlegreen_v98_rate_command_mps.double()*rm).sum()
            if hasattr(env,"_littlegreen_v101_rate_event"):
                re101=env._littlegreen_v101_rate_event.bool()
                if re101.any():
                    v101_rate_n+=re101.sum()
                    v101_rate_current_quality+=(env._littlegreen_v101_rate_current_quality.double()*re101).sum()
                    v101_rate_current_score+=(env._littlegreen_v101_rate_current_score.double()*re101).sum()
                    pair101=re101 & env._littlegreen_v101_rate_pair_valid.bool()
                    v101_rate_pair_valid+=pair101.sum()
                    v101_rate_pair_quality+=(env._littlegreen_v101_rate_pair_quality.double()*pair101).sum()
                    v101_rate_pair_score+=(env._littlegreen_v101_rate_pair_score.double()*pair101).sum()
                    v101_rate_dense_score+=(env._littlegreen_v101_rate_dense_score.double()*re101).sum()
                    rf101=env._littlegreen_v101_rate_foot
                    for fi in range(2):
                        fm101=re101 & (rf101==fi)
                        v101_rate_foot_n[fi]+=fm101.sum()
                        v101_rate_foot_pair_valid[fi]+=(fm101 & env._littlegreen_v101_rate_pair_valid.bool()).sum()
                        v101_rate_foot_dense_score[fi]+=(env._littlegreen_v101_rate_dense_score.double()*fm101).sum()
            if hasattr(env,"_littlegreen_v98_epoch_changed"):
                v98_epoch_resets+=env._littlegreen_v98_epoch_changed.sum()

            reset2=dones.unsqueeze(-1)
            active=torch.where(reset2,torch.zeros_like(active),active); qualified=torch.where(reset2,torch.zeros_like(qualified),qualified); landing=torch.where(reset2,torch.zeros_like(landing),landing); main_seen=torch.where(reset2,torch.zeros_like(main_seen),main_seen); toe_seen=torch.where(reset2,torch.zeros_like(toe_seen),toe_seen); confirm=torch.where(reset2,torch.zeros_like(confirm),confirm); takeover=torch.where(reset2,torch.zeros_like(takeover),takeover); handoff_done=torch.where(reset2,torch.zeros_like(handoff_done),handoff_done); release_watch_active=torch.where(dones,torch.zeros_like(release_watch_active),release_watch_active); release_watch_new_foot=torch.where(dones,torch.full_like(release_watch_new_foot,-1),release_watch_new_foot); release_watch_timer=torch.where(dones,torch.zeros_like(release_watch_timer),release_watch_timer); last_foot=torch.where(dones,torch.full_like(last_foot,-1),last_foot)
            raw_lift=prev_any&(~anyc)&(~active)&moving.unsqueeze(-1)&(~reset2); raw_release+=raw_lift.double().sum(0); low_v_release+=(raw_lift&(vfac.unsqueeze(-1)<0.45)).double().sum(0)
            q_event=robot.data.joint_pos[:,joint_ids]
            for foot_i in range(2):
                mask=raw_lift[:,foot_i]
                if mask.any(): liftoff_pose_sum[foot_i]+=q_event[mask][:,hip_knee_idx[foot_i]].double().sum(0); liftoff_pose_count[foot_i]+=mask.sum()

            # Track hip/knee/ankle kinematics across the actual v6.6 swing baton.
            start_evt=getattr(env,"_littlegreen_v66_start_event",torch.zeros((nenv,2),device=dev,dtype=torch.bool)) & moving.unsqueeze(-1) & (~reset2)
            replant_evt=getattr(env,"_littlegreen_v66_replant_event",torch.zeros((nenv,2),device=dev,dtype=torch.bool)) & (~reset2)
            handoff_evt=getattr(env,"_littlegreen_v66_handoff_event",torch.zeros((nenv,2),device=dev,dtype=torch.bool)) & (~reset2)
            for foot_i in range(2):
                vals=q_event[:,swing_joint_idx[foot_i]]
                sm=start_evt[:,foot_i]
                if sm.any():
                    swing_start_pose_sum[foot_i]+=vals[sm].double().sum(0); swing_start_n[foot_i]+=sm.sum(); swing_diag_start[sm,foot_i]=vals[sm]; swing_diag_min[sm,foot_i]=vals[sm]; swing_diag_max[sm,foot_i]=vals[sm]; swing_diag_peak_pose[sm,foot_i]=vals[sm]; swing_diag_peak_dev[sm,foot_i]=0.0; swing_diag_active[sm,foot_i]=True; swing_diag_replanted[sm,foot_i]=False
                am=swing_diag_active[:,foot_i]&(~swing_diag_replanted[:,foot_i])
                if am.any():
                    swing_diag_min[am,foot_i]=torch.minimum(swing_diag_min[am,foot_i],vals[am]); swing_diag_max[am,foot_i]=torch.maximum(swing_diag_max[am,foot_i],vals[am]); devabs=torch.abs(vals[am]-swing_diag_start[am,foot_i]); better=devabs>swing_diag_peak_dev[am,foot_i]; swing_diag_peak_pose[am,foot_i]=torch.where(better,vals[am],swing_diag_peak_pose[am,foot_i]); swing_diag_peak_dev[am,foot_i]=torch.maximum(swing_diag_peak_dev[am,foot_i],devabs)
                rm=replant_evt[:,foot_i]&swing_diag_active[:,foot_i]&(~swing_diag_replanted[:,foot_i])
                if rm.any():
                    swing_replant_pose_sum[foot_i]+=vals[rm].double().sum(0); swing_replant_n[foot_i]+=rm.sum(); swing_peak_pose_sum[foot_i]+=swing_diag_peak_pose[rm,foot_i].double().sum(0); swing_excursion_sum[foot_i]+=(swing_diag_max[rm,foot_i]-swing_diag_min[rm,foot_i]).double().sum(0); swing_diag_n[foot_i]+=rm.sum(); swing_diag_replanted[rm,foot_i]=True
                hm=handoff_evt[:,foot_i]&swing_diag_active[:,foot_i]
                if hm.any():
                    swing_handoff_pose_sum[foot_i]+=vals[hm].double().sum(0); swing_handoff_n[foot_i]+=hm.sum(); swing_diag_active[hm,foot_i]=False; swing_diag_replanted[hm,foot_i]=False
            if dones.any():
                swing_diag_active[dones]=False; swing_diag_replanted[dones]=False

            lift=raw_lift&(vfac.unsqueeze(-1)>=0.45)
            liftoff_xy=torch.where(lift.unsqueeze(-1),prev_center,liftoff_xy); liftoff_dir=torch.where(lift.unsqueeze(-1),dirw[:,None,:].expand(-1,2,-1),liftoff_dir); liftoff_cmd=torch.where(lift,cmdnorm.unsqueeze(-1).expand(-1,2),liftoff_cmd); active|=lift; qualified=torch.where(lift,torch.zeros_like(qualified),qualified); swing_min_v=torch.where(lift,vfac.unsqueeze(-1).expand(-1,2),swing_min_v); landing=torch.where(lift,torch.zeros_like(landing),landing); main_seen=torch.where(lift,torch.zeros_like(main_seen),main_seen); toe_seen=torch.where(lift,torch.zeros_like(toe_seen),toe_seen); landing_timer=torch.where(lift,torch.zeros_like(landing_timer),landing_timer)
            airborne=active&(~anyc)&moving.unsqueeze(-1)&(~reset2); swing_min_v=torch.where(airborne,torch.minimum(swing_min_v,vfac.unsqueeze(-1)),swing_min_v); air_quality=torch.where(airborne,coh,air_quality)
            qnow=airborne&(whole_air>=0.09)&(main_clear>=0.010)&(toe_clear>=0.010)&(swing_min_v>=0.45); newq=qnow&(~qualified); qual+=newq.double().sum(0); qualified|=qnow
            first=active&qualified&anyc&(~landing)&moving.unsqueeze(-1)&(~reset2); landing|=first; landing_timer=torch.where(first,torch.zeros_like(landing_timer),landing_timer); landing_timer=torch.where(active&landing,landing_timer+dt,landing_timer); main_seen|=active&landing&mainc; toe_seen|=active&landing&toec
            complete=active&qualified&landing&main_seen&toe_seen&anyc&(landing_timer<=0.30)&moving.unsqueeze(-1)&(~reset2); land+=complete.double().sum(0)
            progress=((center_xy-liftoff_xy)*liftoff_dir).sum(-1); vr=torch.clamp(((robot.data.root_lin_vel_w[:,None,:2]*liftoff_dir).sum(-1))/torch.clamp(liftoff_cmd,min=1e-3),0,1)
            progress_sum+=(progress.double()*complete).sum(0); velratio_sum+=(vr.double()*complete).sum(0); coherence_sum+=(air_quality.double()*complete).sum(0); landing_load_sum+=(loadfrac.double()*complete).sum(0); landing_count+=complete.double().sum(0)
            meaningful=complete&(progress>=0.008)&(air_quality>0); one=(meaningful.int().sum(1)==1); newc=meaningful&one.unsqueeze(-1); confirm=torch.where(one.unsqueeze(-1),newc,confirm); confirm_timer=torch.where(one.unsqueeze(-1),torch.zeros_like(confirm_timer),confirm_timer); confirm_min_v=torch.where(newc,vfac.unsqueeze(-1),confirm_min_v)
            timeout=active&landing&(landing_timer>0.30); unq=active&anyc&(~qualified); finish=complete|timeout|unq; active=torch.where(finish,torch.zeros_like(active),active); qualified=torch.where(finish,torch.zeros_like(qualified),qualified); landing=torch.where(finish,torch.zeros_like(landing),landing); main_seen=torch.where(finish,torch.zeros_like(main_seen),main_seen); toe_seen=torch.where(finish,torch.zeros_like(toe_seen),toe_seen); landing_timer=torch.where(finish,torch.zeros_like(landing_timer),landing_timer)
            confirm_timer=torch.where(confirm,confirm_timer+dt,confirm_timer); confirm_min_v=torch.where(confirm,torch.minimum(confirm_min_v,vfac.unsqueeze(-1)),confirm_min_v); lost=confirm&(~anyc); reached=confirm&(confirm_timer>=0.16); cand=reached&anyc&(confirm_min_v>=0.55)&moving.unsqueeze(-1)&(~reset2); oneconf=(cand.int().sum(1)==1); conf=cand&oneconf.unsqueeze(-1)
            if conf.any():
                ids=torch.arange(2,device=dev).view(1,2).expand(nenv,-1); has=last_foot>=0; same=has.unsqueeze(-1)&(ids==last_foot.unsqueeze(-1)); opp=has.unsqueeze(-1)&(~same)
                firsts+=((conf&(~has.unsqueeze(-1))).sum()); alternating+=(conf&opp).sum(); repeats+=(conf&same).sum(); confirmed_count+=conf.double().sum(0); newfoot=torch.argmax(conf.long(),1); bridge_conf=conf&(~same); last_foot=torch.where(oneconf,newfoot,last_foot)
                takeover=torch.where(oneconf.unsqueeze(-1),bridge_conf,takeover); takeover_timer=torch.where(oneconf.unsqueeze(-1),torch.zeros_like(takeover_timer),takeover_timer); takeover_initial_dist=torch.where(bridge_conf,comdist,takeover_initial_dist); handoff_done=torch.where(oneconf.unsqueeze(-1),torch.zeros_like(handoff_done),handoff_done); release_watch_active=torch.where(oneconf,torch.zeros_like(release_watch_active),release_watch_active); release_watch_new_foot=torch.where(oneconf,torch.full_like(release_watch_new_foot,-1),release_watch_new_foot); release_watch_timer=torch.where(oneconf,torch.zeros_like(release_watch_timer),release_watch_timer)
            cdone=reached|lost|(~moving.unsqueeze(-1))|reset2; confirm=torch.where(cdone,torch.zeros_like(confirm),confirm); confirm_timer=torch.where(cdone,torch.zeros_like(confirm_timer),confirm_timer)
            takeover_timer=torch.where(takeover,takeover_timer+dt,takeover_timer); takeover&=(takeover_timer<=0.50)&moving.unsqueeze(-1)&(~reset2)
            load_score=torch.clamp((loadfrac-0.30)/0.25,0,1); com_imp=torch.clamp((takeover_initial_dist-comdist)/0.050,0,1); tscore=(0.70*load_score+0.30*com_imp)*takeover.float()*anyc.float()*vfac.unsqueeze(-1); takeover_score_sum+=tscore.double().sum(); takeover_n+=takeover.sum(); takeover_gt50+=((loadfrac>=0.50)&takeover).sum()
            handoff_candidate=takeover&(~handoff_done)&anyc&(loadfrac>=0.47)&(vfac.unsqueeze(-1)>=0.60); onehandoff=(handoff_candidate.int().sum(1)==1); handoff=handoff_candidate&onehandoff.unsqueeze(-1)
            handoff_count+=handoff.double().sum(0); handoff_done|=handoff; handoff_new=torch.argmax(handoff.long(),1); release_watch_active=torch.where(onehandoff,torch.ones_like(release_watch_active),release_watch_active); release_watch_new_foot=torch.where(onehandoff,handoff_new,release_watch_new_foot); release_watch_timer=torch.where(onehandoff,torch.zeros_like(release_watch_timer),release_watch_timer)
            release_watch_timer=torch.where(release_watch_active,release_watch_timer+dt,release_watch_timer); valid_watch=release_watch_active&(release_watch_timer<=0.45)&moving&(~dones)&(vfac>=0.45)&(release_watch_new_foot>=0); ids_env=torch.arange(nenv,device=dev); newidx=torch.clamp(release_watch_new_foot,0,1); oldidx=1-newidx; release_event=valid_watch&anyc[ids_env,newidx]&(~anyc[ids_env,oldidx])
            if release_event.any(): opposite_release_count+=torch.bincount(oldidx[release_event],minlength=2).double()
            release_watch_active=release_watch_active&(~release_event)&(release_watch_timer<=0.45)
            prev_any=anyc.clone(); prev_center=center_xy.clone()

            samples+=n; moving_samples+=moving.sum(); both+=(count==2).sum(); single+=(count==1).sum(); none+=(count==0).sum(); partial+=partial_now.sum(); cmd_sum+=cmd.double().sum(0); actual_sum+=actual.double().sum(0); track_sq+=torch.square((actual-cmd).double()).sum(0); base_h+=robot.data.root_com_pos_w[:,2].double().sum(); com_lat+=comrel[:,1].abs().double().sum(); com_fwd+=comrel[:,0].double().sum()
            stance=(foot_load.sum(1)>1.0)&moving; footspeed=0.5*torch.minimum(torch.linalg.vector_norm(vel[:,0,:2],dim=1)+torch.linalg.vector_norm(vel[:,1,:2],dim=1),torch.linalg.vector_norm(vel[:,2,:2],dim=1)+torch.linalg.vector_norm(vel[:,3,:2],dim=1)); slide_sum+=(footspeed.double()*stance).sum(); loaded_count+=stance.sum()
            tq=robot.data.applied_torque[:,joint_ids]; q=robot.data.joint_pos[:,joint_ids]; torque_abs+=tq.abs().double().sum(0); torque_hard+=(tq.abs()>=0.95*torque_limit).double().sum(0); q_sum+=q.double().sum(0); raw=action_term.raw_actions; bounded=action_term.bounded_actions; raw_excess+=(raw.abs()>1.0).double().sum(0); bound_sat+=(bounded.abs()>=0.999).double().sum(0); joint_samples+=n
            if args_cli.progress_interval>0 and ((step+1)%args_cli.progress_interval==0 or step+1==args_cli.steps): print(f"[ANALYZE] collected {step+1}/{args_cli.steps}",flush=True)

        # zero-command probe
        zero=torch.zeros((nenv,3),device=dev,dtype=obs.dtype); stand_n=z(); stand_both=z(); stand_full_plant=z(); stand_joint_vel=z(); stand_h=z(); stand_lat=z(); stand_imb=z(); stand_done=z(); stand_torque=torch.zeros(len(joint_names),device=dev,dtype=torch.float64)
        # Sim->real handoff-pose extraction.  The deployment handoff pose is NOT
        # q_default: it is the robust median pose actually occupied by the trained
        # policy late in a zero-command probe, restricted to viable, bilateral
        # main+toe full-plant samples.  This preserves q_default as the observation/
        # action contract while giving Track 2 a smoother state from which to hand
        # authority to the policy.
        handoff_q_chunks=[]; handoff_qd_chunks=[]; handoff_pg_chunks=[]; handoff_h_chunks=[]; handoff_prev_action_chunks=[]; handoff_prev_obs_diff_max=0.0
        handoff_late_start=max(0, int(math.floor(args_cli.standing_steps*(1.0-args_cli.handoff_pose_late_fraction))))
        for s in range(args_cli.standing_settle_steps+args_cli.standing_steps):
            apply_velocity_command_override(env,zero,command_name=args_cli.eval_force_command_name); obs=obs.clone(); obs[:,:3]=zero
            with torch.inference_mode(): act=policy(obs); obs,_,dones,_=vec_env.step(act)[:4]
            dones=dones.to(device=dev,dtype=torch.bool)
            if s<args_cli.standing_settle_steps: continue
            fs=sensor.data.net_forces_w_history[:,:,contact_ids,:]; v=torch.clamp(fs[...,2],min=0).mean(1); c=v>1.0; la=c[:,0]|c[:,1]; ra=c[:,2]|c[:,3]; ps=robot.data.body_pos_w[:,body_ids,:]; cs=torch.stack((0.5*(ps[:,0,:]+ps[:,1,:]),0.5*(ps[:,2,:]+ps[:,3,:])),1); mid=cs.mean(1); cr=quat_apply_inverse(yaw_quat(robot.data.root_quat_w),robot.data.root_com_pos_w-mid); lf=v[:,0]+v[:,1]; rf=v[:,2]+v[:,3]; im=torch.abs(lf-rf)/torch.clamp(lf+rf,min=1e-6)
            stand_n+=n; stand_both+=(la&ra).sum(); stand_full_plant+=(c[:,0]&c[:,1]&c[:,2]&c[:,3]).sum(); stand_h+=robot.data.root_com_pos_w[:,2].double().sum(); stand_lat+=cr[:,1].abs().double().sum(); stand_imb+=im.double().sum(); stand_joint_vel+=torch.sum(torch.square(robot.data.joint_vel[:,joint_ids]),dim=1).double().sum(); stand_done+=dones.sum(); stand_torque+=(robot.data.applied_torque[:,joint_ids].abs()>=0.95*torque_limit).double().sum(0)
            stand_index=s-args_cli.standing_settle_steps
            if stand_index>=handoff_late_start:
                stand_vfac=_viability(robot)
                full_plant=(c[:,0]&c[:,1]&c[:,2]&c[:,3])
                q_now=robot.data.joint_pos[:,joint_ids]
                qd_now=robot.data.joint_vel[:,joint_ids]
                # The 47-D actor contract uses mdp.bounded_last_action at obs[33:45].
                # After vec_env.step(act), action_term.bounded_actions is therefore the
                # exact bounded action that becomes the *previous-action observation*
                # paired with q_now/projected_gravity in the returned observation.
                prev_action_now=action_term.bounded_actions
                if obs.shape[-1] < 45:
                    raise RuntimeError(f"Expected >=45 policy observations for v10.3, got shape {tuple(obs.shape)}")
                prev_action_obs=obs[:,33:45]
                finite=(torch.isfinite(q_now).all(dim=1)&torch.isfinite(qd_now).all(dim=1)&
                        torch.isfinite(robot.data.projected_gravity_b).all(dim=1)&
                        torch.isfinite(prev_action_now).all(dim=1)&torch.isfinite(prev_action_obs).all(dim=1))
                handoff_mask=full_plant&(stand_vfac>=args_cli.handoff_pose_min_viability)&(~dones)&finite
                if handoff_mask.any():
                    handoff_q_chunks.append(q_now[handoff_mask].detach().cpu())
                    handoff_qd_chunks.append(qd_now[handoff_mask].detach().cpu())
                    handoff_pg_chunks.append(robot.data.projected_gravity_b[handoff_mask].detach().cpu())
                    handoff_h_chunks.append(robot.data.root_com_pos_w[handoff_mask,2].detach().cpu())
                    handoff_prev_action_chunks.append(prev_action_now[handoff_mask].detach().cpu())
                    handoff_prev_obs_diff_max=max(
                        handoff_prev_obs_diff_max,
                        float(torch.max(torch.abs(prev_action_obs[handoff_mask]-prev_action_now[handoff_mask])).item()),
                    )

        N=max(_f(samples),eps); M=max(_f(moving_samples),eps); J=max(joint_samples,eps); SN=max(_f(stand_n),eps); TN=max(_f(takeover_n),eps)

        # Build a robust deployment handoff pose from all selected env-step samples.
        # Medians are used intentionally: they are insensitive to occasional contact
        # transitions or reset outliers and do not alter the trained q_default.
        handoff_pose=None
        if handoff_q_chunks:
            hq=torch.cat(handoff_q_chunks,dim=0).to(torch.float64)
            hqd=torch.cat(handoff_qd_chunks,dim=0).to(torch.float64)
            hpg=torch.cat(handoff_pg_chunks,dim=0).to(torch.float64)
            hh=torch.cat(handoff_h_chunks,dim=0).to(torch.float64)
            hprev=torch.cat(handoff_prev_action_chunks,dim=0).to(torch.float64)
            hmed=torch.median(hq,dim=0).values
            hp05=torch.quantile(hq,0.05,dim=0)
            hp95=torch.quantile(hq,0.95,dim=0)
            hmad=torch.median(torch.abs(hq-hmed.unsqueeze(0)),dim=0).values
            hvel=torch.median(torch.abs(hqd),dim=0).values
            prev_med=torch.median(hprev,dim=0).values
            prev_p05=torch.quantile(hprev,0.05,dim=0)
            prev_p95=torch.quantile(hprev,0.95,dim=0)
            prev_mad=torch.median(torch.abs(hprev-prev_med.unsqueeze(0)),dim=0).values
            qdef=robot.data.default_joint_pos[0,joint_ids].detach().cpu().to(torch.float64)
            pgmed=torch.median(hpg,dim=0).values
            handoff_pose={
                "schema_version":2,
                "source":"analyze_policy_v11_1.py",
                "task":args_cli.task,
                "checkpoint":checkpoint,
                "command":[0.0,0.0,0.0],
                "selection":{
                    "standing_settle_steps":int(args_cli.standing_settle_steps),
                    "standing_steps":int(args_cli.standing_steps),
                    "late_fraction":float(args_cli.handoff_pose_late_fraction),
                    "late_start_post_settle_step":int(handoff_late_start),
                    "minimum_viability":float(args_cli.handoff_pose_min_viability),
                    "require_bilateral_main_toe_full_plant":True,
                    "exclude_done_samples":True,
                    "selected_samples":int(hq.shape[0]),
                },
                "joint_order":list(joint_names),
                "q_default_rad":[float(x) for x in qdef.tolist()],
                "median_joint_position_rad":[float(x) for x in hmed.tolist()],
                "median_delta_from_q_default_rad":[float(x) for x in (hmed-qdef).tolist()],
                "p05_joint_position_rad":[float(x) for x in hp05.tolist()],
                "p95_joint_position_rad":[float(x) for x in hp95.tolist()],
                "mad_joint_position_rad":[float(x) for x in hmad.tolist()],
                "median_abs_joint_velocity_rad_s":[float(x) for x in hvel.tolist()],
                "previous_action_observation_indices":[33,44],
                "previous_action_observation_semantics":"bounded_last_action(joint_pos): 12-D bounded normalized action from the immediately preceding policy step",
                "median_previous_bounded_action":[float(x) for x in prev_med.tolist()],
                "p05_previous_bounded_action":[float(x) for x in prev_p05.tolist()],
                "p95_previous_bounded_action":[float(x) for x in prev_p95.tolist()],
                "mad_previous_bounded_action":[float(x) for x in prev_mad.tolist()],
                "max_abs_previous_action_obs_vs_action_term":float(handoff_prev_obs_diff_max),
                "median_projected_gravity_b":[float(x) for x in pgmed.tolist()],
                "median_root_com_height_m":float(torch.median(hh).item()),
                "notes":[
                    "Use median_joint_position_rad as the guarded Track 2 pre-position/handoff pose, not as a replacement for q_default.",
                    "Seed Track 2 obs[33:45] with median_previous_bounded_action at policy handoff; do not convert it to radians.",
                    "q_default remains part of the protected observation/action contract.",
                    "Pose and previous-action seed are derived from the exact same late, viable, bilateral main+toe full-plant zero-command samples.",
                ],
            }
        if handoff_pose is not None:
            samples_ok=handoff_pose['selection']['selected_samples']>=100
            plant_ok=(_f(stand_full_plant)/SN)>=0.50
            handoff_pose['deployment_qualification']={
                'status':'DIAGNOSTIC_ONLY' if not (samples_ok and plant_ok) else 'CANDIDATE_REQUIRES_TRACK2_REVIEW',
                'min_samples':100,'min_full_plant_fraction':0.50,
                'observed_full_plant_fraction':_f(stand_full_plant)/SN,
                'note':'Never replace q_default or authorize physical motion automatically.'}
        summary={
            "release":"11.1.0","task":args_cli.task,"checkpoint":checkpoint,"forced_command":args_cli.eval_force_command,"policy_observation_count":47,"action_count":len(joint_names),"steps":args_cli.steps,"num_envs":nenv,"done_count":int(round(_f(falls))),
            "viability":{"mean_factor":_f(viability_sum)/N,"fraction_below_0p45":_f(low_viability)/N,"definition":"soft product: full by <=25deg tilt and >=0.400m COM height; zero by >=55deg or <=0.340m"},
            "velocity":{"mean_raw_along_command_m_s":_f(raw_vel_along)/M,"mean_viability_weighted_along_command_m_s":_f(viable_vel_along)/M,"positive_velocity_low_viability_fraction":_f(forward_low_viability)/M,"mean_command":[_f(x)/N for x in cmd_sum],"mean_actual":[_f(x)/N for x in actual_sum],"tracking_rmse":[(_f(track_sq[i])/N)**0.5 for i in range(3)]},
            "contact":{"both_feet_any_contact_fraction":_f(both)/N,"single_support_any_contact_fraction":_f(single)/N,"no_support_fraction":_f(none)/N,"partial_main_or_toe_contact_fraction":_f(partial)/N},
            "posture":{"mean_base_com_height_m":_f(base_h)/N,"mean_abs_com_lateral_offset_m":_f(com_lat)/N,"mean_com_forward_offset_m":_f(com_fwd)/N,"mean_loaded_foot_speed_proxy_m_s":_f(slide_sum)/max(_f(loaded_count),eps)},
            "counterfactual_planar_com_audit_v98":{
                "definition":"counterfactual diagnostic only: decomposes the retired v9.8 WORLD-XY COM->stance direction into command-parallel and command-orthogonal components; this geometry no longer receives reward",
                "reward_gate_samples":int(round(_f(v98_com_gate_n))),
                "mean_toward_stance_direction_dot_command":_f(v98_com_dir_cmd_dot)/max(_f(v98_com_gate_n),eps),
                "fraction_gate_direction_points_backward":_f(v98_com_backward_dir)/max(_f(v98_com_gate_n),eps),
                "mean_relative_com_velocity_along_command_m_s":_f(v98_com_rel_cmd_vel)/max(_f(v98_com_gate_n),eps),
                "mean_relative_com_lateral_velocity_toward_stance_m_s":_f(v98_com_lat_toward_vel)/max(_f(v98_com_gate_n),eps),
                "mean_stance_offset_along_command_m":_f(v98_com_stance_cmd_offset)/max(_f(v98_com_gate_n),eps),
                "mean_abs_stance_lateral_offset_m":_f(v98_com_stance_lat_abs)/max(_f(v98_com_gate_n),eps),
                "interpretation":"negative direction-dot-command or high backward fraction describes the v9.8 coupling that v9.9 intentionally removes"
            },
            "com_lateral_bridge_v99":{
                "definition":"active reward diagnostic: body-yaw lateral (+/-Y) COM velocity relative to the loaded stance foot, signed toward the stance side; command-parallel COM velocity is diagnostic only",
                "reward_gate_samples":int(round(_f(v99_com_gate_n))),
                "mean_relative_com_lateral_velocity_toward_stance_m_s":_f(v99_com_lat_toward_vel)/max(_f(v99_com_gate_n),eps),
                "mean_lateral_closeness":_f(v99_com_lat_closeness)/max(_f(v99_com_gate_n),eps),
                "mean_abs_lateral_stance_offset_m":_f(v99_com_lat_offset_abs)/max(_f(v99_com_gate_n),eps),
                "mean_relative_com_velocity_along_command_m_s":_f(v99_com_rel_cmd_vel)/max(_f(v99_com_gate_n),eps),
                "interpretation":"for forward/reverse commands the COM bridge shapes left/right unloading without directly paying sagittal motion; acceleration must be earned through dense WORLD-rate feedback"
            },
            "swing_initiation":{"definition":"legacy v6.4-style diagnostic retained for longitudinal comparison; v6.7 lineage clearance/foot-relative metrics are retained in swing_commitment_v67; v6.8 frontier metrics are in swing_frontier_v68","left":{"mean_bridge_score":_f(init_score_sum[0])/max(_f(init_candidate[0]),eps),"bridge_active_fraction":_f(init_active[0])/max(_f(init_candidate[0]),eps),"load_share_le_0p45_fraction":_f(init_unloaded[0])/max(_f(init_candidate[0]),eps),"coherent_rise_gt_0p02_mps_fraction":_f(init_rising[0])/max(_f(init_candidate[0]),eps),"whole_clearance_ge_2mm_fraction":_f(init_clear_2[0])/max(_f(init_candidate[0]),eps),"whole_clearance_ge_5mm_fraction":_f(init_clear_5[0])/max(_f(init_candidate[0]),eps),"whole_clearance_ge_10mm_fraction":_f(init_clear_10[0])/max(_f(init_candidate[0]),eps),"whole_clearance_ge_15mm_fraction":_f(init_clear_15[0])/max(_f(init_candidate[0]),eps),"mean_hip_pitch_at_raw_liftoff_rad":_f(liftoff_pose_sum[0,0])/max(_f(liftoff_pose_count[0]),eps),"mean_knee_pitch_at_raw_liftoff_rad":_f(liftoff_pose_sum[0,1])/max(_f(liftoff_pose_count[0]),eps)},"right":{"mean_bridge_score":_f(init_score_sum[1])/max(_f(init_candidate[1]),eps),"bridge_active_fraction":_f(init_active[1])/max(_f(init_candidate[1]),eps),"load_share_le_0p45_fraction":_f(init_unloaded[1])/max(_f(init_candidate[1]),eps),"coherent_rise_gt_0p02_mps_fraction":_f(init_rising[1])/max(_f(init_candidate[1]),eps),"whole_clearance_ge_2mm_fraction":_f(init_clear_2[1])/max(_f(init_candidate[1]),eps),"whole_clearance_ge_5mm_fraction":_f(init_clear_5[1])/max(_f(init_candidate[1]),eps),"whole_clearance_ge_10mm_fraction":_f(init_clear_10[1])/max(_f(init_candidate[1]),eps),"whole_clearance_ge_15mm_fraction":_f(init_clear_15[1])/max(_f(init_candidate[1]),eps),"mean_hip_pitch_at_raw_liftoff_rad":_f(liftoff_pose_sum[1,0])/max(_f(liftoff_pose_count[1]),eps),"mean_knee_pitch_at_raw_liftoff_rad":_f(liftoff_pose_sum[1,1])/max(_f(liftoff_pose_count[1]),eps)}},
            "swing_commitment_v67":{
                "definition":"candidate swing foot is lower-load foot; 3/10/22/40mm whole-foot ladder; directional velocity is swing-foot center velocity relative to stance-foot center; COM drive is base velocity relative to stance foot toward that foot",
                "candidate_samples":int(round(_f(v67_candidate))),
                "whole_clearance_ge_3mm_fraction":_f(v67_clear3)/max(_f(v67_candidate),eps),
                "whole_clearance_ge_10mm_fraction":_f(v67_clear10)/max(_f(v67_candidate),eps),
                "whole_clearance_ge_22mm_fraction":_f(v67_clear22)/max(_f(v67_candidate),eps),
                "whole_clearance_ge_40mm_fraction":_f(v67_clear40)/max(_f(v67_candidate),eps),
                "whole_clearance_ge_50mm_fraction":_f(v69_clear50)/max(_f(v67_candidate),eps),
                "whole_clearance_ge_60mm_fraction":_f(v69_clear60)/max(_f(v67_candidate),eps),
                "mean_stance_relative_swing_foot_command_velocity_m_s":_f(v67_rel_foot_cmd_sum)/max(_f(v67_rel_foot_cmd_n),eps),
                "positive_stance_relative_swing_foot_command_velocity_fraction":_f(v67_rel_foot_cmd_pos)/max(_f(v67_rel_foot_cmd_n),eps),
                "stance_relative_swing_foot_command_velocity_ge_0p20_fraction":_f(v67_rel_foot_cmd_ge20)/max(_f(v67_rel_foot_cmd_n),eps),
                "mean_com_velocity_toward_stance_relative_to_stance_foot_m_s":_f(v67_com_toward_stance_sum)/max(_f(v67_com_toward_stance_n),eps),
                "mean_stance_foot_xy_speed_during_ready_m_s":_f(v67_stance_speed_sum)/max(_f(v67_ready_n),eps),
                "mean_ready_no_lift_timer_s":_f(v67_ready_timer_sum)/max(_f(v67_ready_n),eps),
                "ready_no_lift_timer_ge_0p18_fraction":_f(v67_ready_timer_ge018)/max(_f(v67_ready_n),eps),
                "mean_coherent_swing_rise_m_s":_f(v67_coherent_rise_sum)/max(_f(v67_coherent_rise_n),eps),
                "mean_base_along_command_before_10mm_swing_m_s":_f(v67_pre_swing_base_along)/max(_f(v67_pre_swing_n),eps),
                "mean_base_along_command_at_ge_18mm_swing_m_s":_f(v67_meaningful_swing_base_along)/max(_f(v67_meaningful_swing_n),eps)
            },
            "swing_frontier_v68":{
                "definition":"v6.8: commanded planar speed softly scales useful stride horizon and target stance-relative swing speed; positive velocity fades behind the remembered directional frontier; touchdown records retained peak progress",
                "swing_stage_samples":int(round(_f(v68_swing_n))),
                "mean_soft_stride_horizon_m":_f(v68_stride_horizon_sum)/max(_f(v68_swing_n),eps),
                "mean_target_stance_relative_swing_speed_m_s":_f(v68_target_speed_sum)/max(_f(v68_swing_n),eps),
                "mean_backtrack_from_frontier_m":_f(v68_backtrack_sum)/max(_f(v68_swing_n),eps),
                "backtrack_gt_6mm_fraction":_f(v68_backtrack_gt6)/max(_f(v68_swing_n),eps),
                "backtrack_gt_14mm_fraction":_f(v68_backtrack_gt14)/max(_f(v68_swing_n),eps),
                "mean_frontier_velocity_credit_factor":_f(v68_frontier_factor_sum)/max(_f(v68_swing_n),eps),
                "loaded_replant_events_sampled":int(round(_f(v68_replant_n))),
                "mean_touchdown_progress_retention_ratio":_f(v68_retention_sum)/max(_f(v68_replant_n),eps),
                "mean_touchdown_directional_progress_m":_f(v68_touchdown_progress_sum)/max(_f(v68_replant_n),eps),
                "mean_peak_directional_progress_m":_f(v68_peak_progress_sum)/max(_f(v68_replant_n),eps),
                "mean_touchdown_soft_stride_horizon_m":_f(v68_touchdown_horizon_sum)/max(_f(v68_replant_n),eps)
            },
            "support_translation_v69":{
                "definition":"v6.9: support-foot retreat is tracked from touchdown through takeover; support advance is new-vs-old stance at successful handoff; COM advance is measured only between consecutive direction-consistent successful handoffs",
                "takeover_samples":int(round(_f(v69_takeover_n))),
                "mean_post_touchdown_retreat_m":_f(v69_retreat_sum)/max(_f(v69_takeover_n),eps),
                "post_touchdown_retreat_gt_5mm_fraction":_f(v69_retreat_gt5)/max(_f(v69_takeover_n),eps),
                "post_touchdown_retreat_gt_12mm_fraction":_f(v69_retreat_gt12)/max(_f(v69_takeover_n),eps),
                "handoff_events_sampled":int(round(_f(v69_handoff_n))),
                "mean_handoff_progress_retention_ratio":_f(v69_handoff_retention_sum)/max(_f(v69_handoff_n),eps),
                "mean_support_to_support_advance_m":_f(v69_support_advance_sum)/max(_f(v69_handoff_n),eps),
                "mean_support_progress_horizon_m":_f(v69_support_horizon_sum)/max(_f(v69_handoff_n),eps),
                "mean_support_progress_score":_f(v69_support_score_sum)/max(_f(v69_handoff_n),eps),
                "direction_consistent_com_cycles":int(round(_f(v69_com_n))),
                "mean_handoff_to_handoff_com_advance_m":_f(v69_com_advance_sum)/max(_f(v69_com_n),eps),
                "mean_com_progress_horizon_m":_f(v69_com_horizon_sum)/max(_f(v69_com_n),eps),
                "mean_com_progress_score":_f(v69_com_score_sum)/max(_f(v69_com_n),eps),
            },
            "locomotion_economics_v98":{
                "definition":"v9.8: v9.7 arithmetic re-clocked onto every validated v6.3 confirmed-step mechanical support-takeover exchange; passage and WORLD support-frontier rate are evaluated at physical gait frequency",
                "command_epoch_reset_samples":int(round(_f(v98_epoch_resets))),
                "event_density":{
                    "v63_confirmed_steps":int(round(_f(confirmed_count.sum()))),
                    "v63_mechanical_support_handoffs":int(round(_f(handoff_count.sum()))),
                    "v66_support_baton_handoffs":int(round(_f(v66_handoff.sum()))),
                    "dense_passage_events":int(round(_f(v98_pass_n))),
                    "dense_passage_to_v63_handoff_ratio":_f(v98_pass_n)/max(_f(handoff_count.sum()),eps),
                    "dense_world_rate_events":int(round(_f(v98_rate_n))),
                    "dense_world_rate_to_v63_handoff_ratio":_f(v98_rate_n)/max(_f(handoff_count.sum()),eps),
                    "v63_handoff_to_confirmed_step_ratio":_f(handoff_count.sum())/max(_f(confirmed_count.sum()),eps),
                },
                "dense_bilateral_passage":{
                    "events":int(round(_f(v98_pass_n))),
                    "mean_pass_margin_m":_f(v98_pass_margin)/max(_f(v98_pass_n),eps),
                    "mean_current_quality":_f(v98_pass_quality)/max(_f(v98_pass_n),eps),
                    "mean_current_signed_score":_f(v98_pass_current_score)/max(_f(v98_pass_n),eps),
                    "pair_valid_fraction":_f(v98_pass_pair_valid)/max(_f(v98_pass_n),eps),
                    "mean_pair_quality":_f(v98_pass_pair_quality)/max(_f(v98_pass_pair_valid),eps),
                    "mean_pair_signed_score":_f(v98_pass_pair_score)/max(_f(v98_pass_pair_valid),eps),
                    "mean_dense_reward_score_before_viability":_f(v98_pass_dense_score)/max(_f(v98_pass_n),eps),
                    "left":{"events":int(round(_f(v98_pass_foot_n[0]))),"mean_pass_margin_m":_f(v98_pass_foot_margin[0])/max(_f(v98_pass_foot_n[0]),eps),"mean_quality":_f(v98_pass_foot_quality[0])/max(_f(v98_pass_foot_n[0]),eps),"mean_current_signed_score":_f(v98_pass_foot_score[0])/max(_f(v98_pass_foot_n[0]),eps)},
                    "right":{"events":int(round(_f(v98_pass_foot_n[1]))),"mean_pass_margin_m":_f(v98_pass_foot_margin[1])/max(_f(v98_pass_foot_n[1]),eps),"mean_quality":_f(v98_pass_foot_quality[1])/max(_f(v98_pass_foot_n[1]),eps),"mean_current_signed_score":_f(v98_pass_foot_score[1])/max(_f(v98_pass_foot_n[1]),eps)},
                },
                "dense_step_support_world_rate":{
                    "events":int(round(_f(v98_rate_n))),"neutral_rate_ratio":0.20,"rate_softness_ratio":0.12,
                    "alternating_event_fraction":_f(v98_rate_alt)/max(_f(v98_rate_n),eps),
                    "left":{"events":int(round(_f(v98_rate_foot_n[0]))),"mean_world_advance_m":_f(v98_rate_advance[0])/max(_f(v98_rate_foot_n[0]),eps),"mean_exchange_time_s":_f(v98_rate_elapsed[0])/max(_f(v98_rate_foot_n[0]),eps),"mean_world_rate_m_s":_f(v98_rate_mps[0])/max(_f(v98_rate_foot_n[0]),eps),"mean_command_m_s":_f(v98_rate_command[0])/max(_f(v98_rate_foot_n[0]),eps),"mean_rate_ratio":_f(v98_rate_ratio[0])/max(_f(v98_rate_foot_n[0]),eps),"mean_signed_score":_f(v98_rate_score[0])/max(_f(v98_rate_foot_n[0]),eps)},
                    "right":{"events":int(round(_f(v98_rate_foot_n[1]))),"mean_world_advance_m":_f(v98_rate_advance[1])/max(_f(v98_rate_foot_n[1]),eps),"mean_exchange_time_s":_f(v98_rate_elapsed[1])/max(_f(v98_rate_foot_n[1]),eps),"mean_world_rate_m_s":_f(v98_rate_mps[1])/max(_f(v98_rate_foot_n[1]),eps),"mean_command_m_s":_f(v98_rate_command[1])/max(_f(v98_rate_foot_n[1]),eps),"mean_rate_ratio":_f(v98_rate_ratio[1])/max(_f(v98_rate_foot_n[1]),eps),"mean_signed_score":_f(v98_rate_score[1])/max(_f(v98_rate_foot_n[1]),eps)},
                },
            },
            "bilateral_world_rate_v101":{
                "definition":"v10.1 keeps v10.0 per-exchange WORLD-rate arithmetic and +2.00 weight, then couples consecutive opposite-leg rate quality with a 50% geometric-mean pair score",
                "events":int(round(_f(v101_rate_n))),
                "pair_valid_fraction":_f(v101_rate_pair_valid)/max(_f(v101_rate_n),eps),
                "mean_current_quality":_f(v101_rate_current_quality)/max(_f(v101_rate_n),eps),
                "mean_current_signed_score":_f(v101_rate_current_score)/max(_f(v101_rate_n),eps),
                "mean_pair_quality":_f(v101_rate_pair_quality)/max(_f(v101_rate_pair_valid),eps),
                "mean_pair_signed_score":_f(v101_rate_pair_score)/max(_f(v101_rate_pair_valid),eps),
                "mean_dense_pair_sensitive_score_before_viability":_f(v101_rate_dense_score)/max(_f(v101_rate_n),eps),
                "left":{"events":int(round(_f(v101_rate_foot_n[0]))),"pair_valid_fraction":_f(v101_rate_foot_pair_valid[0])/max(_f(v101_rate_foot_n[0]),eps),"mean_dense_score":_f(v101_rate_foot_dense_score[0])/max(_f(v101_rate_foot_n[0]),eps)},
                "right":{"events":int(round(_f(v101_rate_foot_n[1]))),"pair_valid_fraction":_f(v101_rate_foot_pair_valid[1])/max(_f(v101_rate_foot_n[1]),eps),"mean_dense_score":_f(v101_rate_foot_dense_score[1])/max(_f(v101_rate_foot_n[1]),eps)},
            },
            "turn_tracking_v1031":{
                "definition":"v10.3.1 training is forward-only: zero sampled lateral/yaw commands; yaw shaping retained for later skills, not exercised in this task",
                "yaw_active_samples":int(round(_f(yaw_active_n))),
                "mean_abs_yaw_command_rad_s":_f(yaw_abs_cmd)/max(_f(yaw_active_n),eps),
                "mean_actual_yaw_rate_in_command_direction_rad_s":_f(yaw_signed_actual)/max(_f(yaw_active_n),eps),
                "mean_signed_yaw_rate_ratio":_f(yaw_ratio_sum)/max(_f(yaw_active_n),eps),
                "correct_yaw_direction_fraction":_f(yaw_correct)/max(_f(yaw_active_n),eps),
                "strong_turn_fraction_of_yaw_active":_f(yaw_strong_n)/max(_f(yaw_active_n),eps),
                "random_turn_focus_sample_fraction":None if force is not None else _f(turn_focus_samples)/N,
                "configured_turn_focus_probability_nonstanding":0.0,
                 "configured_general_ranges":{"lin_vel_x_m_s":[0.12,0.30],"lin_vel_y_m_s":[0.0,0.0],"ang_vel_z_rad_s":[0.0,0.0]},
                 "configured_turn_focus_ranges":None,
            },
            "economics_v111":{
                "definition":"v11.1 experimental attenuated repeated-side exploration plus post-airborne receiving-foot load/COM acceptance; v10.3.1 original gait-completion/WORLD-rate intact",
                "weights":{"track_lin_vel_xy_acquisition":0.80,"support_readiness_progress":0.35,"stance_relative_com_drive":0.18,"finite_swing_initiation":1.90,"support_takeover_progress":1.85,"swing_foot_command_velocity":1.85,"balanced_swing_discovery":1.20,"receiving_foot_load_acceptance":0.85,"coherent_swing_rise_velocity":0.90,"whole_foot_air_time_exploration":0.45,"opposite_support_release":1.80,"completed_gait_step":1.85,"bilateral_passage":1.10,"commanded_world_rate":2.00,"yaw_acquisition":0.55},
                "world_rate_bilateral_mix":0.50,
                 "configured_general_ranges":{"lin_vel_x_m_s":[0.12,0.30],"lin_vel_y_m_s":[0.0,0.0],"ang_vel_z_rad_s":[0.0,0.0]},
            },
            "economics_v100":{
                "definition":"v9.9 mechanics preserved; only four weights change to protect whole-foot gait acquisition while making useful bilateral/world progression more important",
                "weights":{"opposite_support_release":1.80,"completed_gait_step":1.85,"bilateral_passage":1.10,"commanded_world_rate":2.00},
                "v99_weights":{"opposite_support_release":2.20,"completed_gait_step":2.75,"bilateral_passage":0.80,"commanded_world_rate":1.50}
            },
            "hip_pitch_contract_v1031":{
                "definition":"source-contract audit; left/right hip pitch have identical active q_default and residual authority; captured physical envelopes differ slightly but neither clips the +/-0.42 rad policy target window",
                "left":{"physical_lower_rad":-1.92207792722071,"physical_upper_rad":0.681087469821225,"physical_span_rad":2.603165397041935,"q_default_rad":-0.24,"residual_scale_rad":0.42,"policy_target_lower_rad":-0.66,"policy_target_upper_rad":0.18,"tau_median_s":0.07368140888732962,"peak_velocity_large_step_rad_s":0.9022010275417774},
                "right":{"physical_lower_rad":-1.99110706267556,"physical_upper_rad":0.546097160487288,"physical_span_rad":2.537204223162848,"q_default_rad":-0.24,"residual_scale_rad":0.42,"policy_target_lower_rad":-0.66,"policy_target_upper_rad":0.18,"tau_median_s":0.07369036518620001,"peak_velocity_large_step_rad_s":0.9010160816263557},
                "physical_span_difference_rad":0.065961173879087,"physical_span_difference_percent_of_mean":2.566,"urdf_nominal_limits_identical":True,"loaded_return_tau_scale_identical":True
            },
            "swing_joint_diagnostics_v98":{
                "definition":"diagnostic only; event-aligned hip-pitch/knee-pitch/ankle-pitch pose and excursion across the v6.6 swing baton; no joint-specific reward is active",
                "left":{
                    "swing_samples":int(round(_f(swing_diag_n[0]))),
                    "start_pose_rad":{"hip_pitch":_f(swing_start_pose_sum[0,0])/max(_f(swing_start_n[0]),eps),"knee_pitch":_f(swing_start_pose_sum[0,1])/max(_f(swing_start_n[0]),eps),"ankle_pitch":_f(swing_start_pose_sum[0,2])/max(_f(swing_start_n[0]),eps)},
                    "peak_deviation_pose_rad":{"hip_pitch":_f(swing_peak_pose_sum[0,0])/max(_f(swing_diag_n[0]),eps),"knee_pitch":_f(swing_peak_pose_sum[0,1])/max(_f(swing_diag_n[0]),eps),"ankle_pitch":_f(swing_peak_pose_sum[0,2])/max(_f(swing_diag_n[0]),eps)},
                    "mean_swing_excursion_rad":{"hip_pitch":_f(swing_excursion_sum[0,0])/max(_f(swing_diag_n[0]),eps),"knee_pitch":_f(swing_excursion_sum[0,1])/max(_f(swing_diag_n[0]),eps),"ankle_pitch":_f(swing_excursion_sum[0,2])/max(_f(swing_diag_n[0]),eps)},
                    "replant_pose_rad":{"hip_pitch":_f(swing_replant_pose_sum[0,0])/max(_f(swing_replant_n[0]),eps),"knee_pitch":_f(swing_replant_pose_sum[0,1])/max(_f(swing_replant_n[0]),eps),"ankle_pitch":_f(swing_replant_pose_sum[0,2])/max(_f(swing_replant_n[0]),eps)},
                    "handoff_pose_rad":{"hip_pitch":_f(swing_handoff_pose_sum[0,0])/max(_f(swing_handoff_n[0]),eps),"knee_pitch":_f(swing_handoff_pose_sum[0,1])/max(_f(swing_handoff_n[0]),eps),"ankle_pitch":_f(swing_handoff_pose_sum[0,2])/max(_f(swing_handoff_n[0]),eps)},
                },
                "right":{
                    "swing_samples":int(round(_f(swing_diag_n[1]))),
                    "start_pose_rad":{"hip_pitch":_f(swing_start_pose_sum[1,0])/max(_f(swing_start_n[1]),eps),"knee_pitch":_f(swing_start_pose_sum[1,1])/max(_f(swing_start_n[1]),eps),"ankle_pitch":_f(swing_start_pose_sum[1,2])/max(_f(swing_start_n[1]),eps)},
                    "peak_deviation_pose_rad":{"hip_pitch":_f(swing_peak_pose_sum[1,0])/max(_f(swing_diag_n[1]),eps),"knee_pitch":_f(swing_peak_pose_sum[1,1])/max(_f(swing_diag_n[1]),eps),"ankle_pitch":_f(swing_peak_pose_sum[1,2])/max(_f(swing_diag_n[1]),eps)},
                    "mean_swing_excursion_rad":{"hip_pitch":_f(swing_excursion_sum[1,0])/max(_f(swing_diag_n[1]),eps),"knee_pitch":_f(swing_excursion_sum[1,1])/max(_f(swing_diag_n[1]),eps),"ankle_pitch":_f(swing_excursion_sum[1,2])/max(_f(swing_diag_n[1]),eps)},
                    "replant_pose_rad":{"hip_pitch":_f(swing_replant_pose_sum[1,0])/max(_f(swing_replant_n[1]),eps),"knee_pitch":_f(swing_replant_pose_sum[1,1])/max(_f(swing_replant_n[1]),eps),"ankle_pitch":_f(swing_replant_pose_sum[1,2])/max(_f(swing_replant_n[1]),eps)},
                    "handoff_pose_rad":{"hip_pitch":_f(swing_handoff_pose_sum[1,0])/max(_f(swing_handoff_n[1]),eps),"knee_pitch":_f(swing_handoff_pose_sum[1,1])/max(_f(swing_handoff_n[1]),eps),"ankle_pitch":_f(swing_handoff_pose_sum[1,2])/max(_f(swing_handoff_n[1]),eps)},
                },
            },
            "support_baton_v66":{"definition":"event-driven support readiness -> whole-foot swing -> loaded replant -> load+COM takeover -> handoff -> old-support release; first side free, continuation side comes from handoff","swing_start_events_left":int(round(_f(v66_start[0]))),"swing_start_events_right":int(round(_f(v66_start[1]))),"loaded_replant_events_left":int(round(_f(v66_replant[0]))),"loaded_replant_events_right":int(round(_f(v66_replant[1]))),"support_handoff_events_left":int(round(_f(v66_handoff[0]))),"support_handoff_events_right":int(round(_f(v66_handoff[1]))),"old_support_release_events_left":int(round(_f(v66_release[0]))),"old_support_release_events_right":int(round(_f(v66_release[1]))),"handoff_to_old_support_release_ratio":(_f(v66_release.sum())/max(_f(v66_handoff.sum()),eps)),"mean_support_readiness_delta_per_moving_sample":_f(v66_readiness)/M,"mean_directional_trajectory_delta_per_moving_sample":_f(v66_traj)/M,"mean_takeover_delta_per_moving_sample":_f(v66_takeover)/M,"stage_fraction_ready":_f(v66_stage[0])/N,"stage_fraction_swing":_f(v66_stage[1])/N,"stage_fraction_takeover":_f(v66_stage[2])/N,"expected_swing_fraction_free":_f(v66_expected[0])/N,"expected_swing_fraction_left":_f(v66_expected[1])/N,"expected_swing_fraction_right":_f(v66_expected[2])/N,"pending_old_support_release_fraction":_f(v66_pending)/N,"mean_milestone_level":_f(v66_level)/N,"mean_takeover_best_score":_f(v66_takeover_best)/N,"readiness_mean_candidate_stance_load_share":_f(v66_ready_load)/max(_f(v66_ready_n),eps),"readiness_mean_candidate_stance_com_closeness":_f(v66_ready_com)/max(_f(v66_ready_n),eps),"takeover_mean_new_foot_load_share":_f(v66_takeover_load)/max(_f(v66_takeover_n),eps),"takeover_mean_new_foot_com_closeness":_f(v66_takeover_com)/max(_f(v66_takeover_n),eps),"takeover_new_foot_load_ge_0p50_fraction":_f(v66_takeover_load50)/max(_f(v66_takeover_n),eps),"takeover_new_foot_com_closeness_ge_0p56_fraction":_f(v66_takeover_com56)/max(_f(v66_takeover_n),eps)},
            "confirmed_step_sequence":{"raw_whole_foot_releases_left":int(round(_f(raw_release[0]))),"raw_whole_foot_releases_right":int(round(_f(raw_release[1]))),"low_viability_releases_left":int(round(_f(low_v_release[0]))),"low_viability_releases_right":int(round(_f(low_v_release[1]))),"viability_qualified_swings_left":int(round(_f(qual[0]))),"viability_qualified_swings_right":int(round(_f(qual[1]))),"landing_completions_left":int(round(_f(land[0]))),"landing_completions_right":int(round(_f(land[1]))),"confirmed_steps_left":int(round(_f(confirmed_count[0]))),"confirmed_steps_right":int(round(_f(confirmed_count[1]))),"first_confirmed_count":int(round(_f(firsts))),"alternating_confirmed_count":int(round(_f(alternating))),"repeated_same_foot_confirmed_count":int(round(_f(repeats))),"confirmation_duration_s":0.16,"confirmation_min_viability":0.55,"immediate_credit_fraction":0.30,"sequence_multipliers":{"first":1.0,"opposite":2.0,"repeat":0.1},"left":{"mean_landing_progress_m":_f(progress_sum[0])/max(_f(land[0]),eps),"mean_velocity_ratio":_f(velratio_sum[0])/max(_f(land[0]),eps),"mean_swing_coherence":_f(coherence_sum[0])/max(_f(land[0]),eps)},"right":{"mean_landing_progress_m":_f(progress_sum[1])/max(_f(land[1]),eps),"mean_velocity_ratio":_f(velratio_sum[1])/max(_f(land[1]),eps),"mean_swing_coherence":_f(coherence_sum[1])/max(_f(land[1]),eps)}},
            "support_takeover":{"starts_after_confirmed_step":True,"mean_viability_gated_score":_f(takeover_score_sum)/TN,"new_foot_load_ge_0p50_fraction":_f(takeover_gt50)/TN,"handoff_load_threshold":0.47,"handoff_events_left":int(round(_f(handoff_count[0]))),"handoff_events_right":int(round(_f(handoff_count[1]))),"opposite_release_events_left":int(round(_f(opposite_release_count[0]))),"opposite_release_events_right":int(round(_f(opposite_release_count[1])))},
            "standing_probe":{"steps":args_cli.standing_steps,"done_count":int(round(_f(stand_done))),"both_feet_any_contact_fraction":_f(stand_both)/SN,"bilateral_main_toe_full_plant_fraction":_f(stand_full_plant)/SN,"mean_actionable_joint_velocity_l2":_f(stand_joint_vel)/SN,"mean_base_com_height_m":_f(stand_h)/SN,"height_deficit_zero_penalty_m":0.445,"height_deficit_full_penalty_m":0.405,"mean_abs_com_lateral_offset_m":_f(stand_lat)/SN,"mean_left_right_vertical_load_imbalance_fraction":_f(stand_imb)/SN,"zero_command_handoff_pose_selected_samples":(handoff_pose["selection"]["selected_samples"] if handoff_pose is not None else 0),"zero_command_handoff_pose_exported":handoff_pose is not None},
        }
        rows=[]
        for j,name in enumerate(joint_names): rows.append({"index":j,"joint":name,"joint_position_mean_rad":_f(q_sum[j])/J,"torque_abs_mean_nm":_f(torque_abs[j])/J,"hard_torque_fraction":_f(torque_hard[j])/J,"standing_hard_torque_fraction":_f(stand_torque[j])/SN,"raw_action_excess_fraction":_f(raw_excess[j])/J,"bounded_saturation_fraction":_f(bound_sat[j])/J})
        summary["bilateral_acquisition_v111"] = {
            "mean_left_foot_quality": _f(v111_quality_sum[0])/N,
            "mean_right_foot_quality": _f(v111_quality_sum[1])/N,
            "mean_recent_bilateral_pair_quality": _f(v111_pair_sum)/N,
            "fraction_steps_with_quality_gt_0p05": _f(v111_active_samples)/N,
            "mean_unweighted_term": _f(v111_reward_sum)/N,
            "mean_weighted_term_before_dt": 1.20*_f(v111_reward_sum)/N,
            "mean_receiving_foot_load_acceptance_weighted_before_dt":0.85*_f(v111_receiver)/N,
            "distinct_qualified_attempts_left":int(round(_f(v111_attempts[0]))),
            "distinct_qualified_attempts_right":int(round(_f(v111_attempts[1]))),
            "mean_credit_multiplier_left":_f(v111_credit[0])/N,
            "mean_credit_multiplier_right":_f(v111_credit[1])/N,
            "recovery_attenuation_exception_fraction":_f(v111_recovery)/N,
            "strafe_attenuation_exception_fraction":_f(v111_strafe)/N,
            "receiving_load_increments_left":_f(v111_receiver_load[0]),
            "receiving_load_increments_right":_f(v111_receiver_load[1]),
            "receiving_com_closeness_increments_left":_f(v111_receiver_com[0]),
            "receiving_com_closeness_increments_right":_f(v111_receiver_com[1]),
            "receiving_reward_left":_f(v111_receiver_side[0]),
            "receiving_reward_right":_f(v111_receiver_side[1]),
            "receiving_completion_events_left":int(round(_f(v111_receiver_completed[0]))),
            "receiving_completion_events_right":int(round(_f(v111_receiver_completed[1]))),
            "max_consecutive_side_attempts":int(round(_f(v111_max_streak))),
            "whole_foot_airborne_fraction_left":_f(v111_whole_air_samples[0])/N,
            "whole_foot_airborne_fraction_right":_f(v111_whole_air_samples[1])/N,
            "main_contact_fraction_left":_f(v111_main_contact[0])/N,
            "main_contact_fraction_right":_f(v111_main_contact[1])/N,
            "toe_contact_fraction_left":_f(v111_toe_contact[0])/N,
            "toe_contact_fraction_right":_f(v111_toe_contact[1])/N,
            "partial_heel_or_toe_contact_fraction_left":_f(v111_partial_contact_samples[0])/N,
            "partial_heel_or_toe_contact_fraction_right":_f(v111_partial_contact_samples[1])/N,
            "pivot_credit_suppressed_samples_left":int(round(_f(v111_pivot_suppressed_samples[0]))),
            "pivot_credit_suppressed_samples_right":int(round(_f(v111_pivot_suppressed_samples[1]))),
            "non_airborne_swing_quality_leak_samples_left":int(round(_f(v111_ground_reward_leak_samples[0]))),
            "non_airborne_swing_quality_leak_samples_right":int(round(_f(v111_ground_reward_leak_samples[1]))),
            "gate_definition":"new bilateral discovery requires main AND toe airborne; receiving load pays only AFTER true airborne then full-foot replant",
            "definition": "v11.1 distinct qualified bouts; diminishing same-side credit with recovery/strafe exemptions; receiving-foot reward measured separately",
        }
        # Static configurations alone are insufficient evidence of runtime plant.
        # Audit PhysX articulation mass and effective active actuator gains.
        view_masses=robot.root_physx_view.get_masses()[0].detach().cpu()
        summary["physical_plant_v1031"] = {
            "runtime_mass_kg": float(view_masses.sum().item()),
            "expected_provisional_mass_kg": 3.852,
            "corrected_usd": str(env_cfg.scene.robot.spawn.usd_path),
            "implicit_actuator_profiles": {
                group: {"stiffness": float(env_cfg.scene.robot.actuators[group].stiffness),
                        "damping": float(env_cfg.scene.robot.actuators[group].damping),
                        "effort_limit_sim_nm": float(env_cfg.scene.robot.actuators[group].effort_limit_sim)}
                for group in ("legs", "ankles")
            },
            "policy_step_s": float(env.step_dt),
            "physics_step_s": float(env.physics_dt),
            "mass_ground_truth_validated": False,
        }
        if abs(summary["physical_plant_v1031"]["runtime_mass_kg"]-3.852)>0.03:
            raise RuntimeError("v10.3 runtime articulation mass deviates from provisional corrected audit")
        _write_json(out/"policy_analysis.json",summary)
        with (out/"joint_action_stats.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if handoff_pose is not None:
            _write_json(out/"zero_command_handoff_pose.json",handoff_pose)
            hp_rows=[]
            for j,name in enumerate(joint_names):
                hp_rows.append({
                    "index":j,
                    "joint":name,
                    "q_default_rad":handoff_pose["q_default_rad"][j],
                    "median_joint_position_rad":handoff_pose["median_joint_position_rad"][j],
                    "median_delta_from_q_default_rad":handoff_pose["median_delta_from_q_default_rad"][j],
                    "p05_joint_position_rad":handoff_pose["p05_joint_position_rad"][j],
                    "p95_joint_position_rad":handoff_pose["p95_joint_position_rad"][j],
                    "mad_joint_position_rad":handoff_pose["mad_joint_position_rad"][j],
                    "median_abs_joint_velocity_rad_s":handoff_pose["median_abs_joint_velocity_rad_s"][j],
                    "previous_action_observation_index":33+j,
                    "median_previous_bounded_action":handoff_pose["median_previous_bounded_action"][j],
                    "p05_previous_bounded_action":handoff_pose["p05_previous_bounded_action"][j],
                    "p95_previous_bounded_action":handoff_pose["p95_previous_bounded_action"][j],
                    "mad_previous_bounded_action":handoff_pose["mad_previous_bounded_action"][j],
                })
            with (out/"zero_command_handoff_pose.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(hp_rows[0].keys())); w.writeheader(); w.writerows(hp_rows)
            print(f"[ANALYZE] Handoff: {out/'zero_command_handoff_pose.json'} ({handoff_pose['selection']['selected_samples']} selected samples)",flush=True)
        else:
            print("[ANALYZE][WARN] No late viable bilateral-full-plant samples qualified for handoff-pose export.",flush=True)
        _write_json(status,{"status":"complete","task":args_cli.task,"checkpoint":checkpoint,"steps":args_cli.steps,"handoff_pose_exported":handoff_pose is not None}); print(f"[ANALYZE] JSON:   {out/'policy_analysis.json'}",flush=True); print(f"[ANALYZE] Joints: {out/'joint_action_stats.csv'}",flush=True)
    except Exception as exc:
        print(f"[ANALYZE][ERROR] {type(exc).__name__}: {exc}",flush=True); traceback.print_exc(); _write_json(status,{"status":"error","task":args_cli.task,"checkpoint":checkpoint,"error":repr(exc)}); raise
    finally:
        if vec_env is not None: vec_env.close()
        elif base_env is not None: base_env.close()

if __name__=="__main__":
    try: main()
    finally: simulation_app.close()
