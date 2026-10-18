"""v11.1 R3: event-aware swing discovery, bounded recovery and receiving-foot preparation.

Neither term changes the actuator/observation/phase contract or schedules a side.
The helper is evaluated once per policy step even though three reward terms call it.
The state is reset on episode restart and on velocity-command epoch change.
"""
from __future__ import annotations
import math
import torch
from isaaclab.managers import SceneEntityCfg
from .rewards import _v67_whole_foot_kinematics


def repeat_multiplier(streak: torch.Tensor, decay: float, floor: float) -> torch.Tensor:
    """1 for first side attempt; decays on distinct same-side attempts only."""
    return torch.clamp(torch.pow(torch.as_tensor(decay, device=streak.device),
                                 torch.clamp(streak - 1, min=0).float()), min=floor, max=1.0)


def monotone_acceptance_increment(load: torch.Tensor, closeness: torch.Tensor,
                                  load_peak: torch.Tensor, com_peak: torch.Tensor):
    """Bounded incremental positive progress; repeated load oscillations do not pay twice."""
    l = torch.clamp((load - load_peak).clamp(min=0.0) / 0.20, 0.0, 1.0)
    c = torch.clamp((closeness - com_peak).clamp(min=0.0) / 0.12, 0.0, 1.0)
    return 0.70 * l + 0.30 * c


def recovery_window_update(tilt, rate, previous_tilt, previous_rate, armed,
                           stable_steps, remaining_s, dt, window_s=0.30):
    """Re-arm only after genuinely quiet upright samples; static lean is not recovery.

    A new disturbance must depart from that quiet state with BOTH significant tilt
    and fast roll/pitch motion. The exemption expires and cannot restart until the
    robot has settled, irrespective of long-lived lean or oscillation.
    """
    stable = (tilt < math.radians(10.0)) & (rate < 0.25)
    stable_steps = torch.where(stable, stable_steps+1, torch.zeros_like(stable_steps))
    armed = armed | (stable_steps >= 5)
    onset = (armed & (tilt >= math.radians(12.0)) & (rate >= 0.50) &
             ((tilt - previous_tilt >= math.radians(2.0)) |
              (rate - previous_rate >= 0.25)))
    remaining_s = torch.where(onset, torch.full_like(remaining_s, window_s),
                              (remaining_s-dt).clamp(min=0.0))
    armed = armed & (~onset)
    # Persistent leaning after the finite correction interval cannot renew it.
    exempt = (remaining_s > 0.0) & (rate > 0.25)
    return exempt, onset, armed, stable_steps, remaining_s


def load_preparation_score(load, closeness):
    """Both received load and COM proximity must improve; bounded [0,1]."""
    load_quality = ((load-0.12)/0.43).clamp(0.0, 1.0)
    com_quality = ((closeness-0.28)/0.32).clamp(0.0, 1.0)
    return torch.sqrt(load_quality * com_quality)


def _evaluate_v111(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.08, clearance_full_m: float = 0.022,
    rise_full_mps: float = 0.100, forward_full_mps: float = 0.22,
    max_stance_speed_mps: float = 0.12, attempt_threshold: float = 0.065,
    rearm_threshold: float = 0.025, rearm_steps: int = 4,
    attempt_cooldown_s: float = 0.25, individual_bootstrap: float = 0.55,
    opposite_attempt_bonus: float = 0.45, repeat_decay: float = 0.68,
    repeat_floor: float = 0.20, recency_decay: float = 0.985,
    full_tilt_deg: float = 25.0, zero_tilt_deg: float = 55.0,
    full_height_m: float = 0.400, zero_height_m: float = 0.340,
    recovery_window_s: float = 0.30, prep_window_s: float = 0.75,
) -> dict[str, torch.Tensor]:
    if not (0.0 < rearm_threshold < attempt_threshold < 1.0):
        raise ValueError('v11.1 invalid swing attempt hysteresis')
    if rearm_steps < 1 or not (0.0 < repeat_decay < 1.0 and 0 < repeat_floor < 1):
        raise ValueError('v11.1 invalid repeat attenuation')
    if not 0.0 < recency_decay < 1.0 or not 0 < individual_bootstrap < 1.0:
        raise ValueError('v11.1 invalid recency and bootstrap')
    if min(clearance_full_m,rise_full_mps,forward_full_mps,max_stance_speed_mps) <= 0:
        raise ValueError('v11.1 invalid physical scales')
    if not (0.10 <= recovery_window_s <= 0.50 and 0.20 <= prep_window_s <= 1.50):
        raise ValueError('v11.1 R3 invalid reward windows')

    lengths = env.episode_length_buf
    previous = getattr(env, '_lgh111_cached_ep_len', None)
    cached = getattr(env, '_lgh111_cached_result', None)
    # Two registered reward terms call this once per step, but must see the
    # SAME event updates and receive independent output channels.
    if previous is not None and cached is not None and torch.equal(previous, lengths):
        return cached

    k = _v67_whole_foot_kinematics(env,command_name,sensor_cfg,asset_cfg,
                                  full_tilt_deg,zero_tilt_deg,full_height_m,zero_height_m)
    n = lengths.shape[0]
    d = k['load_share'].device
    dtype = k['load_share'].dtype
    zeros = lambda: torch.zeros((n,2),device=d,dtype=dtype)
    bools = lambda: torch.zeros((n,2),device=d,dtype=torch.bool)
    reset = (lengths < previous) if previous is not None else torch.ones(n,device=d,dtype=torch.bool)
    # Commands change every several seconds in training. Old history must not
    # punish fresh swing acquisition under a new direction/standing epoch.
    prior_cmd = getattr(env,'_lgh111_cmd_epoch',None)
    if prior_cmd is not None:
        reset = reset | ((k['command'][:,:3]-prior_cmd).abs().amax(dim=1) >= 0.015)
    env._lgh111_cmd_epoch = k['command'][:,:3].detach().clone()

    def state(name,template):
        v=getattr(env,'_lgh111_'+name,None)
        if v is None or v.shape != template.shape:
            v=template.clone()
        if v.ndim>1:
            return torch.where(reset[:,None],template,v)
        return torch.where(reset,template,v)

    other = torch.tensor([1,0],device=d)
    stance_vel = k['center_vel'][:,other,:]
    relative = k['center_vel']-stance_vel
    along = (relative*k['command_dir_world'][:,None,:]).sum(-1)
    forward = ((along-0.010)/forward_full_mps).clamp(0.0,1.0)
    rise = ((k['coherent_rise']-0.005)/rise_full_mps).clamp(0.0,1.0)
    clear = ((k['whole_clear']-0.002)/clearance_full_m).clamp(0.0,1.0)
    unload = ((0.55-k['load_share'])/0.35).clamp(0.0,1.0)
    foot_speed = torch.linalg.vector_norm(stance_vel,dim=-1)
    anchor = (1.0-foot_speed/max_stance_speed_mps).clamp(0.0,1.0)
    # IMPORTANT: any_contact is (main_contact OR toe_contact) for each foot.
    # Negation therefore requires BOTH physical main and toe contact to be gone.
    # Soft precursor kinematics are still computed for diagnostics, but must
    # never pay bilateral discovery or arm an attempt during toe/heel pivoting.
    airborne = ~k['any_contact']
    partial_contact = k['any_contact'] & (~k['full_plant'])
    precursor = 0.25*rise + 0.20*clear + 0.55*forward*clear*(0.25+0.75*airborne.float())
    # Both main and toe should be planted on the current *other* side.
    other_planted=k['full_plant'][:,other] & (k['load_share'][:,other] >= 0.54)
    moving=k['command_norm']>command_threshold
    directional=((along+0.010)/0.075).clamp(0.0,1.0)
    ungated_quality=(unload*precursor*directional*other_planted.float()*anchor*
                     k['viability'][:,None]*moving[:,None].float()).clamp(0,1)
    # Do not credit an airborne 'attempt' while even one foot component is
    # in contact. This includes the dense bootstrap term, event counts,
    # recent bilateral quality, opposite-side bonus, and repeat history.
    quality=ungated_quality*airborne.float()

    recent=state('recent',zeros())
    recent=torch.maximum(recent*recency_decay,quality)
    below=state('below',torch.full((n,2),rearm_steps,device=d,dtype=torch.long))
    armed=state('armed',torch.ones((n,2),device=d,dtype=torch.bool))
    cooldown=state('cooldown',zeros())
    dt=float(env.step_dt)
    cooldown=(cooldown-dt).clamp(min=0)
    events=(quality>attempt_threshold)&armed&(cooldown<=0)&(k['whole_clear']>0.004)
    # Only new bouts can change repetition history; a one-second airborne
    # interval remains one event, not fifty consecutive repeated attempts.
    below=torch.where(quality<rearm_threshold,below+1,torch.zeros_like(below))
    armed=torch.where(below>=rearm_steps,torch.ones_like(armed),armed)
    armed=torch.where(events,torch.zeros_like(armed),armed)
    cooldown=torch.where(events,torch.full_like(cooldown,attempt_cooldown_s),cooldown)

    # R3: distinguish a short corrective disturbance from an enduring forward
    # lean. Onset requires prior quiet re-arming and rising tilt/rate.
    gravity=k['asset'].data.projected_gravity_b
    tilt=torch.atan2(torch.linalg.vector_norm(gravity[:,:2],dim=1), -gravity[:,2])
    angular=k['asset'].data.root_ang_vel_b
    ang_rate=torch.linalg.vector_norm(angular[:,:2],dim=1)
    prev_tilt=state('recovery_prev_tilt',tilt.clone())
    prev_rate=state('recovery_prev_rate',ang_rate.clone())
    recovery_armed=state('recovery_armed',torch.zeros(n,device=d,dtype=torch.bool))
    recovery_stable_steps=state('recovery_stable_steps',torch.zeros(n,device=d,dtype=torch.long))
    recovery_remaining=state('recovery_remaining',torch.zeros(n,device=d,dtype=dtype))
    recovery,recovery_onset,recovery_armed,recovery_stable_steps,recovery_remaining=(
        recovery_window_update(tilt,ang_rate,prev_tilt,prev_rate,recovery_armed,
                               recovery_stable_steps,recovery_remaining,dt,recovery_window_s))
    strafe=(k['command'][:,1].abs()>=0.10)&(k['command'][:,1].abs()>=0.8*k['command'][:,0].abs())
    exempt=recovery|strafe

    last=state('last_side',torch.full((n,),-1,device=d,dtype=torch.long))
    streak=state('repeat_streak',torch.zeros(n,device=d,dtype=torch.long))
    credit=state('side_credit',torch.ones((n,2),device=d,dtype=dtype))
    event_counts=state('event_counts',torch.zeros((n,2),device=d,dtype=torch.long))
    side_event=events.clone()
    for side in (0,1):
        # All genuine swing bouts advance the side history. Recovery may
        # temporarily waive attenuation, but must not erase those bouts.
        new=events[:,side]
        same=last==side
        new_streak=torch.where(same,streak+1,torch.ones_like(streak))
        new_credit=repeat_multiplier(new_streak,repeat_decay,repeat_floor)
        # This is a prospective 'other-leg discovery bonus', bounded and
        # obtainable on its first qualified attempt after unilateral practice.
        bonus=torch.where(~same & (streak>=2),1.0+opposite_attempt_bonus,1.0)
        credit[:,side]=torch.where(new,new_credit*bonus,credit[:,side])
        streak=torch.where(new,new_streak,streak)
        last=torch.where(new,torch.full_like(last,side),last)
        event_counts[:,side]+=events[:,side].long()
    # No attenuation of necessary balance recovery or strafe steps. The
    # exemption does NOT grant any independent positive reward.
    effective_credit=torch.where(exempt[:,None],torch.ones_like(credit),credit)
    current=(quality*effective_credit).clamp(0,1.45)
    paired=torch.sqrt((recent[:,0]*recent[:,1]).clamp(min=0.0))
    swing_reward=(individual_bootstrap*0.5*current.sum(dim=1) +
                  (1.0-individual_bootstrap)*paired*current.max(dim=1).values).clamp(0,1.0)

    # R3: early receiving-foot preparation is available following real whole-
    # foot flight, including failed forward travel, BEFORE strict baton takeover.
    # A single command epoch maintains a lifetime-high coupled load/COM score
    # for each side. Repeated rocking to an earlier peak earns no new credit.
    prep_seen_air=state('prep_seen_air',bools())
    prep_air_age=state('prep_air_age',zeros())
    prep_prev_plant=state('prep_prev_plant',k['full_plant'].clone())
    prep_active=state('prep_active',bools())
    prep_age=state('prep_age',zeros())
    prep_high=state('prep_high',zeros())
    prep_seen_air=prep_seen_air | (airborne & (k['whole_clear']>=0.006) &
                                  (k['viability'][:,None]>=0.55) & moving[:,None])
    prep_air_age=torch.where(airborne & (k['whole_clear']>=0.006),
                             torch.zeros_like(prep_air_age),prep_air_age+dt)
    prep_seen_air=prep_seen_air & (prep_air_age<=0.90)
    # Stance-foot anchoring is required during receiving-foot preparation.
    prep_anchor=(torch.linalg.vector_norm(k['center_vel'][:,other,:],dim=-1)<max_stance_speed_mps)
    prep_fresh=(prep_seen_air & (~prep_prev_plant) & k['full_plant'] &
                k['full_plant'][:,other] & (k['load_share'][:,other]>=0.30) &
                prep_anchor & (k['viability'][:,None]>=0.55) & moving[:,None])
    prep_active=prep_active|prep_fresh
    prep_age=torch.where(prep_fresh,torch.zeros_like(prep_age),prep_age+dt*prep_active.float())
    # Establish the touchdown baseline without paying a touchdown bonus.
    prep_score=load_preparation_score(k['load_share'],k['closeness'])
    prep_high=torch.where(prep_fresh,torch.maximum(prep_high,prep_score),prep_high)
    prep_valid=(prep_active & k['full_plant'] & k['full_plant'][:,other] &
                (k['load_share'][:,other]>=0.30) & prep_anchor &
                (k['viability'][:,None]>=0.55) & moving[:,None])
    prep_increment=(prep_score-prep_high).clamp(min=0.0)*prep_valid.float()
    prep_high=torch.where(prep_valid,torch.maximum(prep_high,prep_score),prep_high)
    prep_active=prep_active & (prep_age<=prep_window_s) & (k['load_share']<0.62)
    prep_seen_air=prep_seen_air & (~prep_fresh)
    prep_prev_plant=k['full_plant'].clone()
    prep_reward=prep_increment.sum(dim=1).clamp(0.0,1.0)

    # Receiving-foot load progression is armed only by TRUE whole-foot airborne
    # forward directional passage. No plant/heel-only movement can arm it.
    pending=state('pending',bools())
    pending_age=state('pending_age',zeros())
    load_peak=state('load_peak',zeros())
    com_peak=state('com_peak',zeros())
    eligible=airborne & (k['whole_clear']>=0.008)&(along>0.030)&(k['viability'][:,None]>=0.55)&other_planted & moving[:,None]
    fresh=eligible&(~pending)
    pending=pending|fresh
    load_peak=torch.where(fresh,k['load_share'],load_peak)
    com_peak=torch.where(fresh,k['closeness'],com_peak)
    pending_age=torch.where(fresh,torch.zeros_like(pending_age),pending_age+dt*pending.float())
    # Full main-and-toe contact AND motionless former stance is essential.
    replant=k['full_plant'] & (k['load_share']>0.03) & (foot_speed<max_stance_speed_mps)
    valid=pending & replant & moving[:,None] & (k['viability'][:,None]>=0.55)
    progress=monotone_acceptance_increment(k['load_share'],k['closeness'],load_peak,com_peak)
    progress=progress*valid.float()*k['viability'][:,None]
    load_delta=((k['load_share']-load_peak).clamp(min=0.0))*valid.float()
    com_delta=((k['closeness']-com_peak).clamp(min=0.0))*valid.float()
    load_peak=torch.where(valid,torch.maximum(load_peak,k['load_share']),load_peak)
    com_peak=torch.where(valid,torch.maximum(com_peak,k['closeness']),com_peak)
    # Stop the reward window on actual load acceptance or expiry.
    completed=valid & (k['load_share']>=0.55)&(k['closeness']>=0.53)
    pending=pending & (~completed) & (pending_age<=0.90)
    receiver_reward=progress.sum(dim=1).clamp(0.0,1.0)

    for name,value in {
        'recent':recent,'below':below,'armed':armed,'cooldown':cooldown,
        'last_side':last,'repeat_streak':streak,'side_credit':credit,
        'event_counts':event_counts,'pending':pending,'pending_age':pending_age,
        'recovery_prev_tilt':tilt,'recovery_prev_rate':ang_rate,
        'recovery_armed':recovery_armed,'recovery_stable_steps':recovery_stable_steps,
        'recovery_remaining':recovery_remaining,
        'prep_seen_air':prep_seen_air,'prep_air_age':prep_air_age,
        'prep_prev_plant':prep_prev_plant,'prep_active':prep_active,
        'prep_age':prep_age,'prep_high':prep_high,
        'load_peak':load_peak,'com_peak':com_peak,
    }.items():
        setattr(env,'_lgh111_'+name,value.detach().clone())

    output={
        'swing':swing_reward,'receiver':receiver_reward,'prep':prep_reward,
        'quality':quality,'ungated_quality':ungated_quality,
        'airborne':airborne,'partial_contact':partial_contact,
        'pivot_credit_blocked':partial_contact & (ungated_quality>0),
        'paired':paired,'events':side_event,
        'side_credit':effective_credit,'repeat_streak':streak,'recovery_exempt':recovery,
        'strafe_exempt':strafe,'recovery_onset':recovery_onset,
        'recovery_window_s':recovery_remaining,'recovery_armed':recovery_armed,
        'recovery_exempt_events':side_event & recovery[:,None],
        'prep_fresh':prep_fresh,'prep_valid':prep_valid,'prep_progress':prep_increment,
        'prep_score':prep_score,'prep_high':prep_high,
        'receiver_load_delta':load_delta,
        'receiver_com_delta':com_delta,'receiver_progress':progress,'receiver_complete':completed,
        'receiver_pending':pending,
    }
    env._lgh111_cached_ep_len=lengths.clone().detach()
    env._lgh111_cached_result={key:v.detach() for key,v in output.items()}
    return env._lgh111_cached_result



def bilateral_swing_discovery_v111(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.08,
    clearance_full_m: float = 0.022,
    rise_full_mps: float = 0.100,
    forward_full_mps: float = 0.22,
    max_stance_speed_mps: float = 0.12,
    attempt_threshold: float = 0.065,
    rearm_threshold: float = 0.025,
    rearm_steps: int = 4,
    attempt_cooldown_s: float = 0.25,
    individual_bootstrap: float = 0.55,
    opposite_attempt_bonus: float = 0.45,
    repeat_decay: float = 0.68,
    repeat_floor: float = 0.20,
    recency_decay: float = 0.985,
    full_tilt_deg: float = 25.0,
    zero_tilt_deg: float = 55.0,
    full_height_m: float = 0.400,
    zero_height_m: float = 0.340,
    recovery_window_s: float = 0.30,
    prep_window_s: float = 0.75,
) -> torch.Tensor:
    """Isaac Lab-compatible explicit parameters; no variadic kwargs.

    ManagerBase validates all RewardTermCfg.params against this signature.
    Both terms call one per-step cached implementation.
    """
    return _evaluate_v111(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        command_threshold=command_threshold,
        clearance_full_m=clearance_full_m,
        rise_full_mps=rise_full_mps,
        forward_full_mps=forward_full_mps,
        max_stance_speed_mps=max_stance_speed_mps,
        attempt_threshold=attempt_threshold,
        rearm_threshold=rearm_threshold,
        rearm_steps=rearm_steps,
        attempt_cooldown_s=attempt_cooldown_s,
        individual_bootstrap=individual_bootstrap,
        opposite_attempt_bonus=opposite_attempt_bonus,
        repeat_decay=repeat_decay,
        repeat_floor=repeat_floor,
        recency_decay=recency_decay,
        full_tilt_deg=full_tilt_deg,
        zero_tilt_deg=zero_tilt_deg,
        full_height_m=full_height_m,
        zero_height_m=zero_height_m,
        recovery_window_s=recovery_window_s,
        prep_window_s=prep_window_s,
    )['swing']

def receiving_foot_load_acceptance_v111(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.08,
    clearance_full_m: float = 0.022,
    rise_full_mps: float = 0.100,
    forward_full_mps: float = 0.22,
    max_stance_speed_mps: float = 0.12,
    attempt_threshold: float = 0.065,
    rearm_threshold: float = 0.025,
    rearm_steps: int = 4,
    attempt_cooldown_s: float = 0.25,
    individual_bootstrap: float = 0.55,
    opposite_attempt_bonus: float = 0.45,
    repeat_decay: float = 0.68,
    repeat_floor: float = 0.20,
    recency_decay: float = 0.985,
    full_tilt_deg: float = 25.0,
    zero_tilt_deg: float = 55.0,
    full_height_m: float = 0.400,
    zero_height_m: float = 0.340,
    recovery_window_s: float = 0.30,
    prep_window_s: float = 0.75,
) -> torch.Tensor:
    """Isaac Lab-compatible explicit parameters; no variadic kwargs.

    ManagerBase validates all RewardTermCfg.params against this signature.
    Both terms call one per-step cached implementation.
    """
    return _evaluate_v111(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        command_threshold=command_threshold,
        clearance_full_m=clearance_full_m,
        rise_full_mps=rise_full_mps,
        forward_full_mps=forward_full_mps,
        max_stance_speed_mps=max_stance_speed_mps,
        attempt_threshold=attempt_threshold,
        rearm_threshold=rearm_threshold,
        rearm_steps=rearm_steps,
        attempt_cooldown_s=attempt_cooldown_s,
        individual_bootstrap=individual_bootstrap,
        opposite_attempt_bonus=opposite_attempt_bonus,
        repeat_decay=repeat_decay,
        repeat_floor=repeat_floor,
        recency_decay=recency_decay,
        full_tilt_deg=full_tilt_deg,
        zero_tilt_deg=zero_tilt_deg,
        full_height_m=full_height_m,
        zero_height_m=zero_height_m,
        recovery_window_s=recovery_window_s,
        prep_window_s=prep_window_s,
    )['receiver']

def receiving_foot_preparation_v111(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.08,
    clearance_full_m: float = 0.022,
    rise_full_mps: float = 0.100,
    forward_full_mps: float = 0.22,
    max_stance_speed_mps: float = 0.12,
    attempt_threshold: float = 0.065,
    rearm_threshold: float = 0.025,
    rearm_steps: int = 4,
    attempt_cooldown_s: float = 0.25,
    individual_bootstrap: float = 0.55,
    opposite_attempt_bonus: float = 0.45,
    repeat_decay: float = 0.68,
    repeat_floor: float = 0.20,
    recency_decay: float = 0.985,
    full_tilt_deg: float = 25.0,
    zero_tilt_deg: float = 55.0,
    full_height_m: float = 0.400,
    zero_height_m: float = 0.340,
    recovery_window_s: float = 0.30,
    prep_window_s: float = 0.75,
) -> torch.Tensor:
    """Isaac Lab-compatible explicit parameters; no variadic kwargs.

    ManagerBase validates all RewardTermCfg.params against this signature.
    Both terms call one per-step cached implementation.
    """
    return _evaluate_v111(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        command_threshold=command_threshold,
        clearance_full_m=clearance_full_m,
        rise_full_mps=rise_full_mps,
        forward_full_mps=forward_full_mps,
        max_stance_speed_mps=max_stance_speed_mps,
        attempt_threshold=attempt_threshold,
        rearm_threshold=rearm_threshold,
        rearm_steps=rearm_steps,
        attempt_cooldown_s=attempt_cooldown_s,
        individual_bootstrap=individual_bootstrap,
        opposite_attempt_bonus=opposite_attempt_bonus,
        repeat_decay=repeat_decay,
        repeat_floor=repeat_floor,
        recency_decay=recency_decay,
        full_tilt_deg=full_tilt_deg,
        zero_tilt_deg=zero_tilt_deg,
        full_height_m=full_height_m,
        zero_height_m=zero_height_m,
        recovery_window_s=recovery_window_s,
        prep_window_s=prep_window_s,
    )['prep']
