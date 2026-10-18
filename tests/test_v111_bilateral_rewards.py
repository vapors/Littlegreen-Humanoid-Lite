"""Actual v11.1 reward-function AST executed against deterministic tensor states.

No Isaac Lab or PhysX required; this tests the state/reward code, not robot
mechanics. The real physics/env checks must still be run on the Isaac Lab host.
"""
from __future__ import annotations
import ast, math
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/mdp/bilateral_acquisition_v111.py'


def _code():
    tree=ast.parse(MODULE.read_text())
    names={'repeat_multiplier','monotone_acceptance_increment','recovery_window_update',
           'load_preparation_score','_evaluate_v111','receiving_foot_preparation_v111',
           'bilateral_swing_discovery_v111','receiving_foot_load_acceptance_v111'}
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    globals_={'torch':torch,'math':math,'__name__':'test_v111_runtime',
              '_v67_whole_foot_kinematics':lambda env,*a,**kw:env.kinematics}
    exec(compile(tree,str(MODULE),'exec'),globals_)
    return globals_


def _state(side='left', forward=0.22, vx=0.20, planted=False,
           receiving_load=None, receiving_closeness=None, recovery=False, strafe=False, contact_mode=None):
    # Ordered [left,right]. Opposite side full-foot planted while chosen leg swings.
    swing=0 if side=='left' else 1
    load=[0.12,0.88] if swing==0 else [0.88,0.12]
    contact=[False,True] if swing==0 else [True,False]
    clear=[0.027,-0.027] if swing==0 else [-0.027,0.027]
    rise=[0.12,-0.12] if swing==0 else [-0.12,0.12]
    vel=[[0.22,0.0],[0.0,0.0]] if swing==0 else [[0.0,0.0],[0.22,0.0]]
    closeness=[0.39,0.61] if swing==0 else [0.61,0.39]
    if planted:
        contact=[True,True];load=[0.5,0.5];clear=[0.,0.];rise=[0.,0.];vel=[[0.,0.],[0.,0.]]
    if receiving_load is not None:
        contact=[True,True];load=[receiving_load,1-receiving_load]
        closeness=[receiving_closeness,1-receiving_closeness]
        clear=[0.,0.];rise=[0.,0.];vel=[[0.,0.],[0.,0.]]
    # Distinguish contact from kinematic clearance: a toe/heel pivot can have
    # positive rise, clearance and forward motion while one component is down.
    if contact_mode in ('main_only','toe_only'):
        contact[swing]=True
    if contact_mode in ('main_only','toe_only'):
        planted_part = contact.copy()
        planted_part[swing]=False
    else:
        planted_part = contact.copy()
    if strafe:
        command=[0.0,0.20,0.0];direction=[0.,1.]
        if not planted and receiving_load is None:
            vel=[[0.,0.22],[0.,0.]] if swing==0 else [[0.,0.],[0.,0.22]]
    else:
        command=[vx,0.,0.];direction=[1.,0.]
    return {'load_share':torch.tensor([load]),'any_contact':torch.tensor([contact]),
            'full_plant':torch.tensor([planted_part]),
            'whole_clear':torch.tensor([clear]),'coherent_rise':torch.tensor([rise]),
            'center_vel':torch.tensor([vel]),'closeness':torch.tensor([closeness]),
            'viability':torch.tensor([1.]),
            'command_norm':torch.tensor([math.hypot(command[0],command[1])]),
            'command':torch.tensor([command]),'command_dir_world':torch.tensor([direction]),
            'asset':SimpleNamespace(data=SimpleNamespace(
                projected_gravity_b=torch.tensor([[0.34,0.,-0.94] if recovery else [0.,0.,-1.]]),
                root_ang_vel_b=torch.tensor([[0.7,0.,0.] if recovery else [0.,0.,0.]])))}


class Env:
    step_dt=0.02
    def __init__(self):
        self.episode_length_buf=torch.tensor([0]);self.kinematics=_state()
    def step(self,state):
        self.episode_length_buf+=1;self.kinematics=state
        return _code()['_evaluate_v111'](self,'base_velocity',None,None)


def test_repeat_credit_and_opposite_leg_bonus():
    env=Env()
    def attempt(side):
        r=env.step(_state(side));assert r['events'][0,0 if side=='left' else 1];return r
    def pause():
        for _ in range(16):env.step(_state(planted=True))
    one=attempt('left')
    assert abs(one['side_credit'][0,0].item()-1)<1e-6
    pause();two=attempt('left')
    assert abs(two['side_credit'][0,0].item()-.68)<1e-6
    pause();three=attempt('left')
    assert abs(three['side_credit'][0,0].item()-.68**2)<1e-6
    pause();right=attempt('right')
    assert abs(right['side_credit'][0,1].item()-1.45)<1e-6
    assert right['repeat_streak'].item()==1
    assert env._lgh111_event_counts.tolist()==[[3,1]]


def test_finite_recovery_and_strafe_suppress_attenuation_without_new_income():
    env=Env(); env.step(_state('left'))
    for _ in range(16):env.step(_state(planted=True))
    env.step(_state('left'))
    for _ in range(16):env.step(_state(planted=True))
    recovered=env.step(_state('left',recovery=True))
    assert recovered['recovery_exempt'].item()
    assert recovered['recovery_onset'].item()
    assert recovered['side_credit'][0,0].item()==1.0
    assert recovered['repeat_streak'].item()==3  # recovery event still counts in history
    for _ in range(20): late=env.step(_state(planted=True,recovery=True))
    assert not late['recovery_exempt'].item(), 'sustained lean must lose its exemption'
    for _ in range(16):env.step(_state(planted=True,strafe=True))
    lateral=env.step(_state('left',strafe=True))
    assert lateral['strafe_exempt'].item()
    assert lateral['side_credit'][0,0].item()==1.0


def test_receiving_load_monotone_and_no_static_farming():
    env=Env(); first=env.step(_state('left'))
    assert first['receiver'].item()==0
    second=env.step(_state(receiving_load=.30,receiving_closeness=.46))
    assert second['receiver'].item()>0
    third=env.step(_state(receiving_load=.30,receiving_closeness=.46))
    assert third['receiver'].item()==0
    # Releasing load again and reaccepting up to the old peak earns nothing.
    env.step(_state(receiving_load=.19,receiving_closeness=.42))
    again=env.step(_state(receiving_load=.30,receiving_closeness=.46))
    assert again['receiver'].item()==0
    same_step=_code()['_evaluate_v111'](env,'base_velocity',None,None)
    assert same_step['receiver'].item()==again['receiver'].item()
    assert env._lgh111_event_counts.tolist()==[[1,0]]


def test_no_rewards_at_zero_command_and_episode_reset():
    env=Env();env.step(_state('left'))
    env.episode_length_buf[:]=0
    out=_code()['_evaluate_v111'](env,'base_velocity',None,None)
    assert out['repeat_streak'].item()==1  # new episode first valid attempt
    env.episode_length_buf+=1
    still=_state(planted=True,vx=0.0)
    env.kinematics=still
    zero=_code()['_evaluate_v111'](env,'base_velocity',None,None)
    assert zero['swing'].item()==0
    assert zero['receiver'].item()==0


def test_load_reward_helpers_are_bounded():
    glob=_code()
    f=glob['monotone_acceptance_increment']
    assert f(torch.tensor([.5]),torch.tensor([.5]),torch.tensor([.6]),torch.tensor([.6])).item()==0
    assert 0<=f(torch.tensor([.9]),torch.tensor([.9]),torch.tensor([0.]),torch.tensor([0.])).item()<=1


def test_isaac_lab_reward_manager_signature_contract():
    """Both v11.1 RewardTermCfg callables must accept every configured key by name.

    Isaac Lab checks function signatures during manager initialization and does
    not regard **kwargs as accepting arbitrary RewardTermCfg.params.
    """
    import inspect
    funcs = _code()
    helper = inspect.signature(funcs['_evaluate_v111'])
    helper_keys = set(helper.parameters) - {'env'}
    reward_cfg = ROOT / ('source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/'
                         'tasks/locomotion/velocity/config/littlegreen_humanoid/'
                         'rewards/locomotion.py')
    tree = ast.parse(reward_cfg.read_text())
    rewards_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and
                       n.name == 'LocomotionRewardsCfg')
    for term_name, func_name in [('balanced_swing_discovery', 'bilateral_swing_discovery_v111'),
                                 ('receiving_foot_load_acceptance', 'receiving_foot_load_acceptance_v111'),
                                 ('receiving_foot_preparation', 'receiving_foot_preparation_v111')]:
        term = next(n.value for n in rewards_cls.body if isinstance(n, ast.Assign) and
                    len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and
                    n.targets[0].id==term_name)
        cfg_params = next(k.value for k in term.keywords if k.arg=='params')
        assert isinstance(cfg_params,ast.Dict)
        configured = {k.value for k in cfg_params.keys if isinstance(k,ast.Constant)}
        # **_VIABILITY is the set of 4 additional required keys supplied by the config.
        assert any(k is None for k in cfg_params.keys)
        configured.update({'full_tilt_deg','zero_tilt_deg','full_height_m','zero_height_m'})
        fn_sig=inspect.signature(funcs[func_name])
        assert all(p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                   for p in fn_sig.parameters.values())
        expected={p for p in fn_sig.parameters if p!='env'}
        assert expected == configured == helper_keys, (term_name, configured ^ expected)
        fn_sig.bind(object(), **{key: 1 for key in configured})

def test_explicit_reward_wrappers_share_per_step_cache():
    f=_code()
    env=Env()
    env.kinematics=_state('left')
    configured={'command_name':'base_velocity','sensor_cfg':None,'asset_cfg':None}
    swing=f['bilateral_swing_discovery_v111'](env,**configured)
    counts=env._lgh111_event_counts.clone()
    receiver=f['receiving_foot_load_acceptance_v111'](env,**configured)
    prep=f['receiving_foot_preparation_v111'](env,**configured)
    assert swing.shape==receiver.shape==prep.shape==(1,)
    assert torch.equal(env._lgh111_event_counts,counts)


def test_partial_contact_never_pays_new_bilateral_reward():
    for side in ('left','right'):
        for mode in ('main_only','toe_only'):
            env=Env()
            result=env.step(_state(side,contact_mode=mode))
            i=0 if side=='left' else 1
            assert result['ungated_quality'][0,i] > 0.0, (side,mode)
            assert result['partial_contact'][0,i].item(), (side,mode)
            assert not result['airborne'][0,i].item(), (side,mode)
            assert result['quality'].abs().max().item()==0.0
            assert result['swing'].item()==0.0
            assert result['events'].sum().item()==0
            assert result['paired'].item()==0.0
            assert result['receiver'].item()==0.0
            assert env._lgh111_event_counts.sum().item()==0
            assert env._lgh111_recent.abs().max().item()==0.0

def test_full_main_toe_release_is_required_before_receiver_arm():
    for mode in ('main_only','toe_only'):
        env=Env()
        env.step(_state('left',contact_mode=mode))
        replanted=env.step(_state(receiving_load=.50,receiving_closeness=.58))
        assert replanted['receiver'].item()==0.0
        assert not replanted['receiver_pending'].any().item()
    env=Env()
    air=env.step(_state('left'))
    assert air['airborne'][0,0].item()
    assert air['events'][0,0].item()
    partial_replant=env.step(_state(receiving_load=.35,receiving_closeness=.48,contact_mode='toe_only'))
    assert partial_replant['receiver'].item()==0.0
    qualified=env.step(_state(receiving_load=.40,receiving_closeness=.52))
    assert qualified['receiver'].item()>0.0

def test_full_air_swing_still_pays_and_repeat_penalty_stays():
    env=Env()
    full=env.step(_state('right'))
    assert full['airborne'][0,1].item()
    assert full['events'][0,1].item()
    assert full['swing'].item()>0.0
    for _ in range(16):env.step(_state(planted=True))
    repeated=env.step(_state('right'))
    assert repeated['side_credit'][0,1].item()<1.0

def test_reward_weights_restored_without_changing_gait_gates():
    reward_cfg=ROOT / 'source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/littlegreen_humanoid/rewards/locomotion.py'
    node=next(n for n in ast.parse(reward_cfg.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LocomotionRewardsCfg')
    expected={'track_lin_vel_xy_acquisition':.85,'coherent_swing_rise_velocity':.90,
              'whole_foot_air_time_exploration':.45,'balanced_swing_discovery':1.2,'receiving_foot_load_acceptance':.85,
              'receiving_foot_preparation':.75}
    for n in node.body:
        if not isinstance(n,ast.Assign) or not isinstance(n.targets[0],ast.Name):continue
        key=n.targets[0].id
        if key in expected:
            wt=next(v.value for v in n.value.keywords if v.arg=='weight')
            assert abs(ast.literal_eval(wt)-expected[key])<1e-8
            del expected[key]
    assert not expected


def test_static_lean_never_grants_recovery_exemption():
    env=Env()
    for _ in range(30): result=env.step(_state('left',recovery=True))
    assert not result['recovery_exempt'].item()
    assert result['side_credit'][0,0].item() <= 1.0
    assert not result['recovery_armed'].item()


def test_recovery_requires_quiet_rearm_after_window():
    env=Env()
    for _ in range(6): env.step(_state(planted=True))
    first=env.step(_state('left',recovery=True))
    assert first['recovery_onset'].item()
    for _ in range(20): timed=env.step(_state(planted=True,recovery=True))
    assert not timed['recovery_exempt'].item()
    assert not timed['recovery_onset'].item()
    for _ in range(5): env.step(_state(planted=True))
    again=env.step(_state('left',recovery=True))
    assert again['recovery_onset'].item()


def test_preparation_requires_full_foot_airborne_and_real_replant():
    for mode in ('main_only','toe_only'):
        env=Env();env.step(_state('left',contact_mode=mode))
        p=env.step(_state(receiving_load=.22,receiving_closeness=.38))
        assert not p['prep_fresh'].any().item()
        q=env.step(_state(receiving_load=.40,receiving_closeness=.50))
        assert q['prep'].item()==0
    env=Env();env.step(_state('left'))
    first=env.step(_state(receiving_load=.22,receiving_closeness=.38))
    assert first['prep_fresh'][0,0].item()
    assert first['prep'].item()==0
    second=env.step(_state(receiving_load=.34,receiving_closeness=.49))
    assert second['prep'].item()>0
    assert second['prep_progress'][0,0].item()>0
    same=env.step(_state(receiving_load=.34,receiving_closeness=.49))
    assert same['prep'].item()==0
    # Repeated rocking back and forth below the previous peak cannot pay.
    env.step(_state(receiving_load=.23,receiving_closeness=.39))
    again=env.step(_state(receiving_load=.34,receiving_closeness=.49))
    assert again['prep'].item()==0


def test_preparation_expires_and_does_not_pay_while_falling_or_sliding():
    env=Env();env.step(_state('left'))
    env.step(_state(receiving_load=.20,receiving_closeness=.38))
    for _ in range(43): env.step(_state(receiving_load=.20,receiving_closeness=.38))
    after=env.step(_state(receiving_load=.42,receiving_closeness=.56))
    assert after['prep'].item()==0
