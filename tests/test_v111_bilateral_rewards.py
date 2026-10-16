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
    names={'repeat_multiplier','monotone_acceptance_increment','_evaluate_v111',
           'bilateral_swing_discovery_v111','receiving_foot_load_acceptance_v111'}
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    globals_={'torch':torch,'math':math,'__name__':'test_v111_runtime',
              '_v67_whole_foot_kinematics':lambda env,*a,**kw:env.kinematics}
    exec(compile(tree,str(MODULE),'exec'),globals_)
    return globals_


def _state(side='left', forward=0.22, vx=0.20, planted=False,
           receiving_load=None, receiving_closeness=None, recovery=False, strafe=False):
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
    if strafe:
        command=[0.0,0.20,0.0];direction=[0.,1.]
        if not planted and receiving_load is None:
            vel=[[0.,0.22],[0.,0.]] if swing==0 else [[0.,0.],[0.,0.22]]
    else:
        command=[vx,0.,0.];direction=[1.,0.]
    return {'load_share':torch.tensor([load]),'any_contact':torch.tensor([contact]),
            'full_plant':torch.tensor([contact]),
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


def test_recovery_and_strafe_suppress_attenuation_without_new_income():
    env=Env(); env.step(_state('left'))
    for _ in range(16):env.step(_state(planted=True))
    env.step(_state('left'))
    for _ in range(16):env.step(_state(planted=True))
    recovered=env.step(_state('left',recovery=True))
    assert recovered['recovery_exempt'].item()
    assert recovered['side_credit'][0,0].item()==1.0
    assert recovered['repeat_streak'].item()==2  # exempt recovery doesn't extend penalty history
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
