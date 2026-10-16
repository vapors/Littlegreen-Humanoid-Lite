"""v11 source-only release assertions. Isaac Lab runtime checked independently on host."""
from pathlib import Path
try:  # Python 3.11+ has tomllib; Isaac Lab host often uses Python 3.10.
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import toml as tomllib

import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
CFG=Path('source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/littlegreen_humanoid')

def test_release_versions():
 assert (ROOT/'VERSION').read_text().strip()=='11.1.0'
 for rel in ['pyproject.toml','source/littlegreen_humanoid_lite/pyproject.toml','source/littlegreen_humanoid_lite_assets/pyproject.toml','source/littlegreen_humanoid_lite_lowlevel/pyproject.toml']:
  assert tomllib.loads((ROOT/rel).read_text())['project']['version']=='11.1.0', rel

def test_registration_is_single_active_task():
 txt=(ROOT/CFG/'__init__.py').read_text()
 assert txt.count('gym.register(')==1
 assert txt.count('Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111')==1
 assert 'v110' not in txt

def test_explicit_v111_reward_change_allowlist():
 base=json.loads((ROOT/'manifests/CONTRACT_V11.json').read_text())['reward_terms']
 current=json.loads((ROOT/'manifests/CONTRACT_V111.json').read_text())['reward_terms']
 changes=json.loads((ROOT/'manifests/REWARD_V111_DIFF.json').read_text())['changes']
 expected={'track_lin_vel_xy_acquisition','coherent_swing_rise_velocity',
           'whole_foot_air_time_exploration','balanced_swing_discovery',
           'receiving_foot_load_acceptance'}
 assert set(changes)==expected
 assert len(base)==44 and len(current)==45
 assert all(base[n]==current[n] for n in base.keys()-expected)
 tree=ast.parse((ROOT/CFG/'rewards/locomotion.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocomotionRewardsCfg')
 for node in cls.body:
  if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
  name=node.targets[0].id
  if name not in current:continue
  assert ast.dump(node.value,include_attributes=False)==current[name]['expression_ast'],name
 assert current['balanced_swing_discovery']['weight']==1.20
 assert current['receiving_foot_load_acceptance']['weight']==0.85

def test_ppo_values_against_baseline():
 a=json.loads((ROOT/'manifests/CONTRACT_V11.json').read_text())['ppo_value_expressions']
 b=json.loads((ROOT/'manifests/CONTRACT_V111.json').read_text())['ppo_value_expressions']
 for field in a:
  if field!='experiment_name': assert a[field]==b[field],field

def test_active_modules_compile():
 for p in (ROOT/'source').rglob('*.py'):compile(p.read_text(),str(p),'exec')
 for p in (ROOT/'scripts/rsl_rl').glob('*.py'):compile(p.read_text(),str(p),'exec')

def test_fresh_resume_source_guard_and_export():
 txt=(ROOT/'scripts/rsl_rl/train_eval.py').read_text()
 assert 'run_contract_v11.json' in txt and '_verify_resume_snapshot(' in txt
 assert 'source_contract_fingerprints' in txt
 assert 'agent_cfg.resume' in txt
 assert '_verify_v41_runtime_contract' not in txt
 export=(ROOT/'scripts/rsl_rl/export_policy_littlegreen.py').read_text()
 assert 'SUPPORTED_TASKS = (LOCOMOTION_TASK,)' in export
 assert '"11.1.0"' in export

def test_assets_and_absent_generated_files():
 usd=ROOT/'source/littlegreen_humanoid_lite_assets/data/usd/lilgreen_mass_v2926.usd'
 assert usd.is_file() and usd.stat().st_size>10_000_000
 for p in ROOT.rglob('*'):
  # Running pytest creates __pycache__, which the release ZIP builder excludes.
  assert not (p.is_file() and p.suffix in {'.pt','.onnx','.zip'})


def test_training_entrypoint_hydra_configuration_injection():
    """Regression: train_eval.py main() must be decorated to receive its two required configs.

    Stands up a fake Hydra decorator around an AST-isolated main, without launching
    Isaac Sim or pretending a real environment was instantiated.
    """
    import copy
    trainer = ROOT / 'scripts/rsl_rl/train_eval.py'
    tree = ast.parse(trainer.read_text(), filename=str(trainer))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    assert len(funcs) == 1
    main_node = funcs[0]
    assert len(main_node.decorator_list) == 1
    decorator_expr = main_node.decorator_list[0]
    assert isinstance(decorator_expr, ast.Call)
    assert isinstance(decorator_expr.func, ast.Name)
    assert decorator_expr.func.id == 'hydra_task_config'
    assert len(decorator_expr.args) == 2
    assert ast.unparse(decorator_expr.args[0]) == 'args_cli.task'
    assert isinstance(decorator_expr.args[1], ast.Constant)
    assert decorator_expr.args[1].value == 'rsl_rl_cfg_entry_point'
    assert [a.arg for a in main_node.args.args] == ['env_cfg', 'agent_cfg']
    # Execute just an AST-isolated representative of the decorated main to verify
    # that a zero-argument script call receives both injected config objects.
    isolated = copy.deepcopy(main_node)
    isolated.body = [ast.Return(value=ast.Tuple(elts=[ast.Name(id='env_cfg',ctx=ast.Load()), ast.Name(id='agent_cfg',ctx=ast.Load())],ctx=ast.Load()))]
    for a in isolated.args.args:a.annotation = None
    isolated.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[isolated],type_ignores=[]))
    supplied_env, supplied_agent = object(), object()
    events = []
    def hydra_task_config(task, agent_entry):
        events.append((task, agent_entry))
        def decorate(func):
            def invoke():
                return func(supplied_env, supplied_agent)
            return invoke
        return decorate
    from types import SimpleNamespace
    scope = {'hydra_task_config':hydra_task_config, 'args_cli':SimpleNamespace(task='Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111')}
    exec(compile(module, str(trainer), 'exec'),scope)
    assert scope['main']() == (supplied_env, supplied_agent)
    assert events == [('Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111','rsl_rl_cfg_entry_point')]


def test_reward_facade_exports_both_new_terms():
 facade=(ROOT/CFG/'mdp.py').read_text()
 assert 'bilateral_swing_discovery_v111' in facade
 assert 'receiving_foot_load_acceptance_v111' in facade


def test_zero_command_export_requires_safe_runtime_semantics():
 export=(ROOT/'scripts/rsl_rl/export_policy_littlegreen.py').read_text()
 assert '"command_zero_supported": task == LOCOMOTION_TASK' in export
 assert '"command_zero_vector": [0.0, 0.0, 0.0]' in export
 assert '"runtime_must_preserve_zero_command"' in export
 assert 'zero_or_positive_forward_with_deadzone_and_clamp;reject_unsupported_axes' in export
 assert 'command_unsupported_negative_forward_behavior' in export
