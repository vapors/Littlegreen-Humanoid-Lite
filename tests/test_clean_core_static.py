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
 assert (ROOT/'VERSION').read_text().strip()=='11.0.0'
 for rel in ['pyproject.toml','source/littlegreen_humanoid_lite/pyproject.toml','source/littlegreen_humanoid_lite_assets/pyproject.toml','source/littlegreen_humanoid_lite_lowlevel/pyproject.toml']:
  assert tomllib.loads((ROOT/rel).read_text())['project']['version']=='11.0.0', rel

def test_registration_is_single_active_task():
 txt=(ROOT/CFG/'__init__.py').read_text()
 assert txt.count('gym.register(')==1
 assert txt.count('Velocity-Lilgreen-Locomotion-ST3215-Loaded-v110')==1
 assert 'v1031' not in txt

def test_reward_expressions_equal_frozen_baseline():
 base=json.loads((ROOT/'manifests/CONTRACT_BASELINE.json').read_text())['reward_terms']
 current=json.loads((ROOT/'manifests/CONTRACT_V11.json').read_text())['reward_terms']
 assert len(base)==len(current)==44
 assert base==current
 tree=ast.parse((ROOT/CFG/'rewards/locomotion.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocomotionRewardsCfg')
 for node in cls.body:
  if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
  name=node.targets[0].id
  if name not in current:continue
  assert ast.dump(node.value,include_attributes=False)==current[name]['expression_ast'], name
 assert current['balanced_swing_discovery']['weight']==0.85

def test_ppo_values_against_baseline():
 a=json.loads((ROOT/'manifests/CONTRACT_BASELINE.json').read_text())['ppo_value_expressions']
 b=json.loads((ROOT/'manifests/CONTRACT_V11.json').read_text())['ppo_value_expressions']
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
 assert '"11.0.0"' in export

def test_assets_and_absent_generated_files():
 usd=ROOT/'source/littlegreen_humanoid_lite_assets/data/usd/lilgreen_mass_v2926.usd'
 assert usd.is_file() and usd.stat().st_size>10_000_000
 for p in ROOT.rglob('*'):
  # Running pytest creates __pycache__, which the release ZIP builder excludes.
  assert not (p.is_file() and p.suffix in {'.pt','.onnx','.zip'})
