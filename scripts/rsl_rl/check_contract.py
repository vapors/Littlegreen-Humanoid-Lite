#!/usr/bin/env python3
"""Offline release contract and package provenance checks (NO Isaac Lab runtime)."""
from pathlib import Path
try:  # Python 3.11+ has tomllib; Isaac Lab host often uses Python 3.10.
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import toml as tomllib

import ast,hashlib,json,sys
root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'manifests/CONTRACT_V111.json').read_text())
checks={}
checks['version']=(root/'VERSION').read_text().strip()=='11.1.0'
for rel in ['pyproject.toml','source/littlegreen_humanoid_lite/pyproject.toml','source/littlegreen_humanoid_lite_assets/pyproject.toml','source/littlegreen_humanoid_lite_lowlevel/pyproject.toml']:
 checks[rel+' version']=tomllib.loads((root/rel).read_text())['project']['version']=='11.1.0'
checks['task'] = manifest['task']=='Velocity-Lilgreen-Locomotion-ST3215-Loaded-v111'
checks['reward_diff_documented']=len(json.loads((root/'manifests/REWARD_V111_DIFF.json').read_text())['changes'])==6
checks['reward_module_compiles'] = compile((root/'source/littlegreen_humanoid_lite/littlegreen_humanoid_lite/tasks/locomotion/velocity/config/littlegreen_humanoid/rewards/locomotion.py').read_text(),'locomotion.py','exec') is not None
checks['corrected_usd']=(root/'source/littlegreen_humanoid_lite_assets/data/usd/lilgreen_mass_v2926.usd').is_file()
print(json.dumps({'status':'PASS_STATIC' if all(checks.values()) else 'FAIL_STATIC','checks':checks},indent=2))
sys.exit(0 if all(checks.values()) else 1)
