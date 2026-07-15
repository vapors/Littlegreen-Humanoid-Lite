"""Compatibility entry point for the Littlegreen policy exporter.

The canonical implementation is ``export_policy_littlegreen.py``.
"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("export_policy_littlegreen.py")), run_name="__main__")
