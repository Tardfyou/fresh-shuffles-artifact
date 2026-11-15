"""Native PyTorch reproduction package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT/'vendor/kvcloak/defense/core', ROOT/'vendor/kvcloak/defense/config', ROOT/'science_base']:
    sys.path.insert(0, str(path))
