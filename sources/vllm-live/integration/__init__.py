"""Real vLLM V0 protected-cache integration for one CUDA GPU."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT/'science',ROOT/'vendor/kvcloak/core',ROOT/'vendor/kvcloak/config']:sys.path.insert(0,str(p))
