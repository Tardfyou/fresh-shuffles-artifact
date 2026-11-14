"""Resolve operator-configured storage without changing scientific inputs."""
import json
from pathlib import Path
from . import ROOT


def configured(name,default):
    path=ROOT/'runtime/settings.json'
    settings=json.loads(path.read_text()) if path.exists() else {}
    value=Path(settings.get(name,default)).expanduser()
    return value if value.is_absolute() else ROOT/value


def result_root():return configured('output_dir','results')
def model_cache():return configured('cache_dir','model_cache/hub')
