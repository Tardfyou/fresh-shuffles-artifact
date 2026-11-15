"""Verify the pre-reviewed scientific sources and frozen input files."""
import json
import zipfile
from . import ROOT


def verify(required=True):
    path=ROOT/'SCIENTIFIC_LOCK.json'
    if not path.exists():
        if required:raise RuntimeError('Scientific lock is absent; use the issued package')
        return dict(development_only=True)
    manifest=json.loads(path.read_text());changed=[]
    with zipfile.ZipFile(ROOT/manifest['baseline_archive']) as baseline:
        for name in manifest['files']:
            p=ROOT/name
            if not p.is_file() or p.read_bytes()!=baseline.read(name):changed.append(name)
    if changed:raise RuntimeError('Frozen scientific files changed: '+', '.join(changed))
    return dict(pass_lock=True,files=len(manifest['files']),version=manifest['version'])
