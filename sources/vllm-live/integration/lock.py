"""Direct-byte lock for reviewed science files and frozen inputs."""
import json,zipfile
from . import ROOT


def verify():
    manifest=json.loads((ROOT/'SCIENTIFIC_LOCK.json').read_text())
    with zipfile.ZipFile(ROOT/manifest['baseline_archive']) as z:
        changed=[name for name in manifest['files'] if not (ROOT/name).is_file() or (ROOT/name).read_bytes()!=z.read(name)]
    assert not changed,'Frozen files changed: '+', '.join(changed)
    return dict(version=manifest['version'],files=len(manifest['files']),direct_byte_comparison_pass=True)
