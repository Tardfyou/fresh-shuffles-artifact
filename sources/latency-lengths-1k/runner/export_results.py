"""Create a private raw handoff archive for Codex acceptance, not publication."""
import argparse
import json
from pathlib import Path
import zipfile
from . import ROOT
from .paths import result_root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=result_root());p.add_argument('--dest',type=Path,default=ROOT/'gpu-results-private-handoff.zip');a=p.parse_args()
    files=[x for x in a.output.rglob('*') if x.is_file() and x.suffix not in {'.lock'} and x.resolve()!=a.dest.resolve()]
    with zipfile.ZipFile(a.dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for x in files:z.write(x,'results/'+x.relative_to(a.output).as_posix())
        for x in [ROOT/'SCIENTIFIC_LOCK.json',ROOT/'protocol/jobs.json',ROOT/'protocol/smoke_jobs.json',ROOT/'runtime/settings.json']:
            if x.exists():z.write(x,x.relative_to(ROOT).as_posix())
        z.writestr('HANDOFF_SCOPE.txt','Private raw acceptance handoff. Includes errors, logs, runtime state and original transport fragments. Do not sync directly to the anonymous repository; export accepted portable data after review.\n')
    with zipfile.ZipFile(a.dest) as z:assert z.testzip() is None
    print(json.dumps(dict(archive=str(a.dest),result_files=len(files),private_acceptance_handoff=True)))
