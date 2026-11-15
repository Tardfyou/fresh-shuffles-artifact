"""Preserve a crashed writer's original bytes and recover complete JSONL rows."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from . import ROOT
from .paths import result_root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',type=Path,default=result_root());a=p.parse_args()
    folder=a.output/a.job
    lock=(folder/'JOB.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=folder/'records.jsonl';raw=path.read_bytes()
    if raw.endswith(b'\n'):
        for line in raw.splitlines():json.loads(line)
        print('No incomplete transport fragment; no change made')
    else:
        cut=raw.rfind(b'\n')+1;valid=raw[:cut]
        for line in valid.splitlines():json.loads(line)
        archive=folder/'transport_fragments';archive.mkdir(exist_ok=True)
        preserved=archive/('original-'+str(time.time_ns())+'.jsonl.partial')
        path.rename(preserved);path.write_bytes(valid)
        report=dict(original_bytes=len(raw),retained_complete_bytes=len(valid),original_preserved_as=str(preserved.relative_to(folder)),
                    truncated_observation_must_be_retried_on_same_id=True,complete_observations_unchanged=True)
        (archive/('repair-'+str(time.time_ns())+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
