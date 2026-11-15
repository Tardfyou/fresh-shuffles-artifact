"""Fast queue/observation progress view; no models or CUDA initialization."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import statistics
from . import ROOT
from .paths import result_root


def rows(path):
    if not path.exists():return []
    result=[]
    with path.open() as f:
        for line in f:
            if not line.endswith('\n'):continue
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('status')=='measured':result.append(r)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=result_root());a=p.parse_args()
    summaries=[]
    for file in a.output.glob('QUEUE_*.json'):
        data=json.loads(file.read_text());counts=Counter(r.get('status','unknown') for r in data.values())
        print(file.stem,dict(counts))
        for jid,entry in data.items():
            if entry.get('status') not in ['running','paused','blocked']:continue
            measured=rows(a.output/jid/'records.jsonl');last=measured[-30:];eta=None
            expected=json.loads((a.output/jid/'job.json').read_text()).get('expected_observations') if (a.output/jid/'job.json').exists() else None
            if len(last)>2 and expected:
                intervals=[b['recorded_unix']-a['recorded_unix'] for a,b in zip(last,last[1:]) if a.get('invocation_id')==b.get('invocation_id') and b['recorded_unix']>a['recorded_unix']]
                if intervals:eta=max(0,expected-len(measured))*statistics.median(intervals)/60
            print(json.dumps(dict(job=jid,status=entry.get('status'),measured=len(measured),expected=expected,
                                  eta_minutes=round(eta,1) if eta is not None else None,reason=entry.get('reason'),pid=entry.get('pid')),ensure_ascii=False))
