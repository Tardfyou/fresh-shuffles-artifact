"""Summarize each model/block/length cell separately, retaining every raw timing."""
import argparse,json,math,statistics
from collections import defaultdict
from pathlib import Path
from . import ROOT
from .records import load

def percentile(a,q):
 a=sorted(a);x=(len(a)-1)*q;lo=math.floor(x);hi=math.ceil(x);return a[lo]+(a[hi]-a[lo])*(x-lo)
def summary(rows):
 ms=[r['seconds']*1000 for r in rows]
 out=dict(timing_calls=len(rows),distinct_inputs=len({r['source_sid'] for r in rows}),tokens_min=min(r['tokens'] for r in rows),tokens_max=max(r['tokens'] for r in rows),p50_ms=statistics.median(ms),p95_ms=percentile(ms,.95),p99_ms=percentile(ms,.99))
 if rows[0]['operation']!='prefill':
  out['p50_seconds_per1000_tokens']=statistics.median(r['seconds_per_1000_tokens'] for r in rows)
  out['p50_milliseconds_per_GiB']=statistics.median(r['milliseconds_per_GiB'] for r in rows)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result={}
 for job in json.loads((ROOT/'protocol/jobs.json').read_text()):
  rows=[r for r in load(a.output/job['id']/'records.jsonl') if r['status']=='measured'];assert len(rows)==9000
  groups=defaultdict(list)
  for row in rows:groups[row['method']+'|'+row['operation']].append(row)
  assert len(groups)==9 and all(len(v)==1000 for v in groups.values())
  result[job['id']]=dict(model=job['model'],dtype=job['dtype'],block=job['block'],length_group=job['length_group'],operations={k:summary(v) for k,v in groups.items()})
 (a.output/'LATENCY_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print('Summarized20 cells,180000 measurements; no pooling across length/model/block profiles')
