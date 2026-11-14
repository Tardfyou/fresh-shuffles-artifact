"""Accepted-count summaries; keep each precision/backend/model profile separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from . import ROOT
from .paths import result_root
from .verify_outputs import verify
from .records import load


def prompt_bootstrap(rows,repetitions=2000,seed=20261029):
    """Resample prompts within the original length strata, keeping heads/seeds paired."""
    import numpy as np
    strata=defaultdict(list);correct=[];total=[]
    for i,row in enumerate(rows):
        strata[row['len_bucket']].append(i)
        correct.append(sum(s['correct'] for s in row['seeds'].values()))
        total.append(sum(s['total'] for s in row['seeds'].values()))
    c=np.asarray(correct,dtype=np.int64);n=np.asarray(total,dtype=np.int64)
    rng=np.random.default_rng(seed);rates=[]
    for _ in range(repetitions):
        selected=np.concatenate([rng.choice(group,len(group),replace=True) for _,group in sorted(strata.items())])
        rates.append(c[selected].sum()/n[selected].sum())
    return dict(bootstrap95_percent=(100*np.quantile(rates,[.025,.975])).tolist(),
                bootstrap_repetitions=repetitions,bootstrap_seed=seed,
                bootstrap_unit='prompt within original length stratum; all heads and seeds remain paired',
                independent_prompts=len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=result_root());p.add_argument('--manifest',type=Path,default=ROOT/'protocol/jobs.json');a=p.parse_args();summaries=[]
    for job in json.loads(a.manifest.read_text()):
        folder=a.output/job['id']
        if not (folder/'summary.json').exists():continue
        acceptance=verify(job,folder);rows=[r for r in load(folder/'records.jsonl') if r['status']=='measured']
        summary=dict(job=job['id'],kind=job['kind'],model=job['model']['repo'],revision=job['model']['revision'],dtype=job['dtype'],cloak_dtype=job.get('cloak_dtype'),regime=job.get('regime'),calibrator=job.get('calibrator','corrected'),strategy=job.get('strategy','full'),cache_forward=job.get('cache_forward'),device_class=job.get('device_class'),acceptance=acceptance)
        if job['kind']=='main':
            c=sum(s['correct'] for r in rows for s in r['seeds'].values());n=sum(s['total'] for r in rows for s in r['seeds'].values())
            summary.update(correct=c,total=n,recovery_percent=100*c/n)
            summary.update(prompt_bootstrap(rows))
        elif job['kind']=='official_utility':
            summary['micro_accuracy']=sum(r['correct'] for r in rows)/len(rows)
            if job['benchmark']=='mmlu':
                subjects=defaultdict(list)
                for row in rows:subjects[row['subject']].append(row['correct'])
                summary['subject_macro_accuracy']=sum(sum(v)/len(v) for v in subjects.values())/len(subjects)
        elif job['kind'] in ['byte_defense','sa_functional']:
            summary.update(functional_passes=sum(r['functional_pass'] for r in rows),requests=len(rows),consumer_steps=sum(len(r.get('segments',[])) for r in rows))
        summaries.append(summary)
    (a.output/'MEASURED_SUMMARIES.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps(summaries,indent=2))
