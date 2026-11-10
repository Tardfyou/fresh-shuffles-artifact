"""Execute a frozen native-backend measurement job."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from . import ROOT
from .core import Harness, stable_seed, seed_all, clone_cache, synchronize, first_cache
from .records import Records


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def samples_for(job):
    rows=[json.loads(line) for line in (ROOT/'data/samples_main_public.jsonl').open()]
    if job.get('subset'):
        selected=json.loads((ROOT/'data/experiment_subsets.json').read_text())[job['subset']]
        selected={r['sid'] if isinstance(r,dict) else r for r in selected}
        rows=[r for r in rows if r['sid'] in selected]
    if job.get('smoke'):
        rows=rows[:2]
    return rows


def score_main(h,store,job):
    samples=samples_for(job)
    for i,row in enumerate(samples):
        sid=row['sid']
        if sid in store.done:continue
        start=time.monotonic()
        ids=h.ids(row['text']);cache=h.cache(ids)
        before=tuple(x.clone() for x in cache[0]);scores={}
        for seed in [5001,5002]:
            protected=h.protect(cache,seed)
            assert all(torch.equal(x,y) for x,y in zip(before,cache[0])), 'Official protection contaminated a reused plain cache'
            scores[str(seed)]=h.score(ids,protected)
        synchronize(h.device)
        store.append(sid,status='measured',len_bucket=row['len_bucket'],truth=ids[0].tolist(),
                     seeds=scores,seconds=time.monotonic()-start)
        print(job['id'],i+1,len(samples),sid,sum(s['correct'] for s in scores.values()),flush=True)
    return store.finish([r['sid'] for r in samples])


def score_transfer(h,store,job):
    samples=samples_for(job)
    old_cloak=h.cloak;old_secrets=h.secrets;old_cfg=copy.deepcopy(h.cfg)
    h.reset_key(9999,hold_column=old_cfg if job['kind']=='sa_rotation' else None)
    rotated=h.cloak
    newsecrets,newcal=h.calibrate()
    write(store.root/'rotated-calibration.json',newcal)
    for i,row in enumerate(samples):
        sid=row['sid']
        if sid in store.done:continue
        ids=h.ids(row['text']);cache=h.cache(ids);seed=stable_seed(sid)
        a=h.protect(cache,seed,old_cloak);b=h.protect(cache,seed,rotated)
        conditions={'same_epoch':h.score(ids,a,old_secrets),'stale_after_rotation':h.score(ids,b,old_secrets)}
        if job['kind']=='lifecycle':conditions['recalibrated']=h.score(ids,b,newsecrets)
        store.append(sid,status='measured',truth=ids[0].tolist(),conditions=conditions)
        print(job['id'],i+1,len(samples),flush=True)
    return store.finish([r['sid'] for r in samples])


def score_keys(h,store,job):
    samples=samples_for(job)
    keys=job.get('key_seeds',[42,1234,5678,9012,34567])
    for key in keys:
        h.reset_key(key);h.secrets,cal=h.calibrate();write(store.root/f'calibration-{key}.json',cal)
        for row in samples:
            sid=f'{row["sid"]}-{key}'
            if sid in store.done:continue
            ids=h.ids(row['text']);cache=h.cache(ids)
            scores=h.score(ids,h.protect(cache,stable_seed(sid)))
            store.append(sid,status='measured',source_sid=row['sid'],key_seed=key,truth=ids[0].tolist(),scores=scores)
    return store.finish([f'{r["sid"]}-{k}' for k in keys for r in samples])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'protocol/jobs.json')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--device',default='cpu')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'model_cache/hub')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    jobs=json.loads(args.manifest.read_text())
    job=next(r for r in jobs if r['id']==args.job)
    root=args.output/job['id'];root.mkdir(parents=True,exist_ok=True)
    store=Records(root,job)
    write(root/'job.json',job)
    try:
        if job['kind'] in ['main','lifecycle','sa_rotation','cross_key']:
            h=Harness(job['model'],device=args.device,dtype=job['dtype'],block=job.get('block',16),
                      calibrator=job.get('calibrator','corrected'),cache_dir=args.cache_dir,
                      strategy=job.get('strategy','full'),threads=args.threads,offline=args.offline,
                      external_gate=job.get('external_gate'))
            write(root/'environment.json',h.environment())
            write(root/'calibration.json',h.calibration)
            write(root/'decoder-reference-gate.json',h.verify_decoder())
            gate=dict(h.gate,revision=job['model']['revision'])
            write(root/'native-gate.json',gate)
            call={'main':score_main,'lifecycle':score_transfer,'sa_rotation':score_transfer,'cross_key':score_keys}[job['kind']]
            result=call(h,store,job)
        else:
            from .studies import run_study
            result=run_study(job,args,store)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write(root/f'worker-error-{time.time_ns()}.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
