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
from .records import Records,PauseRequested


def write(path,obj):
    from . import records
    path=Path(path)
    text=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if records.CURRENT_INVOCATION:
        archived=path.parent/'invocations'/records.CURRENT_INVOCATION/path.name
        archived.parent.mkdir(parents=True,exist_ok=True)
        if archived.exists():assert archived.read_text()==text, 'Metadata changed within one invocation'
        else:archived.write_text(text)
    path.write_text(text)


def gpu_allocation_fraction(total_bytes,requested_gib):
    if total_bytes<=0 or not 1<=requested_gib<=78:
        raise ValueError('GPU allocator cap must be between1 and78GiB')
    return min(1.0,(requested_gib*(1<<30))/total_bytes)


def samples_for(job):
    rows=[json.loads(line) for line in (ROOT/'data/samples_main_public.jsonl').open()]
    if job.get('subset'):
        selected=json.loads((ROOT/'data/experiment_subsets.json').read_text())[job['subset']]
        selected={r['sid'] if isinstance(r,dict) else r for r in selected}
        rows=[r for r in rows if r['sid'] in selected]
    if job.get('smoke'):
        rows=rows[:2]
    if job.get('shards'):
        rows=[r for i,r in enumerate(rows) if i%job['shards']==job['shard']]
    if job.get('completed_ids'):
        completed=set(job['completed_ids'])
        rows=[r for r in rows if r['sid'] not in completed]
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
    p.add_argument('--max-gpu-allocation-gib',type=float,default=78)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    from .lock import verify
    verify(required=torch.device(args.device).type=='cuda')
    if torch.device(args.device).type=='cuda':
        import transformers
        if torch.__version__.split('+')[0]!='2.6.0' or transformers.__version__!='4.51.3':
            raise RuntimeError('Use the pinned Torch2.6.0/Transformers4.51.3 environment; record and report incompatibilities instead of silently changing it')
    if torch.device(args.device).type=='cuda':
        total=torch.cuda.get_device_properties(args.device).total_memory
        # The author raised the single-card cap to78GiB after GPU OOM.
        fraction=gpu_allocation_fraction(total,args.max_gpu_allocation_gib)
        torch.cuda.set_per_process_memory_fraction(fraction,args.device)
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
                      external_gate=job.get('external_gate'),
                      configuration_layers=job.get('configuration_layers'),cloak_dtype=job.get('cloak_dtype','bfloat16'),
                      cache_forward=job.get('cache_forward','first_layer'))
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
        if torch.device(args.device).type=='cuda':
            free,total=torch.cuda.mem_get_info(args.device)
            write(root/'resource-usage.json',dict(maximum_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                maximum_reserved_bytes=torch.cuda.max_memory_reserved(args.device),ending_free_bytes=free,total_bytes=total,
                allocator_limit_gib=args.max_gpu_allocation_gib,single_gpu=True))
        print(json.dumps(result),flush=True)
    except PauseRequested:
        print('Paused after a complete retained observation',flush=True)
        raise SystemExit(75)
    except Exception as exc:
        write(root/f'worker-error-{time.time_ns()}.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        store.close()


if __name__=='__main__':main()
