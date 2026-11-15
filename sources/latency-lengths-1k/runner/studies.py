"""Focused studies on the same frozen cohorts and accepted scientific rules."""
from collections import Counter
import copy
import json
import math
import time

import torch

from . import ROOT
from .core import Harness, stable_seed, seed_all, clone_cache, synchronize
from .worker import samples_for, write
from attack_lib import cluster_states, recover_secrets, ground_truth_errors
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak
from scaling import scaled


def harness(job,args,**extra):
    h=Harness(job['model'],device=args.device,dtype=job['dtype'],block=job.get('block',16),
                   calibrator=job.get('calibrator','corrected'),cache_dir=args.cache_dir,
                   strategy=job.get('strategy','full'),threads=args.threads,offline=args.offline,
                   configuration_layers=job.get('configuration_layers',1),cloak_dtype=job.get('cloak_dtype','bfloat16'),
                   cache_forward=job.get('cache_forward','first_layer'),**extra)
    if hasattr(h,'calibration'):write(args.output/job['id']/'calibration.json',h.calibration)
    return h


def exact_probe(tok,budget):
    for ch in 'abxyz1e0':
        lo,hi=1,max(4096,budget*16)
        while lo<hi:
            mid=(lo+hi+1)//2
            if len(tok(ch*mid).input_ids)<=budget:lo=mid
            else:hi=mid-1
        ids=tok(ch*lo,return_tensors='pt').input_ids
        if ids.shape[1]!=budget:continue
        raw=ids[0].tolist();k=len(raw)-1
        while k>=0 and raw[k]==raw[-1]:k-=1
        if len(raw)-k-1>=max(1,budget//2):return ids,k+1,ch,lo
    raise ValueError('Exact raw-entry probe not found; do not synthesize a replacement token sequence')


def probe(job,args,store):
    h=harness(job,args,calibration=False)
    write(store.root/'environment.json',h.environment())
    keys=[]
    trials=range(2 if job.get('smoke') else 100)
    for trial in trials:
        key=stable_seed(trial,'probe-key');seed_all(key)
        cfg=get_kvcloak_config(1,h.H,h.D,16,h.theta,1.,1.,2.)
        cloak=KVCloak(cfg,h.cloak_dtype,False,False,True)
        seed_all(stable_seed(trial,'probe-marker'));cloak._prepare_device_tensors(h.device)
        keys.append((key,cfg,cloak))
    budgets=[256,2048] if job.get('smoke') else [16,32,64,128,256,512,1024,2048]
    for budget in budgets:
        ids,prefix,ch,nchars=exact_probe(h.tok,budget);ids=ids.to(h.device)
        cache=h.cache(ids);plain=cache[0][1][0].float().cpu().clone()
        first=(prefix+15)//16;count=budget//16
        for trial,(key,cfg,cloak) in enumerate(keys):
            sid=f'probe-keys-{budget}-{trial}'
            if sid in store.done:continue
            protected=h.protect(cache,stable_seed(sid),cloak)
            heads=[]
            for head in range(h.H):
                blocks=protected[0][1][0,head].float().cpu().reshape(-1,16,h.D)[first:count]
                n,centers,*_=cluster_states(blocks)
                row=dict(head=head,states=n,coverage=n==16,parameter_ok=False,decoder_ok=False)
                if n==16:
                    sec=recover_secrets(centers,plain[head,first*16:count*16],16)
                    diagnostics=ground_truth_errors(sec,cfg[0][head][1],plain[head,first*16:count*16])
                    row['diagnostics']=diagnostics
                    row['parameter_ok']=diagnostics['cos_a']>=.999 and diagnostics['S_min_signed_cos']>=.999 and diagnostics['S_bijection']
                    sec={k:v.to(h.device) if torch.is_tensor(v) else v for k,v in sec.items()}
                    pred=h.decode_block(protected[0][1][0,head,first*16:(first+1)*16].float(),sec,head)
                    row['decoder_ok']=Counter(pred)==Counter(ids[0,first*16:(first+1)*16].tolist())
                heads.append(row)
            store.append(sid,status='measured',budget=budget,key_seed=key,trial=trial,
                         input_ids=ids[0].tolist(),prefix_tokens=prefix,raw_character=ch,raw_characters=nchars,
                         heads=heads,coverage=all(r['coverage'] for r in heads),
                         calibration=all(r['parameter_ok'] and r['decoder_ok'] for r in heads))
        print(job['id'],'budget',budget,'complete',flush=True)
    return store.finish([f'probe-keys-{b}-{i}' for b in budgets for i in trials])


def fused(job,args,store):
    from fusion import fusion_llama
    h=harness(job,args,calibration=False)
    public_weight=h.public_model.model.layers[0].self_attn.v_proj.weight.clone()
    defender=copy.deepcopy(h.public_model)
    fusion_llama(defender,h.cfg)
    assert torch.equal(h.public_model.model.layers[0].self_attn.v_proj.weight,public_weight)
    cloak=KVCloak(h.cfg,h.cloak_dtype,True,False,True)
    secrets,cal=h.calibrate(cloak=cloak,known_model=defender)
    write(store.root/'environment.json',dict(h.environment(),fused=True,public_dictionary_separate=True))
    write(store.root/'calibration.json',cal)
    rows=samples_for(job)
    for row in rows:
        if row['sid'] in store.done:continue
        ids=h.ids(row['text']);cache=h.cache(ids,model=defender)
        protected=h.protect(cache,stable_seed(row['sid']),cloak)
        store.append(row['sid'],status='measured',truth=ids[0].tolist(),scores=h.score(ids,protected,secrets))
    return store.finish([r['sid'] for r in rows])


def scaled_score(h,ids,protected,secrets):
    truth=ids[0].tolist();records=[]
    for head,sec in secrets.items():
        bags=[];counts=[]
        for k in range(len(truth)//16):
            C=protected[0][1][0,head,k*16:(k+1)*16].float()
            best=None
            for j in range(16):
                X=(sec['s']@(C-torch.outer(sec['s'][j],sec['a_hat'])))/sec['ratios']
                sims=torch.nn.functional.normalize(X@sec['M'].T,dim=-1)@h.normalized[:,head].T
                value=sims.max(1).values.sum().item()
                if best is None or value>best[0]:best=(value,sims.argmax(1).tolist())
            bags.append(best[1]);counts.append(sum((Counter(best[1])&Counter(truth[k*16:(k+1)*16])).values()))
        records.append(dict(head=head,bags=bags,block_correct=counts,correct=sum(counts),total=len(truth)))
    return dict(correct=sum(r['correct'] for r in records),total=h.H*len(truth),heads=records)


def ratio(job,args,store):
    from attack_lib import probe_ids
    h=harness(job,args,calibration=False);h.reset_key(42,ratio=job['ratio']);h.secrets,cal=h.calibrate()
    ids,prefix,_=probe_ids(h.tok,target=2048,max_positions=h.model.config.max_position_embeddings)
    ids=ids.to(h.device);plain=h.cache(ids)[0][1][0].float().cpu().clone();protected=h.protect(h.cache(ids),1234)
    first=(prefix+15)//16;count=ids.shape[1]//16;adapt={}
    for head in range(h.H):
        blocks=protected[0][1][0,head].float().cpu().reshape(-1,16,h.D)[first:count]
        n,cents,*_=cluster_states(blocks)
        if n!=16:continue
        fit=scaled(cents,plain[head,first*16:count*16])
        adapt[head]={k:v.to(h.device) if torch.is_tensor(v) else v for k,v in fit.items()}
    write(store.root/'environment.json',dict(h.environment(),ratio=job['ratio']))
    write(store.root/'calibration.json',cal)
    write(store.root/'fit.json',{head:{k:v.detach().cpu().tolist() if torch.is_tensor(v) else v for k,v in sec.items()} for head,sec in adapt.items()})
    rows=samples_for(job)
    for row in rows:
        if row['sid'] in store.done:continue
        ids=h.ids(row['text']);prot=h.protect(h.cache(ids),5001)
        scores={'same_epoch':h.score(ids,prot)}
        if len(adapt)==h.H:scores['adapted']=scaled_score(h,ids,prot,adapt)
        else:scores['adapted']=dict(correct=0,total=h.H*ids.shape[1],calibration_failed_heads=sorted(set(range(h.H))-set(adapt)))
        store.append(row['sid'],status='measured',truth=ids[0].tolist(),ratio=job['ratio'],scores=scores)
    return store.finish([r['sid'] for r in rows])


def topk(scores,true_idx,k):
    value=scores[true_idx];greater=sum(x>value+1e-12 for x in scores);ties=sum(abs(x-value)<=1e-12 for x in scores)
    return min(1.,max(0.,(k-greater)/ties))


def privacy(job,args,store):
    h=harness(job,args)
    rows=[json.loads(line) for line in (ROOT/'data/privacy_inputs.jsonl').open()]
    if job.get('smoke'):rows=rows[:2]
    write(store.root/'environment.json',h.environment())
    old_cloak=h.cloak;secs=h.secrets
    h.reset_key(9999)
    wrong_cloak=h.cloak
    for row in rows:
        sid=row['sid']
        if sid in store.done:continue
        ids=torch.tensor([row['truth']],device=h.device)
        cache=h.cache(ids);observed=h.score(ids,h.protect(cache,stable_seed(sid),old_cloak),secs)
        wrong=h.score(ids,h.protect(cache,stable_seed(sid),wrong_cloak),secs)
        all_scores={}
        for label,rec in [('same_epoch',observed),('wrong_secret',wrong)]:
            bags=[Counter(sum(r['bags'],[])) for r in rec['heads']]
            scores=[sum(2*sum((bag&Counter(candidate)).values())/(sum(bag.values())+len(candidate)) if sum(bag.values())+len(candidate) else 0. for bag in bags)/h.H for candidate in row['candidate_tokens']]
            all_scores[label]=dict(scores=scores,topk={str(k):topk(scores,row['true_idx'],k) for k in [1,5,10]})
        store.append(sid,status='measured',etype=row['etype'],truth=row['truth'],candidate_tokens=row['candidate_tokens'],true_idx=row['true_idx'],observed=observed,wrong_secret=wrong,identification=all_scores)
    return store.finish([r['sid'] for r in rows])


def direction(job,args,store):
    h=harness(job,args)
    rows=[json.loads(line) for line in (ROOT/'data/direction_inputs.jsonl').open()]
    if job.get('smoke'):rows=rows[:2]
    write(store.root/'environment.json',h.environment())
    for row in rows:
        if row['sid'] in store.done:continue
        observations=[]
        for target in [0,1]:
            ids=torch.tensor([row['candidate_tokens'][target]],device=h.device)
            rec=h.score(ids,h.protect(h.cache(ids),stable_seed(row['sid']+f':direction:{target}')),heads=[0])
            bags=rec['heads'][0]['bags']
            scores={}
            for global_only in [False,True]:
                observed=[sum(bags,[])] if global_only else bags
                values=[]
                for candidate in row['candidate_order']:
                    tokens=row['candidate_tokens'][candidate]
                    offered=[tokens] if global_only else [tokens[i:i+16] for i in range(0,len(tokens),16)]
                    values.append(2*sum(sum((Counter(a)&Counter(b)).values()) for a,b in zip(observed,offered))-sum(map(len,observed))-sum(map(len,offered)))
                winners=[i for i,v in enumerate(values) if v==max(values)]
                credit=1/len(winners) if row['candidate_order'].index(target) in winners else 0.
                scores['global_bag' if global_only else 'block_bags']=dict(scores=values,accuracy_credit=credit)
            observations.append(dict(target=target,truth=ids[0].tolist(),bags=bags,comparisons=scores))
        store.append(row['sid'],status='measured',candidate_tokens=row['candidate_tokens'],candidate_order=row['candidate_order'],observations=observations,semantic_direction_validated=False)
    return store.finish([r['sid'] for r in rows])


def run_study(job,args,store):
    functions={'probe':probe,'fused':fused,'ratio':ratio,'privacy':privacy,'direction':direction}
    if job['kind'] in functions:return functions[job['kind']](job,args,store)
    from .extended import run_extended
    return run_extended(job,args,store)
