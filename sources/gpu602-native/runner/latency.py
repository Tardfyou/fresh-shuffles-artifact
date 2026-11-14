"""Synchronized end-to-end cache timings, including trusted CPU transfers."""
import json
import os
import random
import time
import uuid
import hashlib

import torch

from . import ROOT
from .core import clone_cache, seed_all, stable_seed, synchronize
from .full_model import Consumer, clone, same_cache
from .worker import write
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak
from corrected.freshness import KeyService
import exp_veilkv_plus as codec


def latency(job,args,store):
    c=Consumer(job,args)
    samples=[json.loads(line) for line in (ROOT/'data/samples_main_public.jsonl').open() if json.loads(line)['or_subset']]
    row=None
    for candidate in sorted(samples,key=lambda r:r['sid']):
        if job.get('source_sid') and candidate['sid']!=job['source_sid']:continue
        ids=c.tokenizer(candidate['text']).input_ids
        if len(ids)>=512:row=candidate;ids=ids[:512];break
    assert row is not None
    out=c.forward(ids,keep=1);plain=clone(out.past_key_values)
    theta={f'layer_{li}':{'key_max_values':K[0].abs().amax(dim=(1,2)).float().cpu().tolist(),
                          'value_max_values':V[0].abs().amax(dim=(1,2)).float().cpu().tolist()} for li,(K,V) in enumerate(plain)}
    seed_all(42);cfg=get_kvcloak_config(c.L,c.H,c.D,job['block'],theta,1.,1.,2.)
    unfused=KVCloak(cfg,c.dtype,False,False,True);fused=KVCloak(cfg,c.dtype,True,False,True)
    seed_all(1234);unfused._prepare_device_tensors(c.device)
    seed_all(1234);fused._prepare_device_tensors(c.device)
    transformed=[]
    for li,(K,V) in enumerate(plain):
        d=unfused._device_cache[c.device]['obf'][li]
        km=d['M_k_batch'].reshape(c.H,c.D,c.D);vm=d['M_v_batch'].reshape(c.H,c.D,c.D)
        transformed.append((torch.matmul(K,km),torch.matmul(V,vm)))
    service=KeyService(store.root/'freshness.sqlite');namespace=uuid.uuid4().hex
    counter=0
    cache_bytes=sum(t.numel()*t.element_size() for pair in plain for t in pair)
    def context(prefix):
        nonlocal counter
        counter+=1
        return dict(**c.dig,tokens=torch.tensor([ids]),n_prefix_blocks=prefix,profile='tee',storage_dtype=c.dtype,
                    principal=b'benchmark',nonce=hashlib.blake2b((namespace+str(counter)).encode(),digest_size=16).digest(),
                    epoch_id=3,variant={},bsz=1,H=c.H,D=c.D,true_len=512,key_service=service)
    methods=['kvcloak_unfused','kvcloak_fused','veilkv_prefix','aead_no_sharing']
    def preparation(method,operation):
        if method.startswith('kvcloak'):
            protector=unfused if method=='kvcloak_unfused' else fused
            source=plain if method=='kvcloak_unfused' else transformed
            if operation=='protect':
                supplied=clone_cache(source)
                return lambda:protector.obfuscate(supplied)
            supplied=protector.obfuscate(clone_cache(source))
            return lambda:protector.deobfuscate(supplied)
        ctx=context(2 if method=='veilkv_prefix' else 0)
        if operation=='protect':
            supplied=clone(plain)
            def protect():
                host=tuple((a.cpu(),b.cpu()) for a,b in supplied)
                return codec.protect_cache(host,ctx)
            return protect
        obj=codec.protect_cache(tuple((a.cpu(),b.cpu()) for a,b in plain),ctx)
        def recover():
            raw=codec.recover_cache(obj,ctx)
            return tuple((a.to(c.device,c.dtype),b.to(c.device,c.dtype)) for a,b in raw)
        return recover
    metadata=dict(c.metadata(),source_sid=row['sid'],tokens=512,cache_bytes=cache_bytes,block_size=job['block'],
                  repetitions=2 if job.get('smoke') else 100,warmups=1 if job.get('smoke') else 20,
                  timing='Wall clock with CUDA synchronization on both boundaries; includes CPU encryption and D2H/H2D copies.',
                  preparation_outside_timing=['fresh input cloning','nonce construction','recover-input protection'],
                  fused_input='VM precomputed outside timing; does not measure fused-model prefill',nonce_namespace=namespace)
    write(store.root/'environment.json',metadata)
    schedule=[(method,operation) for method in methods for operation in ['protect','recover']]
    for _ in range(metadata['warmups']):
        for method,operation in schedule:
            call=preparation(method,operation);synchronize(c.device);result=call();synchronize(c.device);del result
    tasks=[(rep,method,operation) for rep in range(metadata['repetitions']) for method,operation in schedule]
    random.Random(20261109).shuffle(tasks)
    expected=[]
    for rep,method,operation in tasks:
        sid=f'{method}|{operation}|{rep}';expected.append(sid)
        if sid in store.done:continue
        seed_all(5001+rep)
        call=preparation(method,operation)
        synchronize(c.device);start=time.perf_counter();result=call();synchronize(c.device);elapsed=time.perf_counter()-start
        store.append(sid,status='measured',method=method,operation=operation,repetition=rep,
                     seconds=elapsed,seconds_per_1000_tokens=elapsed*1000/512,
                     milliseconds_per_GiB=elapsed*1000/(cache_bytes/(1<<30)),cache_bytes=cache_bytes)
        del result
    # Separate real-native prefill denominator for the relative-overhead view.
    for _ in range(metadata['warmups']):
        out=c.forward(ids,keep=1);synchronize(c.device);del out
    for rep in range(metadata['repetitions']):
        sid='prefill|'+str(rep);expected.append(sid)
        if sid in store.done:continue
        synchronize(c.device);start=time.perf_counter();out=c.forward(ids,keep=1);synchronize(c.device);elapsed=time.perf_counter()-start
        store.append(sid,status='measured',method='plain_model',operation='prefill',repetition=rep,seconds=elapsed,tokens=512)
        del out
    return store.finish(expected)
