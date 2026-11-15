"""Length-stratified paired timings with unchanged cache-operation primitives."""
import hashlib,json,random,time,uuid
import torch
from .core import clone_cache,seed_all,synchronize
from .full_model import Consumer,clone
from .worker import write
from .latency_plan import inputs_for, METHODS, OPERATIONS
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak
from corrected.freshness import KeyService
import exp_veilkv_plus as codec


def fixture(c,job,ids,service,namespace):
    # Match the accepted byte-consumer canonical-prefix construction, outside timing.
    prefix=tuple(ids[:32])
    if prefix not in c.prefixes:
        out=c.forward(list(prefix),keep=1)
        c.prefixes[prefix]=clone(out.past_key_values)
    if len(ids)>32:
        out=c.forward(ids[32:],cache=c.prefixes[prefix],keep=1)
        plain=clone(out.past_key_values)
    else:
        plain=clone(c.prefixes[prefix])
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
    counter=0
    cache_bytes=sum(t.numel()*t.element_size() for pair in plain for t in pair)
    def context(prefix):
        nonlocal counter
        counter+=1
        return dict(**c.dig,tokens=torch.tensor([ids]),n_prefix_blocks=prefix,profile='tee',storage_dtype=c.dtype,
                    principal=b'benchmark',nonce=hashlib.blake2b((namespace+str(counter)).encode(),digest_size=16).digest(),
                    epoch_id=3,variant={},bsz=1,H=c.H,D=c.D,true_len=len(ids),key_service=service)
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
    return preparation, cache_bytes


def latency(job,args,store):
    c=Consumer(job,args)
    draws,warmups=inputs_for(job)
    namespace=uuid.uuid4().hex
    service=KeyService(store.root/'freshness.sqlite')
    write(store.root/'environment.json',dict(c.metadata(),length_group=job['length_group'],
        repetitions=job['repetitions'],warmups=job['warmups'],host_threads=args.threads,
        sampling='Uniform draws with replacement from main3000, stratified by this model tokenizer; no truncation',
        input_schedule=job['input_schedule'],tokens='Per-record actual length',block_size=job['block'],
        timing='CUDA synchronization at both boundaries; byte profile includes CPU crypto and D2H/H2D copies',
        preparation_outside_timing=['input tokenization','canonical public-prefix then private-tail cache construction','threshold/configuration preparation','fresh input cloning','nonce construction','recover-input protection','fused VM preparation'],
        shared_prefix='Up to2 complete16-token blocks; actual public/private sizes recorded',
        prefill='Separate native full-prefill timing on the same input schedule',
        nonce_namespace=namespace))
    schedule=[(m,o) for m in METHODS for o in OPERATIONS]
    for i,item in enumerate(warmups):
        ids=c.tokenizer(item['text']).input_ids
        assert ids==item['input_ids']
        prepare,_=fixture(c,job,ids,service,namespace+'|warmup|'+str(i))
        for method,operation in schedule:
            call=prepare(method,operation);synchronize(c.device);result=call();synchronize(c.device);del result,call
        del prepare
    for trial,item in enumerate(draws):
        pending=[(m,o) for m,o in schedule if f'{m}|{o}|{trial}' not in store.done]
        if not pending:continue
        ids=c.tokenizer(item['text']).input_ids
        assert ids==item['input_ids'] and job['length_group'][0]<=len(ids)<job['length_group'][1]
        prepare,cache_bytes=fixture(c,job,ids,service,namespace+'|formal|'+str(trial))
        random.Random(20261114+trial).shuffle(pending)
        for method,operation in pending:
            seed_all(5001+trial)
            call=prepare(method,operation)
            synchronize(c.device);start=time.perf_counter();result=call();synchronize(c.device);elapsed=time.perf_counter()-start
            store.append(f'{method}|{operation}|{trial}',status='measured',method=method,operation=operation,
                repetition=trial,source_sid=item['sid'],tokens=len(ids),length_group=job['length_group'],
                seconds=elapsed,seconds_per_1000_tokens=elapsed*1000/len(ids),
                milliseconds_per_GiB=elapsed*1000/(cache_bytes/(1<<30)),cache_bytes=cache_bytes,
                public_prefix_tokens=16*min(2,len(ids)//16) if method=='veilkv_prefix' else 0)
            del result,call
        del prepare
        print(job['id'],'cache timings',trial+1,'/',len(draws),flush=True)
    for item in warmups:
        out=c.forward(item['input_ids'],keep=1);synchronize(c.device);del out
    for trial,item in enumerate(draws):
        sid='prefill|'+str(trial)
        if sid in store.done:continue
        synchronize(c.device);start=time.perf_counter();out=c.forward(item['input_ids'],keep=1);synchronize(c.device);elapsed=time.perf_counter()-start
        store.append(sid,status='measured',method='plain_model',operation='prefill',repetition=trial,
            source_sid=item['sid'],seconds=elapsed,tokens=len(item['input_ids']),length_group=job['length_group'])
        del out
    from .latency_plan import expected_keys
    return store.finish(expected_keys(job))
