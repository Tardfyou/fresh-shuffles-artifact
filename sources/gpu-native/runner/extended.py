"""Remaining native-runtime comparisons; no external model API calls."""
from collections import Counter
import copy
import json
import math
import os
import random
import time

import torch

from . import ROOT
from .core import stable_seed, seed_all, synchronize, clone_cache
from .studies import harness
from .worker import samples_for, write
from .full_model import Consumer, clone, same_cache, logits_diff
from run_suite import NATURAL


def ordering(job,args,store):
    from order_beam import beam_cached
    h=harness(job,args)
    prior=copy.deepcopy(h.model).to(torch.bfloat16).eval()
    bos=h.tok.bos_token_id if h.tok.bos_token_id is not None else h.tok.eos_token_id
    rows=[json.loads(l) for l in (ROOT/'data/samples_main_public.jsonl').open() if json.loads(l)['or_subset']]
    if job.get('smoke'):rows=rows[:2]
    write(store.root/'environment.json',dict(h.environment(),prior_dtype='bfloat16',beam_width=16,prior_context_limit=80,context_limit=96))
    for row in rows:
        sid=row['sid']
        if sid in store.done:continue
        ids=h.ids(row['text']);truth=ids[0].tolist()
        source=h.score(ids,h.protect(h.cache(ids),stable_seed(sid)),heads=[0])
        bags=source['heads'][0]['bags'];oracle=[truth[i:i+16] for i in range(0,len(truth),16)]
        if any(len(b)!=16 for b in bags):
            store.append(sid,status='measured',truth=truth,calibration_failed=True,conditions={},bag_source=source)
            continue
        memo={};conditions={}
        for name,blocks,cross in [('end2end',bags,True),('oracle_ms',oracle,True),('no_cross_block',bags,False)]:
            prediction=[];ctx=[bos];reused=0;start=time.monotonic();synchronize(h.device)
            for bag in blocks:
                context=ctx[-80:] if cross else [bos]
                key=(tuple(sorted(Counter(bag).items())),tuple(context))
                if key in memo:ordered=memo[key];reused+=1
                else:ordered=beam_cached(Counter(bag),context,prior,16);memo[key]=ordered
                prediction+=ordered;ctx=(ctx+ordered)[-96:]
            synchronize(h.device)
            assert all(Counter(prediction[i:i+16])==Counter(blocks[i//16]) for i in range(0,len(prediction),16))
            conditions[name]=dict(prediction=prediction,position_correct=sum(a==b for a,b in zip(truth,prediction)),tokens=len(truth),exact_span=truth==prediction,reused_identical_inputs=reused,seconds=time.monotonic()-start)
        rng=random.Random(stable_seed(sid,'random_order'));prediction=[]
        for bag in bags:
            b=list(bag);rng.shuffle(b);prediction+=b
        conditions['random_order']=dict(prediction=prediction,position_correct=sum(a==b for a,b in zip(truth,prediction)),tokens=len(truth),exact_span=truth==prediction)
        store.append(sid,status='measured',truth=truth,bags=bags,oracle_bags=oracle,bag_source=source,conditions=conditions)
        print(job['id'],sid,flush=True)
    return store.finish([r['sid'] for r in rows])


def candidates(job):
    short='Qwen2.5-0.5B' if job['model']['tag']=='qwen25_05b' else 'Llama-3.2-1B'
    rows=[json.loads(line) for line in (ROOT/'data'/f'spectrum_{short}.jsonl').open()]
    return rows[:2] if job.get('smoke') else rows


def candidate_comparison(job,args,store,arithmetic=False):
    h=harness(job,args,calibration=not arithmetic)
    rows=candidates(job)
    align={}
    codec=None
    if arithmetic:
        import exp_veilkv_plus as codec
        import hashlib,uuid
        from corrected.freshness import KeyService
        service=KeyService(store.root/'freshness.sqlite')
        from .identity import checkpoint_identity
        dig=checkpoint_identity(h.model,h.tok,job['model'])
        nonce_namespace=uuid.uuid4().hex
        def protect(ids,sid):
            cache=h.cache(ids)
            host=tuple((a.cpu(),b.cpu()) for a,b in cache)
            ctx=dict(**dig,tokens=ids.cpu(),n_prefix_blocks=0,profile='arithmetic',storage_dtype=torch.float32,
                     principal=sid.encode(),nonce=hashlib.blake2b((nonce_namespace+sid).encode(),digest_size=16).digest(),epoch_id=3,variant={},bsz=1,H=h.H,D=h.D,true_len=ids.shape[1],key_service=service)
            obj=codec.protect_cache(host,ctx)
            return torch.stack([obj.blocks[(0,1)][0]['ct'][0,hh] for hh in range(h.H)]).to(h.device).float()
    else:
        def protect(ids,sid):return h.protect(h.cache(ids),int(sid))[0][1][0].float()
    # Public-text alignment has the same four extra public sentences as before.
    for hh in range(h.H):
        plain=[];protected=[]
        for i,text in enumerate(NATURAL[:4]):
            ids=h.ids(text)
            if ids.shape[1]==0:continue
            cache=h.cache(ids)
            if arithmetic:
                for k in range(ids.shape[1]//16):
                    one=ids[:,k*16:(k+1)*16];plain.append(h.cache(one)[0][1][0,hh].float());protected.append(protect(one,f'train-{hh}-{i}-{k}')[hh])
            else:
                prot=h.protect(cache,5001)
                for k in range(ids.shape[1]//16):plain.append(cache[0][1][0,hh,k*16:(k+1)*16].float());protected.append(prot[0][1][0,hh,k*16:(k+1)*16].float())
        vm,cm=torch.cat(plain),torch.cat(protected)
        u,_,vh=torch.linalg.svd(cm.T@vm);align[hh]=u@vh
    write(store.root/'environment.json',dict(h.environment(),profile='VeilKV arithmetic' if arithmetic else 'KV-Cloak',profile_execution_device='cpu' if arithmetic else str(h.device),candidate_task=True,
        nonce_namespace=nonce_namespace if arithmetic else None,codec_format_version=2 if arithmetic else None,
        alignment_public_sentences=4,alignment_used_only_by='procrustes'))
    def eig(x):return torch.linalg.eigvalsh(x@x.T/x.shape[1])
    def norms(x):return x.norm(dim=-1).sort().values
    for row in rows:
        if row['sid'] in store.done:continue
        ids=torch.tensor([row['truth']],device=h.device);token_ids=torch.tensor(row['candidates'],device=h.device)
        representations={hh:h.dictionary[token_ids,hh] for hh in range(h.H)}
        observed={}
        for seed in ([5001] if arithmetic else [5001,5002]):
            bp=protect(ids,row['sid'] if arithmetic else seed)
            results={}
            rules=['gram','sorted_row_norm','procrustes'] if arithmetic else ['main','gram','sorted_row_norm','procrustes','b1']
            for rule in rules:
                for budget in (['first'] if arithmetic else ['first','all']):
                    heads=[0] if budget=='first' else list(range(h.H));tot=torch.zeros(100,device=h.device);start=time.monotonic();synchronize(h.device)
                    for hh in heads:
                        obs=bp[hh,:16];cand=representations[hh]
                        if rule=='gram':values=-torch.stack([(eig(obs)-eig(x)).norm() for x in cand])
                        elif rule=='sorted_row_norm':values=-torch.stack([(norms(obs)-norms(x)).norm() for x in cand])
                        elif rule=='procrustes':values=torch.stack([torch.nn.functional.cosine_similarity((obs@align[hh]).flatten(),x.flatten(),dim=0) for x in cand])
                        else:
                            cn=torch.nn.functional.normalize(cand,dim=-1)
                            if rule=='main':
                                sec=h.secrets.get(hh)
                                if sec is None:values=torch.full((100,),-1e9,device=h.device)
                                else:
                                    values=torch.full((100,),-1e9,device=h.device)
                                    for j in range(16):
                                        X=torch.nn.functional.normalize(sec['s']@(obs-torch.outer(sec['s'][j],sec['a_hat']))@sec['M'].T,dim=-1)
                                        one=(X[None]@cn.transpose(-1,-2)).max(-1).values.sum(-1)
                                        values=torch.maximum(values,one)
                            else:
                                X=torch.nn.functional.normalize(obs,dim=-1);values=(X[None]@cn.transpose(-1,-2)).max(-1).values.sum(-1)
                        tot+=values
                    if budget=='all' and rule in ['gram','sorted_row_norm','procrustes']:tot/=h.H
                    synchronize(h.device);chosen=int(tot.argmax());truth_score=tot[row['true_idx']].item();better=int((tot>truth_score+1e-9).sum());ties=int((tot-truth_score).abs().le(1e-9).sum())
                    results[rule+'_'+budget]=dict(chosen=chosen,hit=chosen==row['true_idx'],scores=tot.tolist(),top1_expected=1/ties if better==0 else 0.,seconds=time.monotonic()-start,heads=heads)
            observed[str(seed)]=results
        store.append(row['sid'],status='measured',truth=row['truth'],candidates=row['candidates'],true_idx=row['true_idx'],seeds=observed)
    return store.finish([r['sid'] for r in rows])


def calibration_cost(job,args,store):
    h=harness(job,args)
    theta={}
    for text in NATURAL:
        cache=h.model(input_ids=h.ids(text,complete=False),use_cache=True).past_key_values
        for li,(K,V) in enumerate(cache):
            d={'key_max_values':K[0].abs().amax(dim=(1,2)).float().cpu(),'value_max_values':V[0].abs().amax(dim=(1,2)).float().cpu()}
            theta[li]=d if li not in theta else {k:torch.maximum(theta[li][k],v) for k,v in d.items()}
    h.theta={f'layer_{li}':{k:v.tolist() for k,v in d.items()} for li,d in theta.items()}
    h.config_layers=h.L
    write(store.root/'environment.json',h.environment())
    for i in range(5):
        sid='repetition-'+str(i)
        if sid in store.done:continue
        h.reset_key(42)
        synchronize(h.device);start=time.monotonic();secrets,cal=h.calibrate(full_forward=True,capture_evidence=False);synchronize(h.device)
        store.append(sid,status='measured',seconds=time.monotonic()-start,head_count=len(secrets),calibration=cal,
                     phases_included=['probe_construction','public_probe_forward','protected_probe_forward','protection','CPU_calibration'],
                     phases_excluded=['model_loading','dictionary_building','defender_thresholds'])
    return store.finish(['repetition-'+str(i) for i in range(5)])


def paged(job,args,store):
    from get_kvcloak_config import get_kvcloak_config
    from kvcloak import KVCloak
    c=Consumer(job,args);rows=samples_for(job);theta={}
    for text in NATURAL:
        cache=c.forward(c.tokenizer(text).input_ids,keep=1).past_key_values
        for li,(K,V) in enumerate(cache):
            d={'key_max_values':K[0].abs().amax(dim=(1,2)).float().cpu(),'value_max_values':V[0].abs().amax(dim=(1,2)).float().cpu()}
            theta[li]=d if li not in theta else {k:torch.maximum(theta[li][k],v) for k,v in d.items()}
    theta={f'layer_{li}':{k:v.tolist() for k,v in d.items()} for li,d in theta.items()}
    seed_all(42);cfg=get_kvcloak_config(c.L,c.H,c.D,job['block'],theta,1.,1.,2.);cloak=KVCloak(cfg,getattr(torch,job.get('cloak_dtype',job['dtype'])),False,False,True)
    write(store.root/'environment.json',dict(c.metadata(),block_size=job['block'],scope='Native HF block-size/continuation compatibility; not protected-cache vLLM integration'))
    for row in rows:
        if row['sid'] in store.done:continue
        ids=c.tokenizer(row['text']).input_ids;continuation=ids[-6:];history=ids[:-6]
        out=c.forward(history,keep=1);cache=clone(out.past_key_values)
        seed_all(stable_seed(row['sid']));protected=cloak.obfuscate(clone_cache(cache));recovered=cloak.deobfuscate(protected)
        recovered=tuple((a.to(c.device,c.dtype),b.to(c.device,c.dtype)) for a,b in recovered)
        records=[]
        for token in continuation:
            a=c.forward([token],cache=cache,keep=1);b=c.forward([token],cache=recovered,keep=1)
            records.append(logits_diff(a.logits,b.logits));cache=clone(a.past_key_values);recovered=clone(b.past_key_values)
        store.append(row['sid'],status='measured',input_ids=ids,block_size=job['block'],segments=records,
                     scope='Native HF; original block-size compatibility experiment matched without claiming actual engine integration')
    return store.finish([r['sid'] for r in rows])


def vllm_baseline(job,args,store):
    os.environ['VLLM_USE_V1']='0'
    from vllm import LLM,SamplingParams
    from .core import checkpoint,hardware_record
    path=checkpoint(job['model'],args.cache_dir,args.offline)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True);rows=samples_for(job)
    # Preserve the complete frozen prompts instead of silently truncating long
    # items or exhausting a hard-coded number of paged blocks.
    maximum_length=max(1024,max(len(tok(r['text']).input_ids) for r in rows)+16)
    blocks=math.ceil(maximum_length/job['block'])+8
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision('highest')
    engine=LLM(model=str(path),tokenizer=str(path),dtype='bfloat16',tensor_parallel_size=1,
               enforce_eager=True,enable_prefix_caching=True,block_size=job['block'],
               max_model_len=maximum_length,max_num_seqs=1,num_gpu_blocks_override=blocks,gpu_memory_utilization=.5,
               distributed_executor_backend='uni')
    write(store.root/'environment.json',dict(engine='vLLM',version=__import__('vllm').__version__,device=str(args.device),model_dtype='torch.bfloat16',
        block_size=job['block'],model=job['model'],protected_cache_integration=False,
        hardware=hardware_record(args.device),torch=torch.__version__,transformers=__import__('transformers').__version__,
        max_model_len=maximum_length,allocated_cache_blocks=blocks,tf32=False,bf16_reduced_precision_reduction=False,
        evidence_scope='Unmodified engine baseline, separate from native protected-cache block compatibility'))
    params=SamplingParams(temperature=0.,max_tokens=16)
    for row in rows:
        if row['sid'] in store.done:continue
        ids=tok(row['text']).input_ids
        a=engine.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0]
        b=engine.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0]
        store.append(row['sid'],status='measured',input_ids=ids,prediction=a.outputs[0].token_ids,
                     repeated_prediction=b.outputs[0].token_ids,repeated_output_equal=a.outputs[0].token_ids==b.outputs[0].token_ids,
                     protected_cache_integration=False)
    return store.finish([r['sid'] for r in rows])


def run_extended(job,args,store):
    if job['kind']=='ordering':return ordering(job,args,store)
    if job['kind'] in ['spectrum','matched']:return candidate_comparison(job,args,store,arithmetic=job['kind']=='spectrum')
    if job['kind']=='calibration_cost':return calibration_cost(job,args,store)
    if job['kind']=='paged_compat':return paged(job,args,store)
    if job['kind']=='byte_defense':
        from .full_model import byte_validation
        return byte_validation(job,args,store)
    if job['kind']=='sa_functional':
        from .full_model import sa_validation
        return sa_validation(job,args,store)
    if job['kind']=='vllm_baseline':return vllm_baseline(job,args,store)
    if job['kind']=='latency':
        from .latency import latency
        return latency(job,args,store)
    if job['kind']=='official_utility':
        from .utility import run
        return run(job,args,store)
    if job['kind']=='marker_numerics':
        from .numerics import run
        return run(job,args,store)
    if job['kind']=='large_byte':
        from .large_byte import run
        return run(job,args,store)
    raise ValueError('Unregistered scientific job kind '+job['kind'])
