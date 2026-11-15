"""Native-GPU full-model consumer checks with frozen mask/position fixtures."""
import copy
import inspect
import json
import math
import os
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from . import ROOT
from .core import checkpoint, seed_all, stable_seed, synchronize,load_full,hardware_record
from .worker import write
from .identity import checkpoint_identity
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak
from run_suite import NATURAL
from corrected.freshness import KeyService, FreshnessError
import exp_veilkv_plus as codec

SHAPES=['no_pad','left8','right8','left16','right16']
SEGMENTS=[(0,2),(2,3),(5,1)]


def clone(cache):
    return tuple((k.clone(),v.clone()) for k,v in cache)


def same_cache(a,b):
    return len(a)==len(b) and all(x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for left,right in zip(a,b) for x,y in zip(left,right))


def logits_diff(a,b):
    delta=(a.float()-b.float()).abs()
    worst=int(delta.reshape(-1).argmax())
    return dict(exact=torch.equal(a,b),maximum_absolute_error=float(delta.reshape(-1)[worst]),
                worst_flat_index=worst,worst_observed=float(a.reshape(-1)[worst]),worst_plain=float(b.reshape(-1)[worst]),
                observed_argmax=a.argmax(-1).tolist(),plain_argmax=b.argmax(-1).tolist(),
                argmax_equal=torch.equal(a.argmax(-1),b.argmax(-1)),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


class Consumer:
    def __init__(self,job,args):
        self.job=job;self.device=torch.device(args.device);self.dtype=getattr(torch,job['dtype'])
        torch.set_num_threads(args.threads);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.set_float32_matmul_precision('highest')
        path=checkpoint(job['model'],args.cache_dir,args.offline)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=load_full(path,self.dtype,self.device)
        sig=inspect.signature(self.model.forward).parameters
        self.logit_arg='logits_to_keep' if 'logits_to_keep' in sig else ('num_logits_to_keep' if 'num_logits_to_keep' in sig else None)
        cfg=self.model.config;self.H=cfg.num_key_value_heads;self.D=getattr(cfg,'head_dim',None) or cfg.hidden_size//cfg.num_attention_heads;self.L=cfg.num_hidden_layers
        self.dig=checkpoint_identity(self.model,self.tokenizer,job['model'])
        self.prefixes={}

    def forward(self,ids,mask=None,pos=None,cache=None,keep=None):
        kw=dict(input_ids=torch.as_tensor(ids,dtype=torch.long,device=self.device).reshape(1,-1),use_cache=True)
        if mask is not None:kw['attention_mask']=torch.as_tensor(mask,dtype=torch.long,device=self.device).reshape(1,-1)
        if pos is not None:kw['position_ids']=torch.as_tensor(pos,dtype=torch.long,device=self.device).reshape(1,-1)
        if cache is not None:kw['past_key_values']=DynamicCache.from_legacy_cache(clone(cache))
        if keep and self.logit_arg:kw[self.logit_arg]=keep
        return self.model(**kw)

    def consume(self,cache,fix,off,n):
        ids=fix['cont'][off:off+n];mask=fix['mask']+[1]*(off+n)
        pos=list(range(fix['pos'][-1]+1+off,fix['pos'][-1]+1+off+n))
        out=self.forward(ids,mask,pos,cache)
        return clone(out.past_key_values),out.logits[0]

    def baseline(self,fix):
        if self.job['kind']=='byte_defense':
            key=tuple(fix['ids'][:16])
            if key not in self.prefixes:
                pre=self.forward(fix['ids'][:16],fix['mask'][:16],fix['pos'][:16],keep=1)
                self.prefixes[key]=clone(pre.past_key_values)
            # A stored public prefix is canonical. Recomputing it at different
            # GEMM batch lengths need not give identical floating-point bytes.
            out=self.forward(fix['ids'][16:],fix['mask'],fix['pos'][16:],cache=self.prefixes[key],keep=1)
        else:
            out=self.forward(fix['ids'],fix['mask'],fix['pos'],keep=1)
        cache=clone(out.past_key_values);states=[cache];logits=[];sanity=[]
        duplicate=clone(cache)
        complete=self.forward(fix['ids']+fix['cont'],fix['mask']+[1]*6,
            fix['pos']+list(range(fix['pos'][-1]+1,fix['pos'][-1]+7)),keep=6).logits[0][-6:]
        del out
        for off,n in SEGMENTS:
            state,lg=self.consume(states[-1],fix,off,n)
            other,ol=self.consume(duplicate,fix,off,n)
            sanity.append(same_cache(state,other) and torch.equal(lg,ol))
            states.append(state);logits.append(lg);duplicate=other
        return dict(states=states,logits=logits,sanity=sanity,
                    full_vs_cached=[logits_diff(lg,complete[off:off+n]) for lg,(off,n) in zip(logits,SEGMENTS)])

    def metadata(self):
        return dict(model=self.job['model'],device=str(self.device),hardware=hardware_record(self.device),model_dtype=str(self.dtype),
                    torch=torch.__version__,transformers=__import__('transformers').__version__,
                    attention_implementation='eager',tf32=False,logit_filter_argument=self.logit_arg,
                    bf16_reduced_precision_reduction=False,float32_matmul_precision='highest',
                    cache_layers=self.L,heads=self.H,head_dimension=self.D,
                    codec_format_version=2 if self.job['kind'] in ['byte_defense','latency','large_byte'] else None,
                    identity_derivation='pinned_checkpoint_and_tokenizer_configuration_v1',
                    trusted_crypto_device='cpu',consumer_device=str(self.device),
                    transfers_included_in_end_to_end_timing=True,
                    prefill_execution='canonical_public_prefix_then_private' if self.job['kind']=='byte_defense' else 'native_full_prefill')


def selected(job):
    rows=[json.loads(line) for line in (ROOT/'data/defense_inputs.jsonl').open()]
    if job['kind']=='sa_functional':
        ids=set(json.loads((ROOT/'data/experiment_subsets.json').read_text())[job.get('subset','lifecycle')]);rows=[r for r in rows if r['sid'] in ids]
    if job.get('smoke'):rows=rows[:2]
    return [r for i,r in enumerate(rows) if i%job.get('shards',1)==job.get('shard',0)]


def fixtures(job,sid):
    short='Qwen2.5-0.5B' if 'Qwen2.5-0.5B' in job['model']['repo'] else 'Llama-3.2-1B'
    return json.loads((ROOT/'data/defense_fixtures'/short/(sid+'.json')).read_text())


def byte_validation(job,args,store):
    c=Consumer(job,args);write(store.root/'environment.json',c.metadata())
    service=KeyService(store.root/'freshness.sqlite');invocation=uuid.uuid4().hex
    rows=selected(job);conditions=['chained_reencode','identity_epoch','mid_session','shared_prefix']
    expected=[]
    def context(fix,off,principal,nonce,epoch,npb):
        return dict(**c.dig,tokens=torch.tensor([fix['ids']+fix['cont'][:off]]),n_prefix_blocks=npb,
                    profile='tee',storage_dtype=c.dtype,principal=principal,nonce=nonce,epoch_id=epoch,
                    variant={},bsz=1,H=c.H,D=c.D,true_len=len(fix['ids'])+off,key_service=service)
    def nonce(key,hop):
        import hashlib
        return hashlib.blake2b((invocation+'|'+key+'|'+str(hop)).encode(),digest_size=16).digest()
    def protect(cache,ctx,checks,label,chain_seed=None):
        before=clone(cache)
        # The byte format executes at the trusted CPU boundary; transfers are explicit.
        host=tuple((k.cpu(),v.cpu()) for k,v in cache)
        obj=codec.protect_cache(host,ctx,chain_seed=chain_seed)
        raw=codec.recover_cache(obj,ctx,chain_seed=chain_seed)
        restored=tuple((k.to(device=c.device,dtype=c.dtype),v.to(device=c.device,dtype=c.dtype)) for k,v in raw)
        checks[label+'_input_unmodified']=same_cache(before,cache)
        checks[label+'_roundtrip']=same_cache(restored,cache)
        return obj,restored
    def reject(obj,ctx,checks,label,changed=None,artifact=None,chain_seed=None):
        try:codec.recover_cache(artifact or obj,{**ctx,**(changed or {})},chain_seed=chain_seed);checks[label]=False
        except codec.TamperError:checks[label]=True
    def negatives(obj,ctx,checks):
        for field in ['model_dig','tok_dig','rope']:
            value=ctx[field]
            reject(obj,ctx,checks,field+'_only',{field:bytes([value[0]^1])+value[1:]})
        reject(obj,ctx,checks,'principal_only',{'principal':ctx['principal']+b'-other'})
        reject(obj,ctx,checks,'nonce_only',{'nonce':bytes([ctx['nonce'][0]^1])+ctx['nonce'][1:]})
        reject(obj,ctx,checks,'epoch_only',{'epoch_id':ctx['epoch_id']+1})
        reject(obj,ctx,checks,'dtype_only',{'storage_dtype':torch.bfloat16 if c.dtype==torch.float32 else torch.float32})
        reject(obj,ctx,checks,'profile_only',{'profile':'arithmetic'})
        reject(obj,ctx,checks,'length_only',{'true_len':ctx['true_len']-1})
        if ctx['n_prefix_blocks']:
            ts=ctx['tokens'].clone();ts[0,0]=(int(ts[0,0])+1)%c.model.config.vocab_size
            reject(obj,ctx,checks,'prefix_token_only',{'tokens':ts})
            reject(obj,ctx,checks,'policy_only',{'n_prefix_blocks':0})
        altered=copy.copy(obj);altered.blocks={k:list(v) for k,v in obj.blocks.items()}
        idx=ctx['n_prefix_blocks'];entry=dict(altered.blocks[(0,0)][idx]);ct=entry['ct'];entry['ct']=bytes([ct[0]^1])+ct[1:];altered.blocks[(0,0)][idx]=entry
        reject(obj,ctx,checks,'ciphertext_byte',artifact=altered)
        altered=copy.copy(obj);altered.blocks={k:list(v) for k,v in obj.blocks.items()};altered.blocks[(0,0)]=altered.blocks[(0,0)][:-1]
        reject(obj,ctx,checks,'truncation',artifact=altered)
    for index,row in enumerate(rows):
        fixset=fixtures(job,row['sid']);partner=fixtures(job,rows[(index+1)%len(rows)]['sid'])
        for shape in SHAPES:
            fix=fixset[shape];base=None
            for condition in conditions:
                key='|'.join([row['sid'],shape,condition]);expected.append(key)
                if key in store.done:continue
                if base is None:
                    base=c.baseline(fix)
                    assert all(base['sanity']), 'Independent plain controls differ; this is not a defense outcome'
                try:
                    checks={'duplicate_plain_sanity':all(base['sanity'])};segments=[];principal=b'p-'+row['sid'].encode()
                    original=clone(base['states'][0]);start=time.monotonic()
                    def record(state,lg,k):
                        diff=logits_diff(lg,base['logits'][k]);checks[f'segment{k}_logits']=diff['exact'] and diff['finite'];checks[f'segment{k}_cache']=same_cache(state,base['states'][k+1]);segments.append(dict(segment=k+1,logits=diff,full_vs_cached=base['full_vs_cached'][k]))
                    if condition=='chained_reencode':
                        state=clone(original)
                        for k,(off,n) in enumerate(SEGMENTS):
                            ctx=context(fix,off,principal,nonce(key,k),3+k,1 if k==0 else 0)
                            obj,restored=protect(state,ctx,checks,'hop'+str(k))
                            checks[f'hop{k}_plain_cache']=same_cache(restored,base['states'][k])
                            if k==0:
                                negatives(obj,ctx,checks)
                                try:codec.protect_cache(tuple((a.cpu(),b.cpu()) for a,b in original),ctx);checks['duplicate_nonce']=False
                                except FreshnessError:checks['duplicate_nonce']=True
                            reject(obj,ctx,checks,f'hop{k}_wrong_epoch',{'epoch_id':4+k})
                            state,lg=c.consume(restored,fix,off,n);record(state,lg,k)
                    elif condition=='identity_epoch':
                        ctx=context(fix,0,principal+b'A',nonce(key,'A'),3,1);obj,restored=protect(original,ctx,checks,'A')
                        negatives(obj,ctx,checks);state,lg=c.consume(restored,fix,0,2);record(state,lg,0)
                        cb=context(fix,2,principal+b'B',nonce(key,'B'),4,0);pb,rb=protect(state,cb,checks,'B')
                        negatives(pb,cb,checks)
                        reject(pb,cb,checks,'B_principal_A',{'principal':ctx['principal']})
                        reject(obj,ctx,checks,'A_principal_B',{'principal':cb['principal']})
                        state,lg=c.consume(rb,fix,2,3);record(state,lg,1)
                    elif condition=='mid_session':
                        ctx=context(fix,2,principal,nonce(key,0),3,1);obj,restored=protect(base['states'][1],ctx,checks,'mid')
                        negatives(obj,ctx,checks);state,lg=c.consume(restored,fix,2,3);record(state,lg,1)
                        state,lg=c.consume(state,fix,5,1);record(state,lg,2)
                    else:
                        ca=context(fix,0,principal+b'A',nonce(key,'A'),3,1);pa,_=protect(original,ca,checks,'prefix_A')
                        other=partner[shape]
                        canonical=tuple((a[:,:,:16].clone(),b[:,:,:16].clone()) for a,b in original)
                        pre=c.forward(other['ids'][16:],other['mask'],other['pos'][16:],cache=canonical,keep=1);cache=clone(pre.past_key_values)
                        cb=context(other,0,principal+b'B',nonce(key,'B'),3,1)
                        chain={k:v[0]['tag'] for k,v in pa.blocks.items()}
                        pb,_=protect(cache,cb,checks,'prefix_B',chain_seed=chain)
                        for k in pb.blocks:pb.blocks[k][0]=pa.blocks[k][0]
                        recovered=codec.recover_cache(pb,cb,chain_seed=chain)
                        recovered=tuple((a.to(c.device,c.dtype),b.to(c.device,c.dtype)) for a,b in recovered)
                        checks['stored_prefix_reused']=same_cache(recovered,cache)
                        state,lg=c.consume(recovered,other,0,2);plain,pl=c.consume(cache,other,0,2)
                        diff=logits_diff(lg,pl);checks['shared_logits']=diff['exact'];checks['shared_cache']=same_cache(state,plain);segments.append(dict(segment=1,logits=diff))
                        reject(pb,cb,checks,'shared_principal_only',{'principal':ca['principal']},chain_seed=chain)
                        changed=cb['tokens'].clone();changed[0,0]=(int(changed[0,0])+1)%c.model.config.vocab_size
                        reject(pb,cb,checks,'shared_prefix_only',{'tokens':changed},chain_seed=chain)
                    checks['source_unmodified']=same_cache(original,base['states'][0])
                    synchronize(c.device)
                    store.append(key,status='measured',source_sid=row['sid'],shape=shape,condition=condition,checks=checks,segments=segments,functional_pass=all(checks.values()),seconds=time.monotonic()-start,
                                 expected_consumer_steps={'chained_reencode':3,'identity_epoch':2,'mid_session':2,'shared_prefix':1}[condition],dependent_consumer_steps_not_run=0)
                except torch.cuda.OutOfMemoryError:
                    raise
                except (codec.TamperError, FreshnessError) as exc:
                    checks['positive_path_completed']=False
                    expected_steps={'chained_reencode':3,'identity_epoch':2,'mid_session':2,'shared_prefix':1}[condition]
                    store.append(key,status='measured',source_sid=row['sid'],shape=shape,condition=condition,
                                 functional_pass=False,positive_path_error=type(exc).__name__+': '+str(exc),
                                 failure_stage='protected_byte_consumer',checks=checks,segments=segments,
                                 expected_consumer_steps=expected_steps,
                                 dependent_consumer_steps_not_run=expected_steps-len(segments))
        print(job['id'],index+1,len(rows),flush=True)
    return store.finish(expected)


def sa_validation(job,args,store):
    c=Consumer(job,args);write(store.root/'environment.json',c.metadata())
    theta={}
    for text in NATURAL:
        ids=c.tokenizer(text,return_tensors='pt').input_ids[0].tolist();cache=c.forward(ids,keep=1).past_key_values
        for li,(K,V) in enumerate(cache):
            current={'key_max_values':K[0].abs().amax(dim=(1,2)).float().cpu(),'value_max_values':V[0].abs().amax(dim=(1,2)).float().cpu()}
            theta[li]=current if li not in theta else {k:torch.maximum(theta[li][k],v) for k,v in current.items()}
    theta={f'layer_{li}':{k:v.tolist() for k,v in d.items()} for li,d in theta.items()}
    rows=selected(job);expected=[]
    for row in rows:
        for shape,fix in fixtures(job,row['sid']).items():
            pair=row['sid']+'|'+shape
            seed_all(stable_seed(pair,'sa-known-original'))
            original=get_kvcloak_config(c.L,c.H,c.D,16,theta,1.,1.,2.)
            seed_all(stable_seed(pair,'sa-known-refresh'))
            refresh=get_kvcloak_config(c.L,c.H,c.D,16,theta,1.,1.,2.)
            for li in range(c.L):
                for hh in range(c.H):
                    for kv in (0,1):
                        for field in ['M_angles','M_ratios','M_ratio']:refresh[li][hh][kv][field]=copy.deepcopy(original[li][hh][kv][field])
            assert all(torch.equal(original[li][hh][kv]['M_angles'],refresh[li][hh][kv]['M_angles']) for li in range(c.L) for hh in range(c.H) for kv in (0,1))
            assert any(not torch.equal(original[li][hh][kv]['S'],refresh[li][hh][kv]['S']) for li in range(c.L) for hh in range(c.H) for kv in (0,1))
            base=c.baseline(fix);permutations=None
            assert all(base['sanity']), 'Independent plain controls differ; invalidate the comparison'
            for label,cfg in [('original',original),('refresh',refresh)]:
                key=pair+'|'+label;expected.append(key)
                if key in store.done:
                    if label=='original':permutations=store.done[key]['paired_permutations']
                    continue
                cloak=KVCloak(cfg,c.dtype,False,False,True)
                seed_all(stable_seed(pair,'sa-prep'));cloak._prepare_device_tensors(c.device)
                seed_all(stable_seed(pair,'sa-shuffle'))
                source=clone(base['states'][0]);before=clone(source)
                recorded=[];real_argsort=torch.argsort
                def recorder(*a,**kw):
                    result=real_argsort(*a,**kw);recorded.append(result.detach().cpu().tolist());return result
                from unittest.mock import patch
                with patch.object(torch,'argsort',recorder):protected=cloak.obfuscate(DynamicCache.from_legacy_cache(source))
                if permutations is None:permutations=recorded
                else:assert permutations==recorded,'Paired shuffles differ across S/a conditions'
                recovered=cloak.deobfuscate(protected)
                state=tuple((a.to(c.device,c.dtype),b.to(c.device,c.dtype)) for a,b in recovered)
                segments=[]
                for k,(off,n) in enumerate(SEGMENTS):
                    state,lg=c.consume(state,fix,off,n);diff=logits_diff(lg,base['logits'][k])
                    limit=.001 if c.dtype==torch.float32 else .6
                    segments.append(dict(segment=k+1,logits=diff,functional_pass=diff['finite'] and diff['argmax_equal'] and diff['maximum_absolute_error']<=limit,full_vs_cached=base['full_vs_cached'][k]))
                store.append(key,status='measured',source_sid=row['sid'],shape=shape,condition=label,segments=segments,
                             functional_pass=all(r['functional_pass'] for r in segments),
                             duplicate_plain_sanity=all(base['sanity']),M_fixed=True,paired_permutations=recorded,
                             supplied_cache_mutated=not same_cache(source,before))
        print(job['id'],row['sid'],flush=True)
    return store.finish(expected)
