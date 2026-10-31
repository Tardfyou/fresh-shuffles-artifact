"""Matched known-key cache protection/recovery timing; no attack stages."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();dest=args.out;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():raise SystemExit('Use a fresh output directory.')
    os.environ['VEILKV_NONCE_DB']=str(dest/'freshness.sqlite')
    os.environ['VEILKV_BATTERY_OUT']=str(dest/'context')
    os.environ['VEILKV_THREADS']='4'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/kvcloak'))
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from get_kvcloak_config import get_kvcloak_config
    from kvcloak import KVCloak
    from fusion import fusion
    from run_suite import NATURAL
    import exp_veilkv_plus as V
    from corrected.perf_isolation import wait_for_own_jobs
    context=wait_for_own_jobs(dest);torch.set_num_threads(4)
    name='Qwen/Qwen2.5-0.5B'
    tok=AutoTokenizer.from_pretrained(name,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32,local_files_only=True).eval()
    cfg=model.config;layers=cfg.num_hidden_layers;heads=cfg.num_key_value_heads;dim=cfg.hidden_size//cfg.num_attention_heads
    text='You are a corporate assistant. Answer concisely. My badge PIN is 4831 for verification. '+NATURAL[0]*2
    while tok(text,return_tensors='pt').input_ids.shape[1]<512:text+=' '+NATURAL[0]*8
    ids=tok(text,return_tensors='pt').input_ids[:,:512]
    with torch.inference_mode():
        result=model(input_ids=ids,use_cache=True)
        cache=tuple((k.to(torch.bfloat16).clone(),v.to(torch.bfloat16).clone()) for k,v in result.past_key_values.to_legacy_cache())
        plain_logits=result.logits[:,-1].clone();del result
        theta={}
        for sentence in NATURAL:
            state=model(input_ids=tok(sentence,return_tensors='pt').input_ids,use_cache=True).past_key_values
            for i,(k,v) in enumerate(state):
                entry=theta.setdefault(f'layer_{i}',{'key_max_values':torch.zeros(heads),'value_max_values':torch.zeros(heads)})
                for key,value in [('key_max_values',k),('value_max_values',v)]:entry[key]=torch.maximum(entry[key],value[0].abs().amax(dim=(1,2)))
        theta={li:{k:v.tolist() for k,v in entry.items()} for li,entry in theta.items()}
        torch.manual_seed(42);secret=get_kvcloak_config(layers,heads,dim,16,theta,1.,1.,2.)
        unfused=KVCloak(secret,dtype=torch.bfloat16,fused=False,need_ratio=False,add_a=True)
        t=time.perf_counter();fusion(model,secret);fusion_seconds=time.perf_counter()-t
        result=model(input_ids=ids,use_cache=True)
        fused_cache=tuple((k.to(torch.bfloat16).clone(),v.to(torch.bfloat16).clone()) for k,v in result.past_key_values.to_legacy_cache())
        fusion_error=(result.logits[:,-1]-plain_logits).abs().max().item();del result
        fused=KVCloak(secret,dtype=torch.bfloat16,fused=True,need_ratio=False,add_a=True)
        ident=V._digests(model,tok)
        base=dict(**ident,tokens=ids,profile='tee',storage_dtype=torch.bfloat16,principal=b'perf-review',nonce=os.urandom(16),
                  epoch_id=3,variant={},bsz=1,H=heads,D=dim,true_len=512)
        def fresh(source):return DynamicCache.from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in source))
        def private_ctx(public):return dict(base,n_prefix_blocks=public,nonce=os.urandom(16))
        ctx_share=private_ctx(2);ctx_whole=private_ctx(0)
        encrypted_share=V.protect_cache(cache,ctx_share);encrypted_whole=V.protect_cache(cache,ctx_whole)
        roundtrip_checks=[]
        for protected,ctx in [(encrypted_share,ctx_share),(encrypted_whole,ctx_whole)]:
            decoded=V.recover_cache(protected,ctx)
            assert all(torch.equal(a,b) for layer_a,layer_b in zip(cache,decoded) for a,b in zip(layer_a,layer_b))
            roundtrip_checks.append(dict(values_exact=True,typed_bytes_exact=all(a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for la,lb in zip(cache,decoded) for a,b in zip(la,lb)),returned_dtypes=sorted({str(t.dtype) for layer in decoded for t in layer})))
        cipher_unfused=tuple((k.clone(),v.clone()) for k,v in unfused.obfuscate(fresh(cache)))
        cipher_fused=tuple((k.clone(),v.clone()) for k,v in fused.obfuscate(fresh(fused_cache)))
        checks={}
        for label,cloak,source,cipher in [('unfused',unfused,cache,cipher_unfused),('fused',fused,fused_cache,cipher_fused)]:
            decoded=cloak.deobfuscate(fresh(cipher))
            assert len(decoded)==layers and all(k.shape==(1,heads,512,dim) and v.shape==k.shape for k,v in decoded)
            assert all(torch.isfinite(value).all() for layer in decoded for value in layer)
            # A full-prefix, unmasked joint K/V diagnostic, with keys known to
            # the legitimate recovery operation. This is not a causal-LM test.
            torch.manual_seed(917);errors=[]
            for (k,v),(kr,vr) in zip(source,decoded):
                q=torch.randn(1,heads,1,dim)
                a=torch.softmax(q@k.float().transpose(-1,-2)/(dim**.5),dim=-1)@v.float()
                b=torch.softmax(q@kr.float().transpose(-1,-2)/(dim**.5),dim=-1)@vr.float()
                errors.append((a-b).abs().max().item())
            checks[label]={'finite':True,'unmasked_attention_max_error':max(errors)}
        pairs={
          'protect_tee':(lambda: (cache,private_ctx(2)),lambda x:V.protect_cache(*x)),
          'recover_tee':(lambda:(encrypted_share,ctx_share),lambda x:V.recover_cache(*x)),
          'protect_nosharing_aead':(lambda:(cache,private_ctx(0)),lambda x:V.protect_cache(*x)),
          'recover_nosharing_aead':(lambda:(encrypted_whole,ctx_whole),lambda x:V.recover_cache(*x)),
          'protect_kvcloak':(lambda:fresh(cache),unfused.obfuscate),
          'recover_kvcloak':(lambda:fresh(cipher_unfused),unfused.deobfuscate),
          'protect_kvcloak_fused':(lambda:fresh(fused_cache),fused.obfuscate),
          'recover_kvcloak_fused':(lambda:fresh(cipher_fused),fused.deobfuscate)}
        protocol={'model':name,'model_revision':cfg._commit_hash,'python':platform.python_version(),'torch':torch.__version__,
                  'actual_tokens':512,'public_prefix_tokens':32,'private_tail_tokens':480,'threads':4,'layers':layers,'kv_heads':heads,'head_dim':dim,
                  'storage_dtype':'bfloat16','warmup':20,'repetitions':100,'schedule':'round-wise shuffled, seed 20261030',
                  'fresh_input_preparation_outside_timing':True,'fusion_and_model_forward_outside_timing':True,
                  'trust_scope':'software cache operations; same trusted-process key access; hardware transition costs excluded',
                  'variants':'KV-Cloak fused uses the official weight-fusion function and cache produced by the fused model.'}
        (dest/'protocol.json').write_text(json.dumps(protocol,indent=2));print('Fixture and known-key recovery checks ready.',flush=True)
        for name,(prepare,call) in pairs.items():
            for _ in range(20):call(prepare())
        times={name:[] for name in pairs};rng=random.Random(20261030)
        with (dest/'timings.jsonl').open('w') as file:
            for round_ in range(100):
                order=list(pairs);rng.shuffle(order)
                for name in order:
                    prepare,call=pairs[name];input_=prepare();start=time.perf_counter();output=call(input_);duration=time.perf_counter()-start
                    times[name].append(duration*1000/512)
                    file.write(json.dumps({'round':round_,'operation':name,'seconds':duration,'seconds_per_1000_tokens':duration*1000/512})+'\n');file.flush();del output,input_
                if round_%20==19:print('Timed rounds',round_+1,flush=True)
        data={}
        for name,values in times.items():
            data[name]={q:float(np.quantile(values,p)) for q,p in [('p50',.5),('p95',.95),('p99',.99)]}
            data[name].update(mean=float(np.mean(values)),std=float(np.std(values)),samples_seconds_per_1000_tokens=values)
        report={'protocol':protocol,'functional':checks,'byte_roundtrip_exact':True,'fused_logits_max_error':fusion_error,
                'legacy_byte_roundtrip_exact_meaning':'Value equality only; use typed_roundtrip_checks for dtype/bit equality',
                'typed_roundtrip_checks':roundtrip_checks,
                'one_time_weight_fusion_seconds':fusion_seconds,'perf':data,'host_context':context}
        (dest/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({name:{k:v for k,v in value.items() if k!='samples_seconds_per_1000_tokens'} for name,value in data.items()},indent=2))


if __name__=='__main__':main()
