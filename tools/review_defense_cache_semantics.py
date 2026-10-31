"""Known-key VeilKV cache round trips with masks and segmented continuation."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise SystemExit('Use a fresh output directory.')
    os.environ['VEILKV_NONCE_DB']=str(out/'freshness.sqlite')
    os.environ['VEILKV_BATTERY_OUT']=str(out/'context')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/kvcloak'))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    import exp_veilkv_plus as codec
    torch.set_num_threads(4);torch.manual_seed(20261030)
    name='Qwen/Qwen2.5-0.5B'
    tok=AutoTokenizer.from_pretrained(name,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32,local_files_only=True).eval()
    ident=codec._digests(model,tok)
    config=model.config;heads=config.num_key_value_heads;dim=config.hidden_size//config.num_attention_heads
    protocol=dict(model=name,model_revision=config._commit_hash,python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,dtype='float32',threads=4,seed=20261030,prefix_tokens=64,segments=[2,3,1],cases=['unpadded','left_padding','right_padding'],sharing_public_tokens=16,profile='tee byte storage; software key-release stub',full_forward_tolerance=1e-4,scope='Known-key defensive cache operations with caller-supplied trusted masks and positions; synthetic token IDs, not natural-prompt accuracy or an attack experiment')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    shutil.copyfile(__file__,out/'measurement_source.py')
    events=[]
    with torch.inference_mode():
        for case_index,case in enumerate(protocol['cases']):
            batch=1 if case=='unpadded' else 2
            tokens=torch.randint(256,config.vocab_size,(batch,70))
            mask=torch.ones((batch,70),dtype=torch.long)
            if case=='left_padding':mask[0,:16]=0;tokens[0,:16]=tok.pad_token_id
            if case=='right_padding':mask[0,48:64]=0;tokens[0,48:64]=tok.pad_token_id
            positions=(mask.cumsum(-1)-1).clamp_min(0)
            with (out/'inputs.jsonl').open('a') as f:
                f.write(json.dumps(dict(case=case,input_ids=tokens.tolist(),attention_mask=mask.tolist(),position_ids=positions.tolist()))+'\n')
            base=model(input_ids=tokens[:,:64],attention_mask=mask[:,:64],position_ids=positions[:,:64],use_cache=True)
            source=tuple((k.clone(),v.clone()) for k,v in base.past_key_values.to_legacy_cache())
            cursor=64
            for turn,length in enumerate(protocol['segments']):
                ctx=dict(**ident,tokens=tokens[:,:cursor],profile='tee',storage_dtype=torch.float32,principal=f'semantics-{case}'.encode(),nonce=os.urandom(16),epoch_id=case_index+1,variant={},bsz=batch,H=heads,D=dim,true_len=cursor,n_prefix_blocks=1)
                protected=codec.protect_cache(source,ctx)
                recovered=codec.recover_cache(protected,ctx)
                equal=all(torch.equal(a,b) for layer_a,layer_b in zip(source,recovered) for a,b in zip(layer_a,layer_b))
                def fresh(cache):return DynamicCache.from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in cache))
                end=cursor+length
                kwargs=dict(input_ids=tokens[:,cursor:end],attention_mask=mask[:,:end],position_ids=positions[:,cursor:end],use_cache=True)
                plain=model(**kwargs,past_key_values=fresh(source))
                decoded=model(**kwargs,past_key_values=fresh(recovered))
                reference=model(input_ids=tokens[:,:end],attention_mask=mask[:,:end],position_ids=positions[:,:end],use_cache=False)
                delta=(plain.logits-decoded.logits).abs().max().item()
                full_delta=(reference.logits[:,-length:]-decoded.logits).abs().max().item()
                finite=bool(torch.isfinite(decoded.logits).all())
                event=dict(case=case,turn=turn,prefix_tokens=cursor,added_tokens=length,batch=batch,cache_bytes_exact=equal,continuation_logits_max_error=delta,full_forward_logits_max_error=full_delta,finite=finite,passed=equal and finite and delta==0 and full_delta<=protocol['full_forward_tolerance'])
                events.append(event)
                with (out/'checks.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                source=tuple((k.clone(),v.clone()) for k,v in decoded.past_key_values.to_legacy_cache());cursor=end
                print(json.dumps(event),flush=True)
    result=dict(protocol=protocol,cases=3,continuation_checks=len(events),checks=events,all_passed=all(e['passed'] for e in events),cache_roundtrip_exact=all(e['cache_bytes_exact'] for e in events),continuation_logits_max_error=max(e['continuation_logits_max_error'] for e in events),full_forward_logits_max_error=max(e['full_forward_logits_max_error'] for e in events),scope_excludes=['KV-Cloak row-secret rotation','bf16-vs-fp32 equivalence','within-request adversarial inputs','scheduler-level prefix sharing','rollback-resistant state','device transfer','hardware key release'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
