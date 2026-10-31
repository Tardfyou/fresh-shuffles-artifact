"""Validate known-key defensive cache round trips on a frozen public cohort."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

MODELS=('Qwen/Qwen2.5-0.5B','unsloth/Llama-3.2-1B')
DTYPES=('float32','bfloat16')
CONDITIONS=('unpadded','left_padding','right_padding')
SEGMENTS=(2,3,1)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def worker(task):
    model_name,shard,shards,root_string,manifest_string,subset_string=task
    root=Path(root_string);dest=root/model_name.split('/')[-1]/f'part-{shard}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return json.loads((dest/'DONE.json').read_text())
    if (dest/'checks.jsonl').exists():raise ValueError('Partial worker output exists; preserve it and use a fresh run directory.')
    os.environ['VEILKV_NONCE_DB']=str(dest/'freshness.sqlite')
    os.environ['VEILKV_BATTERY_OUT']=str(dest/'context')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/kvcloak'))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    import exp_veilkv_plus as codec
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32,local_files_only=True).eval()
    config=model.config;heads=config.num_key_value_heads;dim=config.hidden_size//config.num_attention_heads
    identity=codec._digests(model,tokenizer)
    ids=json.loads(Path(subset_string).read_text())['lifecycle']
    samples={row['sid']:row for row in rows(Path(manifest_string))}
    assert len(ids)==len(set(ids))==100 and set(ids)<=set(samples)
    selected=[(index,sid) for index,sid in enumerate(ids) if index%shards==shard]
    header=tokenizer.encode('Public system policy: answer the user request clearly and accurately. '*8,add_special_tokens=False)[:16]
    assert len(header)==16
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    environment=dict(model=model_name,model_revision=config._commit_hash,torch=torch.__version__,transformers=transformers.__version__,threads=4,selected_ids=[sid for _,sid in selected])
    (dest/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    written=0;passed=0
    with torch.inference_mode():
        for dtype_index,dtype in enumerate(DTYPES):
            model.to(dtype=getattr(torch,dtype))
            for index,sid in selected:
                body=tokenizer.encode(samples[sid]['text'],add_special_tokens=False)[:134]
                for condition_index,condition in enumerate(CONDITIONS):
                    turn=0
                    try:
                        if len(body)<7:raise ValueError('Insufficient retained source tokens for six continuation tokens.')
                        base=header+body;prefill=len(base)-6
                        if condition=='unpadded':
                            token_rows=[base];mask_rows=[[1]*len(base)]
                        elif condition=='left_padding':
                            token_rows=[[pad]*16+base,header+base]
                            mask_rows=[[0]*16+[1]*len(base),[1]*(len(base)+16)];prefill+=16
                        else:
                            token_rows=[base[:-6]+[pad]*16+base[-6:],header+base]
                            mask_rows=[[1]*(len(base)-6)+[0]*16+[1]*6,[1]*(len(base)+16)];prefill+=16
                        tokens=torch.tensor(token_rows,dtype=torch.long);mask=torch.tensor(mask_rows,dtype=torch.long)
                        positions=(mask.cumsum(-1)-1).clamp_min(0)
                        if dtype_index==0:
                            with (dest/'inputs.jsonl').open('a') as file:
                                file.write(json.dumps(dict(sid=sid,condition=condition,original_model_tokens=len(tokenizer.encode(samples[sid]['text'],add_special_tokens=False)),retained_body_tokens=len(body),prefill_tokens=prefill,input_ids=token_rows,attention_mask=mask_rows,position_ids=positions.tolist()))+'\n')
                        output=model(input_ids=tokens[:,:prefill],attention_mask=mask[:,:prefill],position_ids=positions[:,:prefill],use_cache=True)
                        source=tuple((k.clone(),v.clone()) for k,v in output.past_key_values.to_legacy_cache())
                        cursor=prefill
                        for turn,length in enumerate(SEGMENTS):
                            context=dict(**identity,tokens=tokens[:,:cursor],profile='tee',storage_dtype=getattr(torch,dtype),principal=b'fixed-cohort-validation',nonce=os.urandom(16),epoch_id=1000+index*6+condition_index*2+dtype_index,variant={},bsz=len(token_rows),H=heads,D=dim,true_len=cursor,n_prefix_blocks=1)
                            protected=codec.protect_cache(source,context)
                            recovered=codec.recover_cache(protected,context)
                            byte_equal=all(torch.equal(a,b) for la,lb in zip(source,recovered) for a,b in zip(la,lb))
                            def fresh(values):return DynamicCache.from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in values))
                            end=cursor+length
                            kwargs=dict(input_ids=tokens[:,cursor:end],attention_mask=mask[:,:end],position_ids=positions[:,cursor:end],use_cache=True)
                            plain=model(**kwargs,past_key_values=fresh(source))
                            decoded=model(**kwargs,past_key_values=fresh(recovered))
                            reference=model(input_ids=tokens[:,:end],attention_mask=mask[:,:end],position_ids=positions[:,:end],use_cache=False)
                            delta=(plain.logits-decoded.logits).abs().max().item()
                            baseline_delta=(reference.logits[:,-length:]-plain.logits).abs().max().item()
                            finite=bool(torch.isfinite(decoded.logits).all())
                            event=dict(sid=sid,model=model_name,dtype=dtype,condition=condition,turn=turn,status='measured',prefill_tokens=cursor,added_tokens=length,cache_bytes_exact=byte_equal,protected_vs_plain_logits_max_abs=delta,fresh_vs_plain_cache_logits_max_abs=baseline_delta,finite=finite,passed=byte_equal and finite and delta==0)
                            with (dest/'checks.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
                            written+=1;passed+=int(event['passed'])
                            source=tuple((k.clone(),v.clone()) for k,v in decoded.past_key_values.to_legacy_cache());cursor=end
                        if written%30==0:print(json.dumps(dict(model=model_name,shard=shard,checks=written,passed=passed)),flush=True)
                    except Exception as error:
                        existing=rows(dest/'checks.jsonl')
                        recorded={row['turn'] for row in existing if row['sid']==sid and row['dtype']==dtype and row['condition']==condition}
                        for remaining in range(3):
                            if remaining in recorded:continue
                            event=dict(sid=sid,model=model_name,dtype=dtype,condition=condition,turn=remaining,status='error' if remaining==turn else 'not_run_after_error',error_type=type(error).__name__,error=str(error),passed=False)
                            with (dest/'checks.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
                            written+=1
    summary=dict(model=model_name,shard=shard,checks=written,passed=passed,expected=len(selected)*18,complete=written==len(selected)*18)
    (dest/'DONE.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def aggregate(root, ids):
    data=[row for path in sorted(root.glob('*/part-*/checks.jsonl')) for row in rows(path)]
    expected={(sid,model,dtype,condition,turn) for sid in ids for model in MODELS for dtype in DTYPES for condition in CONDITIONS for turn in range(3)}
    keys=[(row['sid'],row['model'],row['dtype'],row['condition'],row['turn']) for row in data]
    assert len(keys)==len(set(keys)),'Duplicate fixed-cohort observations'
    assert set(keys)<=expected,'Unexpected validation observations'
    grouped=defaultdict(list)
    for row in data:grouped[(row['model'],row['dtype'],row['condition'])].append(row)
    cells=[dict(model=key[0],dtype=key[1],condition=key[2],observations=len(values),unique_ids=len({v['sid'] for v in values}),passed=sum(v['passed'] for v in values),max_codec_logit_error=max((v['protected_vs_plain_logits_max_abs'] for v in values if v['status']=='measured'),default=None),max_fresh_vs_cached_error=max((v['fresh_vs_plain_cache_logits_max_abs'] for v in values if v['status']=='measured'),default=None)) for key,values in sorted(grouped.items())]
    return dict(expected=len(expected),observations=len(data),complete=set(keys)==expected,passed=sum(row['passed'] for row in data),statuses=dict(Counter(row['status'] for row in data)),cells=cells,scope='Known-key, native-model/native-cache dtype byte-profile checks with trusted caller masks and positions. Not a bf16-storage-vs-FP32 accuracy comparison, scheduler integration, attack evaluation or hardware-attestation test.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--subsets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--shards',type=int,default=4)
    args=parser.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise SystemExit('Use a fresh run directory; preserve prior outcomes.')
    ids=json.loads(args.subsets.read_text())['lifecycle'];assert len(ids)==len(set(ids))==100
    protocol=dict(started_at=datetime.now(timezone.utc).isoformat(),models=MODELS,dtypes=DTYPES,conditions=CONDITIONS,segments=SEGMENTS,ids=ids,cohort='Existing frozen lifecycle 100 WildChat IDs; original samples retained',body_token_cap=134,continuation_body_tokens=6,public_policy_prefix_tokens=16,padding_tokens=16,native_model_dtype_matches_storage=True,public_objects='One declared public header/padding block per fixture; fixture epochs isolated, not a cross-request sharing experiment',workers=args.workers,threads_per_worker=4,expected_observations=3600,all_outcomes_retained=True)
    (root/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');shutil.copyfile(__file__,root/'measurement_source.py')
    tasks=[(model,shard,args.shards,str(root.resolve()),str(args.manifest.resolve()),str(args.subsets.resolve())) for model in MODELS for shard in range(args.shards)]
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,task):task for task in tasks}
        for future in as_completed(futures):
            try:print(json.dumps(future.result()),flush=True)
            except Exception as error:errors.append(dict(model=futures[future][0],shard=futures[future][1],error=str(error)))
            (root/'progress.json').write_text(json.dumps(aggregate(root,ids),indent=2)+'\n')
    summary=aggregate(root,ids);summary['worker_errors']=errors;summary['finished_at']=datetime.now(timezone.utc).isoformat()
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
