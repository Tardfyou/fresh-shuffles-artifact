"""Frozen recovery algorithm on the live KV-Cloak offload observer's Values."""
from collections import Counter
import json
import torch
from transformers import AutoModelForCausalLM
from . import ROOT
from .engine import clear,submit,drain,run,compact,compare
from attack_lib import cluster_states,recover_secrets
from frozen_decoder import Decoder


@torch.inference_mode()
def dictionary(snapshot,heads,dim):
    # Public checkpoint only: no defender/fused weights and no target KV inputs.
    model=AutoModelForCausalLM.from_pretrained(snapshot,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True).eval()
    layer=model.model.layers[0];embedding=model.get_input_embeddings().weight;projection=layer.self_attn.v_proj
    chunks=[]
    for off in range(0,len(embedding),2048):chunks.append(projection(layer.input_layernorm(embedding[off:off+2048])).float().reshape(-1,heads,dim))
    result=torch.cat(chunks);del model;torch.cuda.empty_cache();return result


def run_attack(model,worker,tok,spec,rows,store,smoke,snapshot):
    probe=json.loads((ROOT/'data/probe_qwen25_05b.json').read_text());assert tok(probe['raw_text']).input_ids==probe['input_ids']
    prefix=probe['prefix_tokens'];request=store.job+'/own-probe';clear(model,worker,'kvcloak')
    req=[(request,'attacker-own-probe',probe['input_ids'])];submit(model,worker,req,public_tokens=0,block=16,new_tokens=1,capture=True);drain(model,worker,req)
    observed=worker.bridge.observations[request];protected=observed['protected_value']
    assert observed['length']==len(probe['input_ids'])
    (store.folder/'invocations'/store.invocation/'raw-protected-own-probe.json').write_text(json.dumps(dict(input_ids=probe['input_ids'],protected_value=protected.tolist()))+'\n')
    heads=worker.bridge.layout['heads'];dim=worker.bridge.layout['head_dim'];public=dictionary(snapshot,heads,dim);decoder=Decoder(public)
    clear(model,worker,'plain');submit(model,worker,req,public_tokens=0,block=16,new_tokens=1,full_capture=True);drain(model,worker,req)
    own_engine_value=worker.bridge.full_capture[request][0][1][0].float().cpu()
    first=(prefix+15)//16;last=len(probe['input_ids'])//16;secrets={};cal=[]
    known=public[torch.tensor(probe['input_ids'],device=public.device)].permute(1,0,2).cpu()
    dictionary_error=float((own_engine_value-known).norm()/own_engine_value.norm().clamp_min(1e-20))
    assert dictionary_error<.02,'Public dictionary and own real-engine Value layout disagree'
    for h in range(heads):
        blocks=protected[h].reshape(-1,16,dim)[first:last];n,centroids,intra,inter,labels=cluster_states(blocks)
        row=dict(head=h,states=n,complete=n==16,intra=intra,inter=inter,centroids=centroids.tolist(),cluster_labels=labels.tolist())
        if n==16:
            sec=recover_secrets(centroids,known[h,first*16:last*16],16);secrets[h]={k:v.to(public.device) if torch.is_tensor(v) else v for k,v in sec.items()}
            row['estimated_parameters']={k:v.tolist() if torch.is_tensor(v) else v for k,v in sec.items()}
        cal.append(row)
    caldata=dict(input_ids=probe['input_ids'],raw_character=probe['raw_character'],raw_characters=probe['raw_characters'],prefix_tokens=prefix,protected_first_value=protected.tolist(),heads=cal,known_probe_value_from_public_checkpoint=known[:,first*16:(first+1)*16].tolist(),own_probe_dictionary_relative_error=dictionary_error,information_boundary='Recovery receives protected engine Values and a public-checkpoint dictionary; held defender matrices and permutations are not passed to it')
    (store.folder/'invocations'/store.invocation/'attack-calibration.json').write_text(json.dumps(caldata,indent=2)+'\n')
    selected=rows[:2] if smoke else rows;expected=[]
    for row in selected:
        key='attack|'+row['sid'];expected.append(key)
        if key in store.done:continue
        rid=store.job+'/'+row['sid'];requests=[(rid,'target-tenant',row['attack_tokens'])]
        partial={}
        try:
            plain=run(model,worker,requests,public_tokens=0,block=16,mode='plain')
            partial['plain']=compact(plain)
            duplicate=run(model,worker,requests,public_tokens=0,block=16,mode='plain');sanity=compare(plain,duplicate,requests)
            assert sanity['logits_exact'] and sanity['tokens_equal'],'Independent plain engine paths disagree'
            clear(model,worker,'kvcloak');submit(model,worker,requests,public_tokens=0,block=16,capture=True);out,frames=drain(model,worker,requests)
            partial['protected_engine']=compact(dict(outputs=out,frames=frames,bridge=worker.bridge.report()))
            c=worker.bridge.observations[rid]['protected_value'];partial['protected_first_value']=c.tolist();truth=row['attack_tokens'][:len(row['attack_tokens'])//16*16];scores=[]
            for h in range(heads):
                blocks=c[h,:len(truth)].to(public.device).reshape(-1,16,dim)
                predictions=decoder.decode_blocks(blocks,secrets[h],h) if h in secrets else [[] for _ in blocks]
                counts=[sum((Counter(p)&Counter(truth[k*16:k*16+16])).values()) for k,p in enumerate(predictions)]
                scores.append(dict(head=h,bags=predictions,block_correct=counts,correct=sum(counts),total=len(truth),calibration_complete=h in secrets))
            actual=dict(outputs=out,frames=frames,bridge=worker.bridge.report());function=compare(plain,actual,requests)
            store.append(key,sid=row['sid'],kind='attack',input_ids=row['attack_tokens'],truth=truth,protected_first_value=c.tolist(),heads=scores,correct=sum(h['correct'] for h in scores),total=sum(h['total'] for h in scores),known_key_consumer_comparison=function,plain=compact(plain),protected=compact(actual),plain_duplicate_sanity=True)
            print(key,store.done[key]['correct'],store.done[key]['total'],flush=True)
        except Exception as exc:store.error(key,exc,partial);raise
    return store.finish(expected)
