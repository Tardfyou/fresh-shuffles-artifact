"""Known-key VeilKV software validation on the existing frozen OR cohort."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

DEFAULT_ROOT = Path('results/takeover_20261029/corrected/defense_complete602_20261102')
MODELS = [('Qwen/Qwen2.5-0.5B','Qwen2.5-0.5B'),('unsloth/Llama-3.2-1B','Llama-3.2-1B')]
SHAPES = [('no_pad',None,0),('left8','left',8),('right8','right',8),('left16','left',16),('right16','right',16)]
CONDITIONS = ['chained_reencode','identity_epoch','mid_session','shared_prefix']
SEGMENTS = [(0,2),(2,3),(5,1)]


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def setup(root):
    os.environ.setdefault('HF_HOME','hf_cache')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(root/'source/defense_runtime'))
    import torch
    import exp_veilkv_plus as codec
    from veilkv_runtime import recover_for_model
    from corrected.freshness import KeyService, FreshnessError
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    return torch,codec,recover_for_model,KeyService,FreshnessError,AutoModelForCausalLM,AutoTokenizer,DynamicCache


def prepare(root):
    torch,_,_,_,_,_,Tokenizer,_=setup(root)
    original=root.parent/'defense_handoff_v2_20261031_1319/fixtures'
    rows=[json.loads(l) for l in (root/'inputs.jsonl').open()]
    life=[json.loads(l)['sid'] for l in (root.parent.parent/'defense_handoff_20261030_1600/inputs.jsonl').open()]
    assert len(rows)==602 and len({r['sid'] for r in rows})==602
    assert set(life)<=set(r['sid'] for r in rows)
    compared=0;start=time.monotonic()
    for full,short in MODELS:
        tok=Tokenizer.from_pretrained(full,local_files_only=True)
        exemplar=json.loads((original/f'base_{short}_no_pad_{life[0]}.json').read_text())
        prefix=exemplar['prefix_ids'];assert len(prefix)==16
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for row in rows:
            body=tok(row['text'],return_tensors='pt').input_ids[0][:134].tolist()
            assert len(body)>=6, row['sid']
            history=body[:-6];cont=body[-6:];variants={}
            for name,side,n in SHAPES:
                ids=prefix+([pad]*n if side=='left' else [])+history+([pad]*n if side=='right' else [])
                mask=[1]*16+([0]*n if side=='left' else [])+[1]*len(history)+([0]*n if side=='right' else [])
                pos=(torch.cumsum(torch.tensor(mask),0)-1).tolist() if side=='left' else list(range(len(ids)))
                fix={'sid':row['sid'],'model':short,'ids':ids,'mask':mask,'pos':pos,'cont':cont,'prefix_ids':prefix,'hist_len':len(ids),'cont_len':6,'shape':name}
                old=original/f'base_{short}_{name}_{row["sid"]}.json'
                if old.exists():
                    prior=json.loads(old.read_text())
                    for field in ('ids','mask','pos','cont','prefix_ids','hist_len','cont_len'):
                        assert fix[field]==prior[field],(short,name,row['sid'],field)
                    compared+=1
                variants[name]=fix
            write(root/'fixtures'/short/(row['sid']+'.json'),variants)
    write(root/'FIXTURE_VALIDATION.json',{'inputs':602,'models':2,'shape_fixtures':6020,'overlapping_old_fixtures_equal':compared,'source':'Frozen OR602, containing all lifecycle100','seconds':time.monotonic()-start})
    print(json.dumps({'prepared':True,'overlap_checks':compared}),flush=True)


def worker(args):
    if os.environ.get('DEFENSE_WORKER_CPUSET'):
        os.sched_setaffinity(0,{int(x) for x in os.environ['DEFENSE_WORKER_CPUSET'].split(',')})
    root=args.root;torch,codec,recover_model,KeyService,FreshnessError,Model,Tokenizer,DynamicCache=setup(root)
    torch.set_num_threads(args.threads);torch.manual_seed(20261102)
    torch.set_grad_enabled(False)
    full,short=MODELS[args.model];dtype=getattr(torch,args.dtype)
    work=root/('smoke' if args.smoke else 'full')/f'{short}_{args.dtype}_{args.shard}of{args.shards}'
    work.mkdir(parents=True,exist_ok=True)
    completed_before=set()
    if args.resume_completed:
        previous=json.loads((root/'COMPLETED_BEFORE_RESCHEDULE.json').read_text())
        assert previous['model']==short and previous['dtype']==args.dtype
        completed_before=set(previous['keys'])
    cohort=[json.loads(line) for line in (root/'inputs.jsonl').open()]
    if args.smoke:
        first_two=[json.loads(line)['sid'] for line in (root.parent.parent/'defense_handoff_20261030_1600/inputs.jsonl').open()][:2]
        cohort=[row for row in cohort if row['sid'] in first_two]
    my_ids={row['sid'] for index,row in enumerate(cohort) if index%args.shards==args.shard}
    target_new=sum(f'{sid}|{short}|{args.dtype}|{shape}|{condition}' not in completed_before
                   for sid in my_ids for shape,_,_ in SHAPES for condition in CONDITIONS)
    invocation=uuid.uuid4().hex
    tok=Tokenizer.from_pretrained(full,local_files_only=True)
    model=Model.from_pretrained(full,torch_dtype=torch.float32,local_files_only=True).to(dtype).eval()
    parameters=inspect.signature(model.forward).parameters
    logit_arg='logits_to_keep' if 'logits_to_keep' in parameters else ('num_logits_to_keep' if 'num_logits_to_keep' in parameters else None)
    # Trusted public metadata is prepared once, independently of protected values.
    dig=codec._digests(model,tok);cfg=model.config
    heads=cfg.num_key_value_heads;dim=getattr(cfg,'head_dim',None) or cfg.hidden_size//cfg.num_attention_heads
    svc=KeyService(work/'freshness.sqlite')
    env={'created':stamp(),'invocation':invocation,'model':full,'model_revision':getattr(cfg,'_commit_hash',None),'tokenizer_class':tok.__class__.__name__,'dtype':args.dtype,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'threads':torch.get_num_threads(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'logit_filter_argument':logit_arg,'expected_new_records':target_new,'resumed_completed_records':len(completed_before)}
    write(work/f'environment-{invocation}.json',env)
    def clone(cache):return tuple((k.clone(),v.clone()) for k,v in cache)
    def cache_diff(a,b):
        if len(a)!=len(b):return ['layer_count']
        bad=[]
        for li in range(len(a)):
            for kv in (0,1):
                x,y=a[li][kv],b[li][kv]
                if x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)):
                    bad.append([li,kv])
        return bad
    def logits_diff(a,b):
        if a.shape!=b.shape:return {'exact':False,'shape_equal':False}
        exact=torch.equal(a,b)
        return {'exact':bool(exact),'shape_equal':True,'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'max':0.0 if exact else float((a.float()-b.float()).abs().max()),'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))}
    def forward(ids,mask,pos,cache=None,keep=0):
        kw={'input_ids':ids.unsqueeze(0),'attention_mask':mask.unsqueeze(0),'position_ids':pos.unsqueeze(0),'use_cache':cache is not None or keep==1}
        if cache is not None:kw['past_key_values']=DynamicCache.from_legacy_cache(clone(cache))
        if logit_arg and keep:kw[logit_arg]=keep
        return model(**kw)
    def consume(cache,fix,off,n):
        ids=torch.tensor(fix['cont'][off:off+n]);mask=torch.tensor(fix['mask']+[1]*(off+n))
        pos=torch.arange(fix['pos'][-1]+1+off,fix['pos'][-1]+1+off+n)
        result=forward(ids,mask,pos,cache)
        return clone(result.past_key_values.to_legacy_cache()),result.logits[0]
    def baseline(fix):
        ids=torch.tensor(fix['ids']);mask=torch.tensor(fix['mask']);pos=torch.tensor(fix['pos'])
        pre=forward(ids,mask,pos,keep=1);src=clone(pre.past_key_values.to_legacy_cache());del pre
        states=[src];logits=[];sanity=[];full_delta=[];duplicate=clone(src)
        fullpos=torch.cat((pos,torch.arange(fix['pos'][-1]+1,fix['pos'][-1]+7)))
        complete=forward(torch.tensor(fix['ids']+fix['cont']),torch.tensor(fix['mask']+[1]*6),fullpos,keep=6).logits[0][-6:]
        for off,n in SEGMENTS:
            state,lg=consume(states[-1],fix,off,n);other,ol=consume(duplicate,fix,off,n)
            sanity.append(not cache_diff(state,other) and logits_diff(lg,ol)['exact'])
            full_delta.append(logits_diff(lg,complete[off:off+n]))
            states.append(state);logits.append(lg);duplicate=other
        return {'states':states,'logits':logits,'sanity':sanity,'full_delta':full_delta}
    def context(fix,off,principal,nonce,epoch,npb):
        return dict(**dig,tokens=torch.tensor([fix['ids']+fix['cont'][:off]]),n_prefix_blocks=npb,profile='tee',storage_dtype=dtype,principal=principal,nonce=nonce,epoch_id=epoch,variant={},bsz=1,H=heads,D=dim,true_len=len(fix['ids'])+off,key_service=svc)
    def nonce(key,hop):return hashlib.sha256((invocation+'|'+key+'|'+str(hop)).encode()).digest()[:16]
    def protect(cache,ctx,checks,label,chain_seed=None):
        supplied=clone(cache);before=clone(supplied)
        protected=codec.protect_cache(supplied,ctx,chain_seed=chain_seed)
        checks[label+'_input_unmodified']=not cache_diff(supplied,before)
        recovered=tuple(recover_model(protected,ctx,model_dtype=dtype)) if chain_seed is None else tuple(x.to(dtype) for pair in codec.recover_cache(protected,ctx,chain_seed=chain_seed) for x in pair)
        if chain_seed is not None:recovered=tuple((recovered[i],recovered[i+1]) for i in range(0,len(recovered),2))
        checks[label+'_roundtrip']=not cache_diff(recovered,cache)
        return protected,recovered
    def reject(protected,ctx,name,checks,events,change=None,artifact=None,chain_seed=None):
        altered={**ctx,**(change or {})};target=protected if artifact is None else artifact
        changed=sorted(k for k in ctx if altered[k] is not ctx[k] and (k=='tokens' or altered[k]!=ctx[k]))
        result={'test':name,'changed_context_fields':changed,'artifact_changed':artifact is not None}
        try:codec.recover_cache(target,altered,chain_seed=chain_seed);result['rejected']=False
        except codec.TamperError as err:result.update(rejected=True,error=str(err))
        checks[name]=result['rejected'];events.append(result)
    def negatives(protected,ctx,checks,events):
        reject(protected,ctx,'principal_only',checks,events,{'principal':ctx['principal']+b'-other'})
        reject(protected,ctx,'nonce_only',checks,events,{'nonce':bytes([ctx['nonce'][0]^1])+ctx['nonce'][1:]})
        reject(protected,ctx,'epoch_only',checks,events,{'epoch_id':ctx['epoch_id']+1})
        reject(protected,ctx,'dtype_only',checks,events,{'storage_dtype':torch.bfloat16 if dtype==torch.float32 else torch.float32})
        reject(protected,ctx,'profile_only',checks,events,{'profile':'arithmetic'})
        reject(protected,ctx,'length_only',checks,events,{'true_len':ctx['true_len']-1})
        if ctx['n_prefix_blocks']:
            tokens=ctx['tokens'].clone();tokens[0,0]=(int(tokens[0,0])+1)%cfg.vocab_size
            reject(protected,ctx,'prefix_token_only',checks,events,{'tokens':tokens})
            reject(protected,ctx,'public_policy_only',checks,events,{'n_prefix_blocks':0})
        altered=copy.copy(protected);altered.blocks={k:list(v) for k,v in protected.blocks.items()}
        selected=(0,0);index=ctx['n_prefix_blocks'];entry=dict(altered.blocks[selected][index])
        ct=entry['ct'];assert isinstance(ct,bytes) and ct
        entry['ct']=bytes([ct[0]^1])+ct[1:];altered.blocks[selected][index]=entry
        reject(protected,ctx,'ciphertext_byte',checks,events,artifact=altered)
        altered=copy.copy(protected);altered.blocks={k:list(v) for k,v in protected.blocks.items()}
        altered.blocks[selected]=altered.blocks[selected][:-1]
        reject(protected,ctx,'truncation',checks,events,artifact=altered)
    def segment_record(cache,lg,base,index,checks):
        diff=logits_diff(lg,base['logits'][index]);bad=cache_diff(cache,base['states'][index+1])
        checks[f'seg{index+1}_logits']=diff['exact'] and diff.get('finite',False)
        checks[f'seg{index+1}_cache']=not bad
        return {'segment':index+1,'logits':diff,'cache_mismatch':bad,'full_vs_cached':base['full_delta'][index]}
    def trial(fix,base,condition,key,partner):
        checks={'independent_plain_sanity':all(base['sanity'])};events=[];segments=[];p=b'principal-'+fix['sid'].encode()
        original=clone(base['states'][0])
        if condition=='chained_reencode':
            cache=clone(base['states'][0]);first=None
            for index,(off,n) in enumerate(SEGMENTS):
                ctx=context(fix,off,p,nonce(key,index),3+index,1 if index==0 else 0)
                prot,rec=protect(cache,ctx,checks,f'hop{index}')
                checks[f'hop{index}_plain_cache']=not cache_diff(rec,base['states'][index])
                if index==0:first=(prot,ctx);negatives(prot,ctx,checks,events)
                reject(prot,ctx,f'hop{index}_epoch_only',checks,events,{'epoch_id':4+index})
                cache,lg=consume(rec,fix,off,n);segments.append(segment_record(cache,lg,base,index,checks))
            prot,ctx=first
            try:codec.protect_cache(clone(original),ctx);checks['duplicate_nonce']=False
            except FreshnessError:checks['duplicate_nonce']=True
        elif condition=='identity_epoch':
            ca=context(fix,0,p+b'-A',nonce(key,'A'),3,1)
            pa,ra=protect(base['states'][0],ca,checks,'A')
            cache,lg=consume(ra,fix,0,2);segments.append(segment_record(cache,lg,base,0,checks))
            cb=context(fix,2,p+b'-B',nonce(key,'B'),4,0)
            pb,rb=protect(cache,cb,checks,'B')
            checks['B_equals_plain_boundary']=not cache_diff(rb,base['states'][1])
            negatives(pa,ca,checks,events)
            reject(pb,cb,'B_principal_only',checks,events,{'principal':ca['principal']})
            reject(pa,ca,'A_principal_only',checks,events,{'principal':cb['principal']})
            cache,lg=consume(rb,fix,2,3);segments.append(segment_record(cache,lg,base,1,checks))
        elif condition=='mid_session':
            ctx=context(fix,2,p,nonce(key,0),3,1)
            prot,rec=protect(base['states'][1],ctx,checks,'mid')
            negatives(prot,ctx,checks,events)
            cache,lg=consume(rec,fix,2,3);segments.append(segment_record(cache,lg,base,1,checks))
            cache,lg=consume(cache,fix,5,1);segments.append(segment_record(cache,lg,base,2,checks))
        else:
            ca=context(fix,0,p+b'-A',nonce(key,'A'),3,1)
            pa,_=protect(base['states'][0],ca,checks,'prefixA')
            pf=partner;ids=torch.tensor(pf['ids']);mask=torch.tensor(pf['mask']);pos=torch.tensor(pf['pos'])
            raw=forward(ids,mask,pos,keep=1);bcache=clone(raw.past_key_values.to_legacy_cache());del raw
            cb=context(pf,0,b'principal-'+pf['sid'].encode()+b'-B',nonce(key,'B'),3,1)
            seed={coord:rows[0]['tag'] for coord,rows in pa.blocks.items()}
            pb,rb=protect(bcache,cb,checks,'prefixB',chain_seed=seed)
            checks['public_identity_all_layers']=all(pa.blocks[c][0]['artifact_id']==pb.blocks[c][0]['artifact_id'] for c in pa.blocks)
            checks['public_payload_all_layers']=all(pa.blocks[c][0]['ct']==pb.blocks[c][0]['ct'] and pa.blocks[c][0]['tag']==pb.blocks[c][0]['tag'] for c in pa.blocks)
            # Actually reuse stored prefix objects, retaining B's scoped tail.
            for coord in pb.blocks:pb.blocks[coord][0]=pa.blocks[coord][0]
            restored=tuple((k.to(dtype),v.to(dtype)) for k,v in codec.recover_cache(pb,cb,chain_seed=seed))
            checks['stored_prefix_recovery']=not cache_diff(restored,bcache)
            state,lg=consume(restored,pf,0,2);plain,pl=consume(bcache,pf,0,2)
            d=logits_diff(lg,pl);checks['shared_logits']=d['exact'];checks['shared_cache']=not cache_diff(state,plain)
            segments.append({'segment':1,'logits':d,'cache_mismatch':cache_diff(state,plain)})
            reject(pb,cb,'shared_principal_only',checks,events,{'principal':ca['principal']},chain_seed=seed)
            changed=cb['tokens'].clone();changed[0,0]=(int(changed[0,0])+1)%cfg.vocab_size
            reject(pb,cb,'shared_prefix_only',checks,events,{'tokens':changed},chain_seed=seed)
        checks['plain_source_unmodified']=not cache_diff(original,base['states'][0])
        return {'checks':checks,'segments':segments,'rejects':events,'functional_pass':all(checks.values())}
    rows=[json.loads(l) for l in (root/'inputs.jsonl').open()]
    if args.smoke:
        life=[json.loads(l)['sid'] for l in (root.parent.parent/'defense_handoff_20261030_1600/inputs.jsonl').open()][:2]
        rows=[r for r in rows if r['sid'] in life]
    selected=[(index,row) for index,row in enumerate(rows) if index%args.shards==args.shard]
    path=work/'checks.jsonl';done=set(completed_before)
    if path.exists():done.update(json.loads(line)['key'] for line in path.open() if line.strip())
    started=time.monotonic();written=0
    with path.open('a',buffering=1) as output:
        for index,row in selected:
            fixtures=json.loads((root/'fixtures'/short/(row['sid']+'.json')).read_text())
            partner_row=rows[(index+1)%len(rows)]
            partners=json.loads((root/'fixtures'/short/(partner_row['sid']+'.json')).read_text())
            for shape,_,_ in SHAPES:
                fix=fixtures[shape]
                pending=[c for c in CONDITIONS if f'{row["sid"]}|{short}|{args.dtype}|{shape}|{c}' not in done]
                if not pending:continue
                try:base=baseline(fix)
                except Exception as err:
                    for condition in pending:
                        key=f'{row["sid"]}|{short}|{args.dtype}|{shape}|{condition}'
                        output.write(json.dumps({'key':key,'status':'error','stage':'plain_baseline','error':str(err),'traceback':traceback.format_exc(limit=4)})+'\n')
                    continue
                for condition in pending:
                    key=f'{row["sid"]}|{short}|{args.dtype}|{shape}|{condition}';begin=time.monotonic()
                    record={'key':key,'sid':row['sid'],'model':short,'dtype':args.dtype,'shape':shape,'condition':condition,'fixture_file':f'fixtures/{short}/{row["sid"]}.json','invocation':invocation}
                    try:
                        record.update(trial(fix,base,condition,key,partners[shape]));record['status']='measured'
                    except Exception as err:
                        record.update(status='error',error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc(limit=4))
                    record['seconds']=time.monotonic()-begin
                    output.write(json.dumps(record)+'\n');written+=1
                    if written%20==0:print(json.dumps({'worker':work.name,'records':written,'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    print(json.dumps({'worker_complete':work.name,'new_records':written,'seconds':time.monotonic()-started}),flush=True)


def merge(root,smoke=False):
    ids=[json.loads(l)['sid'] for l in (root/'inputs.jsonl').open()]
    if smoke:ids=[json.loads(l)['sid'] for l in (root.parent.parent/'defense_handoff_20261030_1600/inputs.jsonl').open()][:2]
    expected={f'{sid}|{short}|{dtype}|{shape}|{condition}' for sid in ids for _,short in MODELS for dtype in ('float32','bfloat16') for shape,_,_ in SHAPES for condition in CONDITIONS}
    seen=set();counts=Counter();failed=[];segments=Counter();duplicate=[]
    for file in sorted((root/('smoke' if smoke else 'full')).glob('*/checks.jsonl')):
        for line in file.open():
            row=json.loads(line);key=row['key']
            if key in seen:duplicate.append(key)
            seen.add(key);counts[row['status']]+=1
            if row.get('functional_pass') is True:counts['functional_pass']+=1
            else:failed.append({'key':key,'status':row['status'],'failed_checks':[k for k,v in row.get('checks',{}).items() if not v],'error':row.get('error')})
            segments[row.get('condition','unknown')]+=len(row.get('segments',[]))
    report={'created':stamp(),'smoke':smoke,'expected':len(expected),'records':len(seen),'missing':sorted(expected-seen),'unexpected':sorted(seen-expected),'duplicates':duplicate,'counts':dict(counts),'segment_comparisons':dict(segments),'failures':failed,'complete':seen==expected and not duplicate}
    write(root/('SMOKE_ACCOUNTING.json' if smoke else 'ACCOUNTING.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing','unexpected','failures','duplicates')}),flush=True)
    if smoke and (not report['complete'] or failed):raise SystemExit('Smoke gate failed: inspect retained evidence before full run.')


def launch(args):
    if not args.smoke:
        gate=json.loads((args.root/'SMOKE_ACCOUNTING.json').read_text())
        assert gate['complete'] and not gate['failures']
    processes=[];folder=args.root/('smoke' if args.smoke else 'full');folder.mkdir(exist_ok=True)
    # One allowed logical CPU per physical core; each worker stays on a compact
    # core group before model loading, so first-touch weight allocation is local.
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity=((topology/'physical_package_id').read_text().strip(),(topology/'core_id').read_text().strip())
        cores.setdefault(identity,cpu)
    physical=list(cores.values());worker_index=0;affinity={}
    env=dict(os.environ,HF_HOME='hf_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads))
    for model in range(2):
        for dtype in ('float32','bfloat16'):
            for shard in range(args.shards):
                name=f'{MODELS[model][1]}_{dtype}_{shard}of{args.shards}'
                log=(folder/(name+'.log')).open('a')
                cmd=[sys.executable,str(Path(__file__).absolute()),'worker','--root',str(args.root),'--model',str(model),'--dtype',dtype,'--shard',str(shard),'--shards',str(args.shards),'--threads',str(args.threads)]
                if args.smoke:cmd.append('--smoke')
                worker_env=dict(env)
                if not args.smoke and len(physical)>=4*args.shards*args.threads:
                    cpus=physical[worker_index*args.threads:(worker_index+1)*args.threads]
                    worker_env['DEFENSE_WORKER_CPUSET']=','.join(map(str,cpus));affinity[name]=cpus
                proc=subprocess.Popen(cmd,env=worker_env,stdout=log,stderr=subprocess.STDOUT)
                worker_index+=1
                processes.append((name,proc));print(json.dumps({'launched':name,'pid':proc.pid}),flush=True)
    write(folder/'LAUNCH.json',{'created':stamp(),'workers':{n:p.pid for n,p in processes},'threads':args.threads,'cpu_affinity':affinity})
    exits={name:proc.wait() for name,proc in processes}
    write(folder/'EXIT_CODES.json',exits)
    merge(args.root,args.smoke)
    if any(exits.values()):raise SystemExit('Worker exited nonzero; retained logs require inspection.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','worker','launch','merge'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--model',type=int,choices=(0,1),default=0);parser.add_argument('--dtype',choices=('float32','bfloat16'),default='float32')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=4);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume-completed',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root)
    elif args.action=='worker':worker(args)
    elif args.action=='launch':launch(args)
    else:merge(args.root,args.smoke)
