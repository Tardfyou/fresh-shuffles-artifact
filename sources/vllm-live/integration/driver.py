"""Registered CUDA-only engine integration queue, resumable by complete case IDs."""
import argparse,fcntl,json,os,subprocess,time,uuid
from pathlib import Path
import torch
from . import ROOT
from .records import Records
from .engine import setup,clear,submit,drain,run,compare,compact
from . import scope_policy
import exp_veilkv_plus as codec
from get_kvcloak_config import get_kvcloak_config


def settings():
    s=json.loads((ROOT/'runtime/settings.json').read_text());assert 0<s['gpu_memory_utilization']<=.95;assert 1<=s['cpu_threads']<=32
    return s


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def request(job,key,label,row):return (job+'/'+key+'/'+label,'tenant-'+label,row['defense_tokens'])


def expected_keys(rows,smoke=False):
    ids=[r['sid'] for r in rows];types=json.loads((ROOT/'protocol/PLAN.json').read_text())['tamper_types']
    return ([f'functional|{s}' for s in ids[:2 if smoke else 100]]+
            [f'shared|{ids[2*i]}|{ids[2*i+1]}' for i in range(1 if smoke else 20)]+
            [f'isolation|{s}' for s in ids[:1 if smoke else 10]]+
            [f'tamper|{s}|{t}' for s in ids[:1 if smoke else 5] for t in types])


def shared_check(result,requests,public_tokens,physical_block):
    ids=[r[0] for r in requests]
    cached=[result['outputs'][rid]['cached_tokens'] for rid in ids]
    assert all(n is not None and n>=public_tokens for n in cached),'A claimed shared-prefix condition did not produce real prefix-cache hits'
    simultaneous=[step for step in result['bridge']['steps'] if set(ids)<=set(s['request'] for s in step['sequences'])]
    assert simultaneous,'No simultaneous two-request physical block tables were recorded'
    shared=[];private=[]
    for step in simultaneous:
        data={r['request']:r for r in step['sequences']};a=data[ids[0]];b=data[ids[1]];n=public_tokens//physical_block
        shared.append(a['physical_blocks'][:n]==b['physical_blocks'][:n])
        private.append(not(set(a['physical_blocks'][n:])&set(b['physical_blocks'][n:])))
    return dict(actual_cached_tokens=cached,public_physical_pages_shared=all(shared),private_physical_pages_disjoint=all(private))


def case_run(model,worker,store,rows,smoke,block):
    by={r['sid']:r for r in rows};keys=expected_keys(rows,smoke)
    for key in keys:
        if key in store.done:continue
        kind,sid,*other=key.split('|');row=by[sid];partial={}
        try:
            if kind=='tamper':
                variant=other[0];rid=store.job+'/'+key;requests=[(rid,'tenant-A',row['defense_tokens'])];public=32
                clear(model,worker,'veilkv');warm=[(rid+'-warm','public-prefix',row['defense_tokens'][:32])];submit(model,worker,warm,public_tokens=32,block=block,new_tokens=1);drain(model,worker,warm)
                submit(model,worker,requests,public_tokens=32,block=block)
                model.llm_engine.step()
                if variant=='replay':model.llm_engine.step()
                count=worker.bridge.model_calls;worker.bridge.inject(variant,rid);rejected=False;reason=None
                try:model.llm_engine.step()
                except codec.TamperError as error:rejected=True;reason=str(error)
                after=worker.bridge.model_calls
                model.llm_engine.abort_request(rid)
                store.append(key,sid=sid,kind=kind,input_ids=row['defense_tokens'],public_prefix_tokens=32,variant=variant,rejected=rejected,reason=reason,model_calls_before=count,model_calls_after=after,rejected_before_model_execution=rejected and count==after,bridge=worker.bridge.report())
                continue
            if kind=='functional':requests=[request(store.job,key,'A',row)];public=0;prefix=None
            elif kind=='shared':requests=[request(store.job,key,'A',row),request(store.job,key,'B',by[other[0]])];public=32;prefix=row['defense_tokens'][:32]
            else:requests=[request(store.job,key,'A',row),request(store.job,key,'B',row)];public=32;prefix=row['defense_tokens'][:32]
            baseline=run(model,worker,requests,public_tokens=public,block=block,mode='plain',prefix=prefix)
            partial['baseline']=compact(baseline)
            duplicate=run(model,worker,requests,public_tokens=public,block=block,mode='plain',prefix=prefix)
            partial['duplicate_plain']=compact(duplicate)
            sanity=compare(baseline,duplicate,requests);assert sanity['logits_exact'] and sanity['tokens_equal'],'Duplicate-plain control failed; do not count this as a defense result'
            actual=run(model,worker,requests,public_tokens=public,block=block,mode='veilkv',prefix=prefix,epoch_rotate=kind=='functional' and row['ordinal']<10)
            partial['protected']=compact(actual)
            comparison=compare(baseline,actual,requests);bridge=actual['bridge']
            checks=dict(plain_duplicate=True,output_tokens_equal=comparison['tokens_equal'],logits_exact=comparison['logits_exact'],all_protected_roundtrips_exact=all(s['roundtrip_bytes_exact'] for s in bridge['steps']),gpu_pages_zeroed_between_calls=all(s['zeroized_at_rest'] for s in bridge['steps']),actual_restored_objects=bridge['restored_objects']>0,actual_model_execution=bridge['model_calls']>=6)
            sharing=None
            if public:
                sharing=shared_check(actual,requests,public,block);checks.update({k:v for k,v in sharing.items() if isinstance(v,bool)});checks['stored_prefix_reused']=bridge['stored_prefix_reuses']>0
                a,b=[actual['payloads'][r[0]] for r in requests];checks['public_cipher_object_equal']=a['public']==b['public'];checks['private_cipher_objects_differ']=a['private']!=b['private']
            if kind=='functional' and row['ordinal']<10:checks['real_epoch_transition']={s['epoch'] for s in bridge['steps']}=={3,4}
            store.append(key,sid=sid,kind=kind,inputs={r[0]:r[2] for r in requests},baseline=compact(baseline),duplicate_plain=compact(duplicate),protected=compact(actual),comparison=comparison,sharing=sharing,checks=checks,functional_pass=all(checks.values()))
            print(key,'pass',all(checks.values()),flush=True)
        except Exception as exc:store.error(key,exc,partial);raise
    return store.finish(keys)


def configure_cloak(model,worker,tok,job):
    texts=json.loads((ROOT/'data/threshold_texts.json').read_text());requests=[(job+'/threshold-'+str(i),'public-calibration',tok(text).input_ids) for i,text in enumerate(texts)]
    clear(model,worker,'plain');submit(model,worker,requests,public_tokens=0,block=16,new_tokens=1,full_capture=True);drain(model,worker,requests)
    maxima={};captures=worker.bridge.full_capture
    for rid,_,_ in requests:
        for li,pair in enumerate(captures[rid]):
            current={key:x[0].abs().amax(dim=(1,2)).float() for key,x in zip(['key_max_values','value_max_values'],pair)}
            maxima[li]=current if li not in maxima else {k:torch.maximum(maxima[li][k],v) for k,v in current.items()}
    theta={f'layer_{li}':{k:v.tolist() for k,v in d.items()} for li,d in maxima.items()};layout=worker.bridge.layout
    torch.manual_seed(42);cfg=get_kvcloak_config(layout['layers'],layout['heads'],layout['head_dim'],16,theta,1.,1.,2.)
    worker.bridge.configure_cloak(cfg)
    return dict(threshold_texts=texts,theta=theta,configuration_seed=42,device_preparation_seed=1234,official_obfuscation_dtype='float32',protected_storage_dtype='float32',legal_restore='Recorded one-time permutations and actual matrix inverses, withholding both from the attacker')


def worker_job(args):
    from .lock import verify
    verify();s=settings();output=Path(s['output_dir']);output=output if output.is_absolute() else ROOT/output
    spec=next(m for m in json.loads((ROOT/'protocol/models.json').read_text()) if m['tag']==args.model)
    job=('smoke_' if args.smoke else '')+('attack_' if args.attack else 'defense_')+args.model+'_b'+str(args.block)
    store=Records(output/job,job);inv=store.folder/'invocations'/store.invocation;inv.mkdir(parents=True)
    model,worker,tok,env=setup(spec,args.block,s,inv);env.update(invocation=store.invocation,science_version=json.loads((ROOT/'SCIENTIFIC_LOCK.json').read_text())['version']);write(inv/'environment.json',env)
    rows=[json.loads(x) for x in (ROOT/'data'/(args.model+'.jsonl')).open()]
    try:
        if args.attack:
            assert args.model=='qwen25_05b' and args.block==16
            calibration=configure_cloak(model,worker,tok,job);write(inv/'defender-evaluation-only.json',calibration)
            from .attack import run_attack
            snapshot=model.llm_engine.model_config.model
            result=run_attack(model,worker,tok,spec,rows,store,args.smoke,snapshot)
        else:result=case_run(model,worker,store,rows,args.smoke,args.block)
        print(json.dumps(result),flush=True)
    finally:store.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','all','defense','attack'],default='smoke');p.add_argument('--worker',action='store_true');p.add_argument('--attack-worker',dest='attack',action='store_true');p.add_argument('--model');p.add_argument('--block',type=int,default=16);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.worker:return worker_job(args)
    from .lock import verify
    verify();s=settings();output=Path(s['output_dir']);output=output if output.is_absolute() else ROOT/output;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.phase!='smoke':
        gate=json.loads((output/'SMOKE_GATE.json').read_text())
        assert gate['passed'] and gate['source_lock']==verify(),'Complete and accept this issued version of the CUDA smoke queue first'
    models=json.loads((ROOT/'protocol/models.json').read_text());jobs=[]
    if args.phase in ['smoke','all','defense']:
        jobs += [(m['tag'],b,False) for m in models for b in [16,32]]
    if args.phase in ['smoke','all','attack']:jobs.append(('qwen25_05b',16,True))
    reports=[]
    for tag,block,attack in jobs:
        jid=('smoke_' if args.phase=='smoke' else '')+('attack_' if attack else 'defense_')+tag+'_b'+str(block)
        if (output/jid/'summary.json').exists():
            from .verify import verify_job
            reports.append(verify_job(output/jid,tag,block,attack,args.phase=='smoke'));continue
        # Avoid competing with the original GPU queue or other users' compute.
        while True:
            p=subprocess.run(['nvidia-smi','--id='+str(s['gpu']),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            busy=[x for x in p.stdout.splitlines() if x.strip().isdigit()]
            free=subprocess.check_output(['nvidia-smi','--id='+str(s['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if (not busy or s['allow_external_gpu_processes']) and float(free.strip())/1024>=s['minimum_free_gib']:break
            print('Waiting for the selected GPU; no external process is stopped',flush=True);time.sleep(15)
        cmd=[__import__('sys').executable,'-u','-m','integration.driver','--worker','--model',tag,'--block',str(block)]
        if attack:cmd+=['--attack-worker']
        if args.phase=='smoke':cmd+=['--smoke']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(s['gpu']),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
        jid=('smoke_' if args.phase=='smoke' else '')+('attack_' if attack else 'defense_')+tag+'_b'+str(block)
        with (output/(jid+'-'+str(time.time_ns())+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,'Worker failed; retain all records and inspect '+jid
        from .verify import verify_job
        reports.append(verify_job(output/jid,tag,block,attack,args.phase=='smoke'))
    if args.phase=='smoke':write(output/'SMOKE_GATE.json',dict(passed=True,source_lock=verify(),jobs=reports))
    write(output/('QUEUE_'+args.phase+'.json'),dict(complete=True,jobs=reports))


if __name__=='__main__':main()
