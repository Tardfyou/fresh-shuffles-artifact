"""Independently inspect all stratified timing observations; no models or APIs."""
import argparse,collections,json,math,statistics,sys,zipfile
from pathlib import Path
from summarize_gpu_results import records
RUN='latency-lengths-1k-20261115'
def read(p):return json.loads(p.read_text())
def percentile(v,q):
 a=sorted(v);x=(len(a)-1)*q;lo=int(x);hi=math.ceil(x);return a[lo]+(a[hi]-a[lo])*(x-lo)
def check(root):
 src=root/'sources/latency-lengths-1k';raw=root/'results/gpu/accepted'/RUN;summaries=root/'results/gpu/summaries'/RUN
 manifest=read(src/'SCIENTIFIC_LOCK.json')
 with zipfile.ZipFile(src/manifest['baseline_archive']) as z:
  for n in manifest['files']:assert (src/n).read_bytes()==z.read(n),n
 saved=read(summaries/'LATENCY_SUMMARY.json');formal=smoke=0;ops=0;details=[];main={x['sid'] for x in map(json.loads,(src/'data/samples_main_public.jsonl').open())};assert len(main)==3000
 inputs={tag:{x['sid']:x['input_ids'] for x in map(json.loads,(src/f'data/latency_inputs_{tag}.jsonl').open())} for tag in ['llama31_8b','qwen25_05b']}
 methods=['kvcloak_unfused','kvcloak_fused','veilkv_prefix','aead_no_sharing'];jobfiles=['jobs.json','smoke_jobs.json']
 for file in jobfiles:
  for job in read(src/'protocol'/file):
   folder=raw/job['id'];assert read(folder/'job.json')==job;n=job['repetitions'];schedule=read(src/job['input_schedule']);draw=schedule['draws'][:n];lo,hi=job['length_group'];tok=inputs[job['model']['tag']];groups=collections.defaultdict(list);seen=set();envs={}
   expected={f'{m}|{o}|{i}' for m in methods for o in ['protect','recover'] for i in range(n)}|{'prefill|'+str(i) for i in range(n)}
   for x in records(folder):
    if x['status']!='measured':continue
    assert x['sid'] not in seen;seen.add(x['sid']);assert x['job_id']==job['id'] and x['science_version']==manifest['version']
    rep=x['repetition'];assert 0<=rep<n and x['source_sid']==draw[rep] and x['tokens']==len(tok[draw[rep]]) and lo<=x['tokens']<hi
    assert x['length_group']==job['length_group'] and math.isfinite(x['seconds']) and x['seconds']>0
    inv=x['invocation_id']
    if inv not in envs:
     e=read(folder/'invocations'/inv/'environment.json');assert read(folder/'invocations'/inv/'job.json')==job
     assert e['device']=='cuda:0' and e['hardware']['name']=='NVIDIA A100-SXM4-80GB' and e['model']==job['model'] and e['model_dtype']=='torch.'+job['dtype']
     assert e['host_threads']==16 and e['torch']=='2.6.0+cu124' and e['transformers']=='4.51.3';envs[inv]=e
    if x['operation']=='prefill':assert x['method']=='plain_model' and x['sid']=='prefill|'+str(rep)
    else:
     s=job['cache_shape'];b=2*s['layers']*s['heads']*s['head_dimension']*s['element_bytes']*x['tokens'];assert b==x['cache_bytes']
     assert x['sid']==f'{x["method"]}|{x["operation"]}|{rep}'
     assert math.isclose(x['seconds_per_1000_tokens'],x['seconds']*1000/x['tokens'],rel_tol=1e-12)
     assert math.isclose(x['milliseconds_per_GiB'],x['seconds']*1000/(b/(1<<30)),rel_tol=1e-12)
     assert x['public_prefix_tokens']==(32 if x['method']=='veilkv_prefix' else 0)
    groups[x['method']+'|'+x['operation']].append(x)
   assert seen==expected and len(groups)==9 and all(len(v)==n for v in groups.values())
   assert read(folder/'summary.json')['measured']==len(seen)
   mapping=[{x['repetition']:(x['source_sid'],x['tokens']) for x in v} for v in groups.values()];assert all(v==mapping[0] for v in mapping)
   if job.get('smoke'):smoke+=len(seen)
   else:
    formal+=len(seen)
    for k,v in groups.items():
     s=saved[job['id']]['operations'][k];ms=[x['seconds']*1000 for x in v]
     assert s['timing_calls']==1000 and s['distinct_inputs']==len({x['source_sid'] for x in v})
     for q in [.5,.95,.99]:assert math.isclose(s['p'+str(round(100*q))+'_ms'],percentile(ms,q),rel_tol=1e-12)
     if v[0]['operation']!='prefill':
      assert s['p50_seconds_per1000_tokens']==statistics.median(x['seconds_per_1000_tokens'] for x in v)
      assert s['p50_milliseconds_per_GiB']==statistics.median(x['milliseconds_per_GiB'] for x in v)
     ops+=1
   details.append(dict(job=job['id'],rows=len(seen),pass_=True))
 assert formal==180000 and smoke==360 and ops==180
 return dict(pass_saved_records=True,formal_jobs=20,startup_jobs=20,formal_records=formal,startup_records=smoke,operation_series_checked=ops,actual_source_lock_files=len(manifest['files']),length_and_byte_denominators_checked=True,all_percentiles_recomputed=True,paired_inputs_across_methods_verified=True,model_or_api_calls=0,jobs=details)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--report',type=Path);a=p.parse_args();r=check(a.root.resolve())
 if a.report:a.report.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r,indent=2))
