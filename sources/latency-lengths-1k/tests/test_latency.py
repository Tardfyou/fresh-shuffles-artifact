import json,math,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from runner import ROOT
from runner.latency_plan import inputs_for,expected_keys,check_record
from runner.latency_summary import percentile,summary
from runner import latency as module

class ProtocolTests(unittest.TestCase):
 def test_all_counts_and_sampling(self):
  jobs=json.loads((ROOT/'protocol/jobs.json').read_text());smoke=json.loads((ROOT/'protocol/smoke_jobs.json').read_text())
  self.assertEqual(len(jobs),20);self.assertEqual(sum(len(expected_keys(j)) for j in jobs),180000);self.assertEqual(sum(len(expected_keys(j)) for j in smoke),360)
  for j in jobs:
   draws,warm=inputs_for(j);self.assertEqual(len(draws),1000);self.assertEqual(len(warm),20)
   p=json.loads((ROOT/j['input_schedule']).read_text());rng=random.Random(p['seed']);self.assertEqual([rng.choice(p['eligible_ids']) for _ in range(1000)],p['draws']);self.assertEqual([rng.choice(p['eligible_ids']) for _ in range(20)],p['warmups'])
   self.assertTrue(all(j['length_group'][0]<=len(x['input_ids'])<j['length_group'][1] for x in draws))
  for tag in ['llama31_8b','qwen25_05b']:
   for group in [[32,64],[64,128],[128,256],[256,512],[512,768]]:
    matching=[j for j in jobs if j['model']['tag']==tag and j['length_group']==group]
    self.assertEqual(len({j['input_schedule'] for j in matching}),1)
 def test_bad_records_rejected(self):
  j=json.loads((ROOT/'protocol/jobs.json').read_text())[0];draws,_=inputs_for(j);n=len(draws[0]['input_ids']);s=j['cache_shape'];b=2*s['layers']*s['heads']*s['head_dimension']*s['element_bytes']*n
  r=dict(sid='kvcloak_unfused|protect|0',source_sid=draws[0]['sid'],repetition=0,tokens=n,length_group=j['length_group'],method='kvcloak_unfused',operation='protect',seconds=.03,seconds_per_1000_tokens=.03*1000/n,milliseconds_per_GiB=.03*1000/(b/(1<<30)),cache_bytes=b,public_prefix_tokens=0)
  check_record(j,r)
  for key,value in [('tokens',n+1),('source_sid','other'),('seconds_per_1000_tokens',.03*1000/512),('seconds',float('nan')),('cache_bytes',b+1)]:
   with self.assertRaises(AssertionError):check_record(j,dict(r,**{key:value}))
 def test_summary_actual_lengths(self):
  rows=[dict(seconds=s,source_sid='s'+str(i%2),tokens=n,operation='protect',seconds_per_1000_tokens=s*1000/n,milliseconds_per_GiB=2.) for i,(s,n) in enumerate([(.1,32),(.2,64),(.4,128),(.8,256)])]
  d=summary(rows);self.assertEqual(d['timing_calls'],4);self.assertEqual(d['distinct_inputs'],2);self.assertEqual(d['p50_ms'],300.);self.assertAlmostEqual(d['p50_seconds_per1000_tokens'],3.125);self.assertAlmostEqual(percentile([1.,2.,3.,4.],.95),3.85)

class ToyConsumer:
 def __init__(self,job,args):
  self.device=torch.device('cpu');self.dtype=getattr(torch,job['dtype']);self.L=1;self.H=2;self.D=8;self.prefixes={};self.dig=dict(model_dig=b'm'*32,tok_dig=b't'*32,rope=b'r'*32)
 def tokenizer(self,text):return SimpleNamespace(input_ids=list(range(1,int(text)+1)))
 def forward(self,ids,keep=None,cache=None):
  x=torch.tensor(ids,dtype=torch.float32).reshape(1,1,-1,1).expand(1,self.H,-1,self.D).to(self.dtype)/100
  k=x.contiguous();v=(x+.125).contiguous()
  if cache is not None:k=torch.cat([cache[0][0],k],dim=2);v=torch.cat([cache[0][1],v],dim=2)
  return SimpleNamespace(past_key_values=((k,v),))
 def metadata(self):return dict(device='cpu',model_dtype=str(self.dtype),test_only_synthetic_cache=True)

class Store:
 def __init__(self,root):self.root=Path(root);self.done={};self.new=[]
 def append(self,sid,**kw):
  if sid in self.done:raise AssertionError('duplicate timing')
  self.done[sid]=kw;self.new.append((sid,kw))
 def finish(self,expected):
  assert set(self.done)==set(expected)
  return dict(complete=True,measured=len(self.done))

class CachePathTests(unittest.TestCase):
 def execute(self,dtype,block,lengths,resume=False):
  draws=[dict(sid='toy'+str(n),text=str(n),input_ids=list(range(1,n+1))) for n in lengths];job=dict(id='test',dtype=dtype,block=block,repetitions=2,warmups=1,length_group=[32,768],input_schedule='toy');args=SimpleNamespace(threads=1)
  with tempfile.TemporaryDirectory() as tmp:
   store=Store(tmp)
   if resume:store.done['kvcloak_unfused|protect|0']=dict(previous=True)
   with patch.object(module,'Consumer',ToyConsumer),patch.object(module,'inputs_for',return_value=(draws,draws[:1])):result=module.latency(job,args,store)
   self.assertEqual(result['measured'],18)
   for sid,r in store.new:
    self.assertTrue(math.isfinite(r['seconds']) and r['seconds']>=0)
    if r['operation']!='prefill':
     self.assertAlmostEqual(r['seconds_per_1000_tokens'],r['seconds']*1000/r['tokens']);self.assertEqual(r['cache_bytes'],2*1*2*8*(4 if dtype=='float32' else 2)*r['tokens'])
   self.assertEqual(len(store.new),17 if resume else 18)
 def test_short_partial_blocks(self):
  for block in [16,32,64]:
   with self.subTest(block=block):self.execute('float32',block,[32,63])
 def test_long_bfloat16(self):self.execute('bfloat16',16,[513,767])
 def test_resume(self):self.execute('float32',32,[65,129],True)

class AdditionalContracts(unittest.TestCase):
 def test_byte_recovery_at_public_and_partial_boundaries(self):
  from corrected.freshness import KeyService
  from runner.full_model import same_cache
  for dtype in ['float32','bfloat16']:
   for n in [32,33,63,513]:
    job=dict(dtype=dtype,block=64);c=ToyConsumer(job,None);ids=list(range(1,n+1))
    with tempfile.TemporaryDirectory() as tmp:
     prepare,_=module.fixture(c,job,ids,KeyService(Path(tmp)/'freshness.sqlite'),str(n)+dtype)
     recovered=prepare('veilkv_prefix','recover')()
     self.assertTrue(same_cache(recovered,c.forward(ids).past_key_values))
 def test_canonical_prefix_survives_length_dependent_rounding(self):
  from corrected.freshness import KeyService
  from runner.full_model import same_cache
  class LengthSensitive(ToyConsumer):
   def forward(self,ids,keep=None,cache=None):
    result=super().forward(ids,keep,cache)
    if cache is None:
     delta=len(ids)*.00001
     return SimpleNamespace(past_key_values=tuple((a+delta,b+delta) for a,b in result.past_key_values))
    return result
  job=dict(dtype='float32',block=16);c=LengthSensitive(job,None)
  with tempfile.TemporaryDirectory() as tmp:
   service=KeyService(Path(tmp)/'freshness.sqlite')
   for n in [33,63]:
    prepare,_=module.fixture(c,job,list(range(1,n+1)),service,'rounding-'+str(n))
    shared=prepare('veilkv_prefix','recover')();plain=prepare('aead_no_sharing','recover')()
    self.assertTrue(same_cache(shared,plain))
   self.assertEqual(len(c.prefixes),1)
 def test_configure_preserves_venv_entry_and_old_settings(self):
  import run,sys
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'old';new=root/'new';(old/'runtime').mkdir(parents=True);(new/'runtime').mkdir(parents=True);(old/'.venv/bin').mkdir(parents=True);(old/'cache').mkdir();(old/'.venv/bin/python').symlink_to(sys.executable)
   prior=dict(environment_python={'native':'.venv/bin/python'},cache_dir='cache');(old/'runtime/settings.json').write_text(json.dumps(prior));(new/'runtime/settings.json').write_text('{}');args=SimpleNamespace(old_native=str(old),python=None,cache_dir=None,gpu=0,threads=16)
   with patch.object(run,'ROOT',new):run.configure(args)
   conf=json.loads((new/'runtime/CONFIGURED.json').read_text());self.assertEqual(conf['native_python'],str(old/'.venv/bin/python'));self.assertEqual(json.loads((old/'runtime/settings.json').read_text()),prior)

if __name__=='__main__':unittest.main()
