"""Frozen public-operator orbit experiment; no attack or model inference."""
import argparse,base64,collections,datetime,gzip,importlib.util,json,math,platform,re,sys,time,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.open()]
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def prepare(manifest,subsets,cache):
 from transformers import AutoTokenizer
 proto=read(ROOT/'protocol/PROTOCOL.json');model=proto['model'];snapshot=cache/('models--'+model['repo'].replace('/','--'))/'snapshots'/model['revision'];tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
 by={r['sid']:r for r in lines(manifest)};ids=read(subsets)['lifecycle'];assert len(ids)==100
 pool=[];used=set();anchors=[]
 for category in ['code','creative','general','math','qa']:
  eligible=[]
  for sid in ids:
   r=by[sid]
   if r['category']!=category:continue
   for match in re.finditer(r'\b[A-Za-z]{2,12}\b',r['text']):
    word=match.group();encoded=tok(' '+word).input_ids
    if len(encoded)!=2 or encoded[0]!=tok.bos_token_id or encoded[1] in used:continue
    ok=all(tok((' '+word)*(b-1)).input_ids==[tok.bos_token_id]+[encoded[1]]*(b-1) for b in proto['blocks'])
    if not ok:continue
    unit=dict(word=word,token=encoded[1],source_sid=sid,category=category);pool.append(unit);eligible.append(unit);used.add(encoded[1])
  assert len(eligible)>=2,category
  anchors.extend(eligible[:2])
 assert len(pool)>=32 and len(anchors)==10
 fixtures=[]
 for a,anchor in enumerate(anchors):
  remaining=[u for u in pool if u['token']!=anchor['token']];other=remaining[a%len(remaining)]
  for b in proto['blocks']:
   for condition in proto['conditions']:
    if condition=='one_repeated_unit':units=[anchor]*(b-1)
    elif condition=='two_repeated_units':units=[anchor]*((b-1)//2)+[other]*((b-1)-(b-1)//2)
    else:units=[anchor]+remaining[:b-2]
    raw=''.join(' '+u['word'] for u in units);tokens=tok(raw).input_ids;assert tokens==[tok.bos_token_id]+[u['token'] for u in units]
    mult=list(collections.Counter(tokens).values());orbit=math.factorial(b)//math.prod(math.factorial(n) for n in mult)
    fixtures.append(dict(fid=f'anchor{a:02d}-b{b}-{condition}',anchor=anchor,block=b,condition=condition,raw_text=raw,input_ids=tokens,multiplicities=mult,predicted_sequence_orbit=orbit,nominal_sequence_group=math.factorial(b)))
 (ROOT/'data/fixtures.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fixtures))
 write(ROOT/'data/INPUTS.json',dict(fixtures=len(fixtures),anchors=anchors,lexical_pool_size=len(pool),source_cohort='Existing lifecycle100, five heuristic categories',manifest_rows=len(by),bos_token_id=tok.bos_token_id,preparation_only_tokenization=True,model_forward_calls=0))
 print('Frozen',len(fixtures),'fixtures from',len(anchors),'lexical anchors; model forwards0',flush=True)

class TraceTorch:
 def __init__(self,torch):self.module=torch;self.draws=[]
 def __getattr__(self,name):return getattr(self.module,name)
 def randperm(self,*a,**kw):
  value=self.module.randperm(*a,**kw);self.draws.append(value.detach().cpu().tolist());return value

def run(cache,device,out):
 import torch
 from safetensors import safe_open
 torch.set_num_threads(4);proto=read(ROOT/'protocol/PROTOCOL.json');fixtures=lines(ROOT/'data/fixtures.jsonl');assert len(fixtures)==120
 out.mkdir(parents=True,exist_ok=True);assert not (out/'checks.jsonl.gz').exists(),'Never overwrite completed data'
 spec=importlib.util.spec_from_file_location('upstream_permutation',ROOT/'vendor/permute_states.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 trace=TraceTorch(torch);module.torch=trace
 model=proto['model'];snapshot=cache/('models--'+model['repo'].replace('/','--'))/'snapshots'/model['revision'];idx=snapshot/'model.safetensors.index.json';key='model.embed_tokens.weight'
 file=snapshot/(read(idx)['weight_map'][key] if idx.exists() else 'model.safetensors')
 tids=sorted({t for f in fixtures for t in f['input_ids']});embedding={}
 with safe_open(file,framework='pt',device='cpu') as handle:
  view=handle.get_slice(key)
  for tid in tids:embedding[tid]=view[tid:tid+1].clone()
 # Retain actual public embedding rows once, sufficient to reconstruct every output.
 with gzip.open(out/'public_embedding_rows.json.gz','wt') as stream:
  json.dump(dict(model=model,key=key,dtype=str(embedding[tids[0]].dtype),rows={str(t):embedding[t].float().flatten().tolist() for t in tids}),stream)
 # A read-only RNG recorder must match the untouched public function exactly.
 vanilla={};exec((ROOT/'vendor/permute_states.py').read_text(),vanilla)
 sanity=0
 test=torch.arange(24,dtype=torch.float32,device=device).reshape(4,6)
 for profile in ['None','S','D','SD']:
  for seed in range(8):
   torch.manual_seed(seed);y=vanilla['permute_states'](test,profile)
   torch.manual_seed(seed);trace.draws=[];z=module.permute_states(test,profile);assert torch.equal(y,z);sanity+=1
 write(out/'environment.json',dict(torch=torch.__version__,python=platform.python_version(),device=device,hardware=torch.cuda.get_device_name() if device.startswith('cuda') else platform.processor(),stage='Layer0 public embedding lookup, before contextual attention',model_forward_calls=0,source_revision=read(ROOT/'protocol/UPSTREAM.json')['revision'],recorder_equivalence_checks=sanity,protocol_created=proto['created_before_measurements'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 total=0;start=time.time();summaries=[]
 with gzip.open(out/'checks.jsonl.gz','wt') as stream:
  for fixture in fixtures:
   for dtype in [torch.float32,torch.bfloat16]:
    x=torch.cat([embedding[t] for t in fixture['input_ids']],0).to(device=device,dtype=dtype);actual_mult=collections.Counter(v.contiguous().view(torch.uint8).cpu().numpy().tobytes() for v in x);expected=math.factorial(fixture['block'])//math.prod(math.factorial(n) for n in actual_mult.values());assert expected==fixture['predicted_sequence_orbit'],'Embedding collisions change token multiplicity'
    assert torch.equal(module.permute_states(x,'None'),x)
    for profile in ['S','D','SD']:
     count=proto['primary_observations_per_cell'] if profile=='S' else proto['negative_observations_per_cell'];states={};row_perms=set();trial_records=[]
     for trial in range(count):
      seed=20261111+fixture['block']*10000+trial;torch.manual_seed(seed);trace.draws=[];y=module.permute_states(x,profile)
      draws=trace.draws
      if profile=='S':direct=x[torch.tensor(draws[0],device=device)];row_perms.add(tuple(draws[0]))
      elif profile=='D':direct=x[:,torch.tensor(draws[0],device=device)]
      else:direct=x[:,torch.tensor(draws[0],device=device)][torch.tensor(draws[1],device=device)]
      assert torch.equal(y,direct),'Recorder/shape/source mismatch'
      payload=y.contiguous().view(torch.uint8).cpu().numpy().tobytes();state=states.setdefault(payload,len(states));trial_records.append(dict(trial=trial,seed=seed,state=state,permutations=draws,exact_forward_check=True));total+=1
     summary=dict(fid=fixture['fid'],condition=fixture['condition'],block=fixture['block'],dtype=str(dtype),profile=profile,trials=count,observed_states=len(states),nominal_sequence_group=fixture['nominal_sequence_group'],predicted_sequence_orbit=expected,distinct_row_permutations=len(row_perms) if profile=='S' else None,exact_indexing_checks=count,no_failure_observations_excluded=True)
     if profile=='S':assert len(states)<=expected
     stream.write(json.dumps(dict(**summary,trials_detail=trial_records))+'\n');stream.flush();summaries.append(summary)
   if len(summaries)%60==0:print('progress',len(summaries),'/720cells','seconds',round(time.time()-start,1),flush=True)
 write(out/'summary.json',dict(complete=True,cells=len(summaries),trials=total,seconds=time.time()-start,results=summaries,model_forward_calls=0))
 print('complete',len(summaries),'cells',total,'observations',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--cache',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--subsets',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,default=ROOT/'results/cpu_operator');a=p.parse_args()
 if a.prepare:prepare(a.manifest,a.subsets,a.cache)
 else:run(a.cache,a.device,a.out)
