"""Independently replay saved PermLLM observations and public signatures."""
import argparse,base64,gzip,importlib.util,json,sys,zlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def rows(p):
 with gzip.open(p,'rt') as f:
  yield from map(json.loads,f)
def exact(x):return x.contiguous().view(torch.uint8).numpy().tobytes()
def verify(folder,upstream,inputs):
 spec=importlib.util.spec_from_file_location('measure',ROOT/'run_check.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);perm,Comm,Node,Protocol,lookup=m.setup(upstream,inputs);summary=read(folder/'summary.json');fixtures={x['fid']:x for x in map(json.loads,(ROOT/'data/fixtures.jsonl').open())};seen=set();operator=[];trials=0
 for r in rows(folder/'operator.jsonl.gz'):
  key=(r['fid'],r['dtype']);assert key not in seen;seen.add(key);f=fixtures[r['fid']];dtype=getattr(torch,r['dtype'].split('.')[-1]);x=torch.stack([lookup[t] for t in f['input_ids']]).to(dtype);raw={};observable={}
  for t in r['trials_detail']:
   # Reimplement the pinned operator's two levels of sampling independently.
   y,rowperm=m.direct_reference(x,t['seed']);z=y.sort(dim=1).values
   a=raw.setdefault(exact(y),len(raw));b=observable.setdefault(exact(z),len(observable));assert (a,b)==(t['raw_state'],t['observable_state']);assert rowperm.tolist()==t['row_permutation'];assert t['exact_direct_check'] and t['inverse_exact'];trials+=1
  assert len(raw)==r['raw_states'] and len(observable)==r['observable_states']
  first=r['trials_detail'][0];y,_=m.direct_reference(x,first['seed']);assert zlib.decompress(base64.b64decode(r['first_output']['raw']['zlib_base64']))==exact(y)
  assert zlib.decompress(base64.b64decode(r['first_output']['observable']['zlib_base64']))==exact(y.sort(dim=1).values)
  operator.append({k:v for k,v in r.items() if k not in ['first_output','trials_detail']})
 assert len(seen)==240 and operator==summary['operator_results'] and trials==summary['operator_observations']
 native=[];native_trials=stable=passed=0
 for r in rows(folder/'protocol.jsonl.gz'):
  f=fixtures[r['fid']];x=torch.stack([lookup[t] for t in f['input_ids']]);states={};raw={};stable_cell=pass_cell=0
  for t in r['trials_detail']:
   observed,recovered,history=m.one_protocol(x,t['seed'],t['mask_seed'],perm,Comm,Node,Protocol);tags,maximum,bound,_=m.classify(observed,x);ideal,row=m.direct_reference(x,t['seed']);ideal_tags,_,_,_=m.classify(ideal,x)
   good=(maximum<bound and tags==ideal_tags);error=float((recovered-torch.nn.functional.layer_norm(x,[x.shape[1]])).abs().max());functional=error<=.001
   assert maximum==t['max_sorted_row_error'] and bound==t['frozen_error_bound'] and good==t['row_signature_stable'];assert error==t['layernorm_max_absolute_error'] and functional==t['functional_pass'];assert list(tags)==t['observed_row_tags'];assert states.setdefault(tags,len(states))==t['observable_state'];assert raw.setdefault(exact(observed),len(raw))==t['raw_state']
   assert sum(map(len,history.values()))==t['communication_messages'];native_trials+=1;stable+=good;passed+=functional;stable_cell+=good;pass_cell+=functional
  assert stable_cell==r['signature_stable'] and pass_cell==r['functional_pass'] and len(states)==r['observable_states']
  native.append({k:v for k,v in r.items() if k not in ['first_observation','trials_detail']})
 assert len(native)==40 and native==summary['protocol_results'] and native_trials==1280
 repeated=[r for r in operator if r['condition']=='one_repeated_unit'];controls4=[r for r in operator if r['block']==4]
 report=dict(accounting_pass=True,all_registered_cells=280,operator_cells=240,operator_observations=trials,native_three_party_cells=40,native_three_party_observations=native_trials,native_signature_stable=stable,native_layernorm_functional=passed,independent_all_output_partitions_replayed=True,raw_repeated_states=sorted({r['raw_states'] for r in repeated}),public_sorted_repeated_states={str(b):sorted({r['observable_states'] for r in repeated if r['block']==b}) for b in [4,8,16,32]},b4_sorted_controls={kind:sorted({r['observable_states'] for r in controls4 if r['condition']==kind}) for kind in ['one_repeated_unit','two_repeated_units','all_distinct_units']},max_native_row_error=max(r['max_row_error'] for r in native),max_native_layernorm_error=max(r['max_layernorm_error'] for r in native),new_model_forward_calls=0,scope='Native2D permutation + full generic nonlinear protocol; public signature orbit, not raw-ciphertext orbit or full ChatGLM inference')
 (folder/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);print(json.dumps(verify(a.results,a.upstream,a.inputs),indent=2))
