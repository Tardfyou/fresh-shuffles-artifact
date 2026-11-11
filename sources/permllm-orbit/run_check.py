"""Native PermLLM random-permutation operator/protocol checks; no model inference."""
import argparse,base64,gzip,importlib,json,math,platform,subprocess,sys,time,zlib
from collections import Counter
from functools import partial
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def payload(x):return x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
def snapshot(x):return dict(shape=list(x.shape),dtype=str(x.dtype),zlib_base64=base64.b64encode(zlib.compress(payload(x),6)).decode())

def setup(upstream,inputs):
 revision=read(ROOT/'protocol/PROTOCOL.json')['upstream']['revision'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()==revision
 sys.path.insert(0,str(upstream));from perm_llm.common.torch_utils import permute_2d_with_seed
 from perm_llm.common.communication import SimulatedCommunication,Node
 from perm_llm.protocols.element_wise import SS_ElementWise__RandPerm
 with gzip.open(inputs,'rt') as f:data=json.load(f)
 lookup={int(k):torch.tensor(v,dtype=torch.float32) for k,v in data['rows'].items()}
 return permute_2d_with_seed,SimulatedCommunication,Node,SS_ElementWise__RandPerm,lookup

def direct_reference(x,seed):
 g=torch.Generator(device=x.device).manual_seed(seed);r=torch.randperm(x.shape[0],generator=g,device=x.device);cols=torch.stack([torch.randperm(x.shape[1],generator=g,device=x.device) for _ in range(x.shape[0])]);return x.gather(1,cols)[r],r

def one_protocol(x,key,maskseed,permute,Comm,Node,Protocol):
 torch.manual_seed(maskseed);comm=Comm(['n0','n1','n2']);n0,n1,n2=[Node(comm,name) for name in ['n0','n1','n2']];observed=[]
 def operation(y):observed.append(y.detach().clone());return F.layer_norm(y,[x.shape[1]])
 protocol=Protocol(permute,partial(permute,reverse=True),operation,'orbit',n0,n1,n2,{'x':10.,'z':10.},'cpu');protocol.prepare();n0.storage['orbit:new_perm']=key;n0.storage['orbit:new_invperm']=key;comm.new_stage('offline');protocol.offline_execute(list(x.shape));share=torch.rand_like(x)*10-5;n0.storage['orbit:x0']=share;n1.storage['orbit:x1']=x-share;comm.new_stage('online');protocol.online_execute();assert len(observed)==1
 recovered=n0.storage['orbit:z0']+n1.storage['orbit:z1'];assert not comm.communication_buffer
 return observed[0],recovered,comm.comm_history

def classify(observed,x):
 # Classification prototypes use only the attacker's known, registered own probe.
 prototypes=[]
 for v in x.sort(dim=1).values:
  if not any(torch.equal(v,w) for w in prototypes):prototypes.append(v)
 p=torch.stack(prototypes);separation=torch.pdist(p.double()).min().item();bound=separation/4
 d=torch.cdist(observed.sort(dim=1).values.double(),p.double());values,tags=d.min(dim=1)
 return tuple(tags.tolist()),float(values.max()),bound,separation

def run(upstream,inputs,out):
 torch.set_num_threads(4);proto=read(ROOT/'protocol/PROTOCOL.json');fixtures=[json.loads(s) for s in (ROOT/'data/fixtures.jsonl').open()];assert len(fixtures)==120
 assert not (out/'operator.jsonl.gz').exists(),'Do not overwrite results';out.mkdir(parents=True,exist_ok=True)
 permute,Comm,Node,Protocol,lookup=setup(upstream,inputs)
 write(out/'environment.json',dict(device='cpu',torch=torch.__version__,python=platform.python_version(),source_revision=proto['upstream']['revision'],model_forward_calls=0,width=next(iter(lookup.values())).numel(),recorded_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),protocol_registered=proto['registered_utc'],native_three_party_protocol=True))
 start=time.time();summaries=[]
 with gzip.open(out/'operator.jsonl.gz','wt') as stream:
  for f in fixtures:
   for dtype in [torch.float32,torch.bfloat16]:
    x=torch.stack([lookup[t] for t in f['input_ids']]).to(dtype);raw={};observable={};trials=[];first=None
    for trial in range(proto['operator_trials_per_cell']):
     seed=2026111100+f['block']*10000+trial;y=permute(x,seed);expected,r=direct_reference(x,seed);assert torch.equal(y,expected);assert torch.equal(permute(y,seed,reverse=True),x)
     canon=y.sort(dim=1).values;direct_canon=x.sort(dim=1).values[r];assert torch.equal(canon,direct_canon)
     a=raw.setdefault(payload(y),len(raw));b=observable.setdefault(payload(canon),len(observable));trials.append(dict(trial=trial,seed=seed,raw_state=a,observable_state=b,row_permutation=r.tolist(),exact_direct_check=True,inverse_exact=True))
     if first is None:first=dict(raw=snapshot(y),observable=snapshot(canon))
    result=dict(fid=f['fid'],condition=f['condition'],block=f['block'],dtype=str(dtype),trials=len(trials),raw_states=len(raw),observable_states=len(observable),predicted_row_orbit=f['predicted_sequence_orbit'],nominal_row_permutations=f['nominal_sequence_group'],nominal_full_log2=math.lgamma(x.shape[0]+1)/math.log(2)+x.shape[0]*math.lgamma(x.shape[1]+1)/math.log(2))
    stream.write(json.dumps(dict(**result,first_output=first,trials_detail=trials))+'\n');stream.flush();summaries.append(result)
   if len(summaries)%40==0:print('operator cells',len(summaries),'/240','seconds',round(time.time()-start,1),flush=True)
 protocol_results=[]
 with gzip.open(out/'protocol.jsonl.gz','wt') as stream:
  for f in [x for x in fixtures if x['condition']=='one_repeated_unit']:
   x=torch.stack([lookup[t] for t in f['input_ids']]);trial_rows=[];states={};raw={};witness=None
   # Freeze the noise classifier bound from input vectors before any native call.
   _,_,bound,separation=classify(x,x)
   for trial in range(proto['protocol_trials_per_cell']):
    key=2026111100+f['block']*10000+trial;maskseed=991000+f['block']*1000+trial;observed,recovered,history=one_protocol(x,key,maskseed,permute,Comm,Node,Protocol);tags,maximum,used_bound,sep=classify(observed,x);assert used_bound==bound and sep==separation
    exact_perm,r=direct_reference(x,key);direct_tags,_,_,_=classify(exact_perm,x);stable=(maximum<bound and tags==direct_tags);err=float((recovered-F.layer_norm(x,[x.shape[1]])).abs().max());sid=states.setdefault(tags,len(states));rawid=raw.setdefault(payload(observed),len(raw))
    trial_rows.append(dict(trial=trial,seed=key,mask_seed=maskseed,observable_state=sid,raw_state=rawid,observed_row_tags=tags,expected_row_permutation=r.tolist(),max_sorted_row_error=maximum,frozen_error_bound=bound,row_signature_stable=stable,layernorm_max_absolute_error=err,functional_pass=err<=.001,communication_messages=sum(len(v) for v in history.values())))
    if witness is None:witness=dict(plaintext_observer=snapshot(observed),recovered=snapshot(recovered))
   result=dict(fid=f['fid'],block=f['block'],trials=len(trial_rows),signature_stable=sum(t['row_signature_stable'] for t in trial_rows),functional_pass=sum(t['functional_pass'] for t in trial_rows),raw_states=len(raw),observable_states=len(states),max_layernorm_error=max(t['layernorm_max_absolute_error'] for t in trial_rows),max_row_error=max(t['max_sorted_row_error'] for t in trial_rows),frozen_error_bound=bound)
   stream.write(json.dumps(dict(**result,first_observation=witness,trials_detail=trial_rows))+'\n');stream.flush();protocol_results.append(result)
 write(out/'summary.json',dict(complete=True,operator_cells=len(summaries),operator_observations=sum(r['trials'] for r in summaries),protocol_cells=len(protocol_results),protocol_observations=sum(r['trials'] for r in protocol_results),operator_results=summaries,protocol_results=protocol_results,seconds=time.time()-start,all_normal_outcomes_retained=True))
 print('complete',len(summaries),'operator cells,',len(protocol_results),'native protocol cells',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.upstream,a.inputs,a.output)
