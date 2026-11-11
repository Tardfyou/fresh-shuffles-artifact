"""Reconstruct observed output partitions from full saved permutation transcripts."""
import argparse,collections,gzip,json,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def check(folder):
 proto=read(ROOT/'protocol/PROTOCOL.json');fixtures={r['fid']:r for r in map(json.loads,(ROOT/'data/fixtures.jsonl').open())};summ=read(folder/'summary.json');embedding=read_gz(folder/'public_embedding_rows.json.gz');lookup={int(t):torch.tensor(v,dtype=torch.float32) for t,v in embedding['rows'].items()};seen=set();result=[];trials=0
 expected={(fid,dtype,profile) for fid in fixtures for dtype in ['torch.float32','torch.bfloat16'] for profile in ['S','D','SD']}
 paths=[folder/'checks.jsonl.gz'] if (folder/'checks.jsonl.gz').exists() else sorted(folder.glob('checks-*.jsonl.gz'))
 assert paths
 def saved_lines():
  for path in paths:
   with gzip.open(path,'rt') as stream:
    yield from stream
 for line in saved_lines():
  row=json.loads(line);key=(row['fid'],row['dtype'],row['profile']);assert key not in seen;seen.add(key);f=fixtures[row['fid']];dtype=getattr(torch,row['dtype'].split('.')[-1]);x=torch.stack([lookup[t] for t in f['input_ids']]).to(dtype);n=f['block'];profile=row['profile'];states={};sperms=set()
  for i,r in enumerate(row['trials_detail']):
   assert i==r['trial'] and r['seed']==20261111+n*10000+i and r['exact_forward_check'];draws=r['permutations']
   if profile=='S':
    assert len(draws)==1 and sorted(draws[0])==list(range(n));output=tuple(f['input_ids'][j] for j in draws[0]);sperms.add(tuple(draws[0]))
   else:
    assert len(draws)==(1 if profile=='D' else 2) and sorted(draws[0])==list(range(x.shape[1]));y=x[:,draws[0]]
    if profile=='SD':assert sorted(draws[1])==list(range(n));y=y[draws[1]]
    output=y.contiguous().view(torch.uint8).numpy().tobytes()
   sid=states.setdefault(output,len(states));assert sid==r['state'];trials+=1
  assert len(states)==row['observed_states']
  if profile=='S':assert len(sperms)==row['distinct_row_permutations'] and len(states)<=f['predicted_sequence_orbit']
  stripped={k:v for k,v in row.items() if k!='trials_detail'};result.append(stripped)
 assert seen==expected and len(seen)==720
 assert result==summ['results'] and trials==summ['trials']
 repeated=[r for r in result if r['condition']=='one_repeated_unit' and r['profile']=='S'];assert len(repeated)==80
 controls4=[r for r in result if r['block']==4 and r['profile']=='S'];assert all(r['distinct_row_permutations']==math.factorial(4) for r in controls4)
 report=dict(accounting_pass=True,cells=len(seen),trials=trials,independent_output_partition_recount_pass=True,b4_all_24_permutations_observed=True,repeated_cells=len(repeated),repeated_observed_counts_by_block={str(b):sorted({r['observed_states'] for r in repeated if r['block']==b}) for b in proto['blocks']},b4_control_counts={c:sorted({r['observed_states'] for r in controls4 if r['condition']==c}) for c in proto['conditions']},fresh_column_counts={profile:sorted({r['observed_states'] for r in result if r['profile']==profile}) for profile in ['D','SD']},model_forward_calls=0,scope='Exact public-operator verification at layer0; not identification/recovery or a default GELO/STIP break')
 (folder/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');return report

def read_gz(p):
 with gzip.open(p,'rt') as f:return json.load(f)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);print(json.dumps(check(a.results),indent=2))
