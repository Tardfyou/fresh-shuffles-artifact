"""Offline validation of saved result records; no model or manuscript calls."""
import collections
import fcntl
import json
import math
from pathlib import Path
from battery_evidence import summarize as summarize_battery

PROTOCOL='corrected-20261029-v1'

KEYS=[42,1234,5678,9012,34567]

MODELS=[('Qwen2.5-0.5B',2),('Llama-3.2-1B',8),('Phi-3-mini-4k-instruct',32)]

CONDITIONS=['end2end','oracle_ms','no_cross_block','random_order']

class Pending(Exception):pass

def require(test,message):
 if not test:raise ValueError(message)

def load(paths):
 rows=[]
 for path in paths:
  if not path.exists():continue
  with path.open() as f:
   fcntl.flock(f,fcntl.LOCK_SH);raw=f.read();fcntl.flock(f,fcntl.LOCK_UN)
  if raw and not raw.endswith('\n'):raise Pending(f'{path.name}: final line still being written')
  for line in raw.splitlines():
   r=json.loads(line);require(r.get('protocol')==PROTOCOL and r.get('status')=='ok',f'{path.name}: incompatible record');rows.append(r)
 return rows

def exact(rows,expected):
 ids=[r['sid'] for r in rows];require(len(ids)==len(set(ids)),'duplicate sample IDs')
 extra=set(ids)-set(expected);require(not extra,f'{len(extra)} unexpected IDs')
 missing=set(expected)-set(ids)
 if missing:raise Pending(f'{len(ids)}/{len(expected)} records; {len(missing)} missing')
 return {r['sid']:r for r in rows}

def checked_score(score,truth,heads):
 require(len(truth)>0 and len(truth)%16==0,'invalid complete-block truth length')
 require(score['head_coverage']==score['expected_heads']==heads,'incomplete head coverage')
 rr=score['heads'];require({h['head'] for h in rr}==set(range(heads)) and len(rr)==heads,'head IDs differ')
 for h in rr:
  require(h['total']==len(truth) and 0<=h['correct']<=h['total'],'invalid per-head denominator')
  require(len(h['block_correct'])==len(truth)//16 and sum(h['block_correct'])==h['correct'] and all(0<=v<=16 for v in h['block_correct']),'block counts differ')
 require(score['total']==heads*len(truth) and score['correct']==sum(h['correct'] for h in rr),'pooled counts differ')
 return score['correct'],score['total']

def pooled(scores):
 c=sum(v[0] for v in scores);n=sum(v[1] for v in scores);return dict(correct=c,total=n,percent=100*c/n)

def suite(a,subsets,name):
 rows=load([a.results/f'{name}.jsonl']);by=exact(rows,subsets[name]);result={}
 for key in ['same_epoch','stale_after_rotation']+(['recalibrated'] if name=='lifecycle' else []):
  result[key]=pooled([checked_score(r['scores'][key],r['truth'],2) for r in by.values()])
 errors=[r['scores']['functional']['attention_max_error'] for r in by.values()]
 require(all(r['scores']['functional']['finite'] for r in by.values()) and all(math.isfinite(v) for v in errors),'nonfinite functional output')
 result.update(n=len(by),attention_max_error=max(errors),attention_median_error=sorted(errors)[len(errors)//2]);return result

def cross_key(a,subsets):
 result={};source=set(subsets['cross_key']);require(len(source)==100,'cross-key cohort not 100')
 for short,heads in MODELS:
  expected={f'{sid}-{key}' for sid in source for key in KEYS};rows=load([a.results/f'cross_key_{short}.jsonl']);by=exact(rows,expected)
  for r in rows:require(r['model'].split('/')[-1]==short and r['source_sid'] in source and r['sid']==f"{r['source_sid']}-{r['key_seed']}" and r['key_seed'] in KEYS,'cross-key source mapping differs')
  cells={str(key):dict(**pooled([checked_score(r['scores'],r['truth'],heads) for r in rows if r['key_seed']==key]),minimum_prompt_percent=100*min(r['scores']['correct']/r['scores']['total'] for r in rows if r['key_seed']==key)) for key in KEYS}
  rates=[r['scores']['correct']/r['scores']['total'] for r in rows]
  result[short]=dict(n=500,heads=heads,by_key=cells,minimum_prompt_percent=100*min(rates),macro_prompt_percent=100*sum(rates)/500,pooled=pooled([checked_score(r['scores'],r['truth'],heads) for r in rows]))
 return result

def ordering(a,samples):
 ids={sid for sid,s in samples.items() if s['or_subset']};require(len(ids)==602,'ordering cohort not 602')
 bags=exact(load(sorted((a.results/'bags').glob('part_*.jsonl'))),ids);rows=load(sorted((a.results/'order').glob('part_*.jsonl')));by=exact(rows,ids)
 for sid,r in by.items():
  require(set(r['conditions'])==set(CONDITIONS),'four matched conditions missing')
  require(r['inference_dtype']=='bfloat16' and r['beam_width']==16 and r['prior_context_limit']==80 and r['context_limit']==96,'ordering protocol differs')
  bag=bags[sid];truth=bag['truth'];require(len(truth)>0 and len(truth)%16==0,'invalid target length')
  require(r['len_bucket']==bag['len_bucket']==samples[sid]['len_bucket'],'length stratum differs')
  require(r['bag_exact']==[collections.Counter(b)==collections.Counter(truth[i*16:(i+1)*16]) for i,b in enumerate(bag['bags'])],'saved bag exactness differs')
  require({int(h) for h in bag['head_bags']}=={0,1},'bag preparation lacks both heads')
  for name,c in r['conditions'].items():
   pred=c['prediction'];require(len(pred)==len(truth)==c['tokens'],'prediction length differs')
   require(c['position_correct']==sum(x==y for x,y in zip(truth,pred)),'position count differs')
   require(abs(c['position_accuracy']-c['position_correct']/len(truth))<1e-12 and c['exact_match']==(pred==truth),'position metrics differ')
   expected_bags=[truth[i:i+16] for i in range(0,len(truth),16)] if name=='oracle_ms' else bag['bags']
   conservation=[collections.Counter(pred[i:i+16])==collections.Counter(expected_bags[i//16]) for i in range(0,len(pred),16)]
   require(all(conservation) and conservation==c['block_conservation'],'within-block conservation failed')
   require(c['block_exact']==sum(pred[i:i+16]==truth[i:i+16] for i in range(0,len(truth),16)),'exact-block counts differ')
 def aggregate(rr):
  return {name:dict(macro_percent=100*sum(r['conditions'][name]['position_accuracy'] for r in rr)/len(rr),correct=sum(r['conditions'][name]['position_correct'] for r in rr),total=sum(r['conditions'][name]['tokens'] for r in rr),exact_spans=sum(r['conditions'][name]['exact_match'] for r in rr),exact_blocks=sum(r['conditions'][name]['block_exact'] for r in rr),seconds=sum(r['conditions'][name]['seconds'] for r in rr)) for name in CONDITIONS}
 cells={bucket:dict(n=len(rr),conditions=aggregate(rr)) for bucket in sorted({s['len_bucket'] for s in samples.values()},key=lambda value:int(value.split(',')[0][1:])) if (rr:=[r for r in rows if r['len_bucket']==bucket])}
 groups=collections.defaultdict(list)
 for r in rows:groups[tuple(bags[r['sid']]['truth'])].append(r)
 span_balanced={name:100*sum(sum(r['conditions'][name]['position_accuracy'] for r in group)/len(group) for group in groups.values())/len(groups) for name in CONDITIONS}
 return dict(n=602,distinct_evaluated_spans=len(groups),span_balanced_percent=span_balanced,conditions=aggregate(rows),length_bins=cells,exact_oracle_output_matches=sum(r['conditions']['end2end']['prediction']==r['conditions']['oracle_ms']['prediction'] for r in rows),exact_recovered_bag_spans=sum(all(r['bag_exact']) for r in rows),blocks=sum(len(b['bags']) for b in bags.values()))

def direction(a,subsets):
 expected={r['sid'] for r in subsets['direction']};require(len(expected)==553,'swap cohort not 553')
 rows=load(sorted((a.results/'direction').glob('part_*.jsonl')));exact(rows,expected)
 block=glob=0.;equal_global=equal_block=same_tokens=different_length=0
 for r in rows:
  require(r['semantic_direction_validated'] is False and r['head_rule']=='fixed head 0','direction task differs')
  candidates=r['candidate_tokens'];obs=r['observations'];order=r['candidate_order'];require(sorted(order)==[0,1] and {v['target'] for v in obs}=={0,1} and len(obs)==2,'both balanced targets required')
  eg=collections.Counter(candidates[0])==collections.Counter(candidates[1]);eb=len(candidates[0])==len(candidates[1]) and all(collections.Counter(candidates[0][i:i+16])==collections.Counter(candidates[1][i:i+16]) for i in range(0,len(candidates[0]),16))
  require(eg==r['equal_global_bags'] and eb==r['equal_block_bags'],'candidate-control flags differ')
  equal_global+=eg;equal_block+=eb;same_tokens+=candidates[0]==candidates[1];different_length+=len(candidates[0])!=len(candidates[1])
  for v in obs:
   require(v['truth']==candidates[v['target']] and v['tokens']==len(v['truth']) and sum(map(len,v['bags']))==v['tokens'],'direction target mapping differs')
   for key,global_only in [('candidate',False),('global_bag',True)]:
    observed=[sum(v['bags'],[])] if global_only else v['bags'];offered=[]
    for ci in order:
     ts=candidates[ci];offered.append([ts] if global_only else [ts[i:i+16] for i in range(0,len(ts),16)])
    scores=[2*sum(sum((collections.Counter(x)&collections.Counter(y)).values()) for x,y in zip(observed,c))-sum(map(len,observed))-sum(map(len,c)) for c in offered]
    require(scores==v[key]['scores'],'candidate distance scores differ')
    winners=[i for i,x in enumerate(scores) if x==max(scores)];credit=1/len(winners) if order.index(v['target']) in winners else 0
    require(abs(credit-v[key]['accuracy_credit'])<1e-12,'candidate credit differs')
    if global_only:glob+=credit
    else:block+=credit
 return dict(n=553,targets=1106,block_accuracy_percent=100*block/1106,global_accuracy_percent=100*glob/1106,equal_global_pairs=equal_global,equal_block_pairs=equal_block,identical_token_pairs=same_tokens,different_length_pairs=different_length,length_only_accuracy_percent=100*(different_length+.5*(553-different_length))/553,truth_global_bag_oracle_percent=100*(553-.5*equal_global)/553,truth_block_bag_oracle_percent=100*(553-.5*equal_block)/553)
