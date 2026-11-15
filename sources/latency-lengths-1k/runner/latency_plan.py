"""Frozen input binding and dynamic denominators for latency observations."""
from functools import lru_cache
import json,math
from . import ROOT

METHODS=['kvcloak_unfused','kvcloak_fused','veilkv_prefix','aead_no_sharing']
OPERATIONS=['protect','recover']

def read(path):return json.loads(path.read_text())

@lru_cache(maxsize=64)
def frozen(schedule_path,input_file,lo,hi):
    with (ROOT/'data/samples_main_public.jsonl').open() as stream:
        source={r['sid']:r for r in map(json.loads,stream)}
    with (ROOT/input_file).open() as stream:
        tokens={r['sid']:r for r in map(json.loads,stream)}
    plan=read(ROOT/schedule_path)
    assert plan['length_group']==[lo,hi] and len(plan['draws'])==1000 and len(plan['warmups'])==20
    candidates=set(plan['eligible_ids'])
    assert candidates=={sid for sid,r in tokens.items() if lo<=len(r['input_ids'])<hi}
    assert candidates and len(source)==3000
    def item(sid):
        assert sid in candidates
        return dict(sid=sid,text=source[sid]['text'],input_ids=tokens[sid]['input_ids'])
    return [item(sid) for sid in plan['draws']],[item(sid) for sid in plan['warmups']]

def inputs_for(job):
    draws,warm=frozen(job['input_schedule'],job['input_file'],*job['length_group'])
    return draws[:job['repetitions']],warm[:job['warmups']]

def expected_keys(job):
    n=job['repetitions']
    return {f'{m}|{op}|{i}' for m in METHODS for op in OPERATIONS for i in range(n)}|{'prefill|'+str(i) for i in range(n)}

def check_record(job,row):
    draws,_=inputs_for(job);i=row['repetition'];assert 0<=i<job['repetitions'];item=draws[i]
    assert row['source_sid']==item['sid'] and row['tokens']==len(item['input_ids']) and row['length_group']==job['length_group']
    assert math.isfinite(row['seconds']) and row['seconds']>=0
    if row['operation']=='prefill':assert row['sid']=='prefill|'+str(i) and row['method']=='plain_model';return
    assert row['method'] in METHODS and row['operation'] in OPERATIONS
    assert row['sid']==f'{row["method"]}|{row["operation"]}|{i}'
    shape=job['cache_shape'];expected=2*shape['layers']*shape['heads']*shape['head_dimension']*shape['element_bytes']*row['tokens']
    assert row['cache_bytes']==expected
    assert math.isclose(row['seconds_per_1000_tokens'],row['seconds']*1000/row['tokens'],rel_tol=1e-12,abs_tol=1e-10)
    assert math.isclose(row['milliseconds_per_GiB'],row['seconds']*1000/(expected/(1<<30)),rel_tol=1e-12,abs_tol=1e-8)
    assert row['public_prefix_tokens']==(16*min(2,row['tokens']//16) if row['method']=='veilkv_prefix' else 0)
