"""Check complete602-prompt saved cohorts without model, encoder or API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
from summarize_gpu_results import records, summarize

RUN='gpu602-native-nonlive-20261114'

def read(path):
    return json.loads(path.read_text())

def check_score(score, truth):
    if 'heads' not in score:
        assert score['correct']==0 and score['calibration_failed_heads']
        return
    total=correct=0
    heads=set()
    for head in score['heads']:
        assert head['head'] not in heads
        heads.add(head['head'])
        bags=head['bags']
        assert len(bags)==len(truth)//16
        counts=[sum((Counter(bag)&Counter(truth[i*16:(i+1)*16])).values()) for i,bag in enumerate(bags)]
        assert counts==head['block_correct']
        assert sum(counts)==head['correct'] and head['total']==len(truth)
        total+=head['total'];correct+=head['correct']
    assert score['correct']==correct and score['total']==total

def check(root):
    src=root/'sources/gpu602-native';raw=root/'results/gpu/accepted'/RUN
    summaries=root/'results/gpu/summaries'/RUN
    targets=set(read(root/'metadata/gpu602/target602_ids.json'))
    assert len(targets)==602
    assert targets=={x['sid'] for x in map(json.loads,(src/'data/samples_main_public.jsonl').open()) if x['or_subset']}
    jobs=read(src/'protocol/jobs.json')
    assert len(jobs)==50
    defense=[x for x in map(json.loads,(src/'data/defense_inputs.jsonl').open()) if x['sid'] in targets]
    reports=[];total=0;score_cells=0;functional_failures=0;envs={}
    for job in jobs:
        folder=raw/job['id'];assert read(folder/'job.json')==job
        kind=job['kind'];actual=set()
        if kind=='sa_functional':
            rows=[x for i,x in enumerate(defense) if i%job['shards']==job['shard']]
            expected={x['sid']+'|'+shape+'|'+condition for x in rows for shape in ['no_pad','left8','right8','left16','right16'] for condition in ['original','refresh']}
        elif kind=='cross_key':expected={sid+'-'+str(seed) for sid in targets for seed in job.get('key_seeds',[42,1234,5678,9012,34567])}
        else:expected=targets
        for row in records(folder):
            if row['status']!='measured':continue
            assert row['sid'] not in actual;actual.add(row['sid'])
            assert row['job_id']==job['id'] and row['science_version']=='wildchat602-full-cohort-20261113-v1-native'
            key=(job['id'],row['invocation_id'])
            if key not in envs:
                env=read(folder/'invocations'/row['invocation_id']/'environment.json')
                assert env['device']=='cuda:0' and env['model_dtype']=='torch.'+job['dtype']
                assert env['model']==job['model']
                assert env['torch']=='2.6.0+cu124' and env['transformers']=='4.51.3'
                envs[key]=env
            cells=[]
            if kind in ['fused','cross_key']:cells=[row['scores']]
            elif kind=='ratio':cells=list(row['scores'].values())
            elif kind in ['lifecycle','sa_rotation']:cells=list(row['conditions'].values())
            for cell in cells:check_score(cell,row['truth']);score_cells+=1
            if kind=='sa_functional':
                assert row['M_fixed'] and row['duplicate_plain_sanity']
                for seg in row['segments']:
                    diff=seg['logits'];valid=diff['finite'] and diff['argmax_equal'] and diff['maximum_absolute_error'] <=(.001 if job['dtype']=='float32' else .6)
                    assert seg['functional_pass']==valid
                    assert diff['argmax_equal']==(diff['observed_argmax']==diff['plain_argmax'])
                    functional_failures+=not valid
                assert row['functional_pass']==all(s['functional_pass'] for s in row['segments'])
            elif kind in ['matched','spectrum']:
                for methods in row['seeds'].values():
                    for result in methods.values():
                        assert result['chosen']==max(range(100),key=lambda i:result['scores'][i])
                        assert result['hit']==(result['chosen']==row['true_idx'])
            elif kind=='repeat_reference':
                assert row['equal']==(row['a']==row['b'])
                if job['variant']=='C':assert row['reset_verified']
            elif kind=='paged_compat':
                assert all(s['argmax_equal']==(s['observed_argmax']==s['plain_argmax']) for s in row['segments'])
            elif kind=='large_byte':assert row['functional_pass']==all(row['checks'].values())
        assert actual==expected and len(actual)==job['expected_observations'],job['id']
        total+=len(actual);reports.append({'job':job['id'],'records':len(actual)})
    assert total==49364
    recount=summarize(raw)
    assert recount==read(summaries/'PAPER_VALUES.json')
    smoke=read(src/'protocol/smoke_jobs.json');smoke_rows=0
    for job in smoke:
        rows=list(records(raw/job['id']));measured=[r for r in rows if r['status']=='measured']
        assert len(measured)==len({r['sid'] for r in measured})==job['expected_observations']
        smoke_rows+=len(measured)
    assert len(smoke)==10 and smoke_rows==38
    return dict(pass_saved_records=True,formal_jobs=50,formal_records=total,startup_jobs=10,startup_records=smoke_rows,target_ids=602,score_cells_checked=score_cells,sa_failed_segments_retained=functional_failures,model_or_api_calls=0,jobs=reports)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--report',type=Path);args=p.parse_args();result=check(args.root.resolve())
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
