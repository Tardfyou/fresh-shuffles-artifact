"""Recompute saved predictions, denominators and complete measurement keys."""
import argparse
from collections import Counter
import json
from pathlib import Path
from . import ROOT
from .paths import result_root
from .records import load


def expected_keys(job):
    from .worker import samples_for
    kind=job['kind']
    if kind=='probe':
        budgets=[256,2048] if job.get('smoke') else [16,32,64,128,256,512,1024,2048]
        return {f'probe-keys-{b}-{t}' for b in budgets for t in range(2 if job.get('smoke') else 100)}
    if kind=='calibration_cost':return {'repetition-'+str(i) for i in range(5)}
    if kind=='latency':
        n=2 if job.get('smoke') else 100
        return {f'{m}|{op}|{i}' for m in ['kvcloak_unfused','kvcloak_fused','veilkv_prefix','aead_no_sharing'] for op in ['protect','recover'] for i in range(n)}|{'prefill|'+str(i) for i in range(n)}
    if kind in ['privacy','direction']:
        filename='privacy_inputs.jsonl' if kind=='privacy' else 'direction_inputs.jsonl'
        rows=[json.loads(l) for l in (ROOT/'data'/filename).open()]
        return {r['sid'] for r in (rows[:2] if job.get('smoke') else rows)}
    if kind=='official_utility':
        rows=[json.loads(line) for line in (ROOT/'data/utility'/(job['benchmark']+'.jsonl')).open()]
        return {r['sid'] for r in (rows[:2] if job.get('smoke') else rows)}
    if kind=='ordering':
        rows=[json.loads(l) for l in (ROOT/'data/samples_main_public.jsonl').open() if json.loads(l)['or_subset']]
        return {r['sid'] for r in (rows[:2] if job.get('smoke') else rows)}
    if kind=='marker_numerics':
        rows=[r for r in map(json.loads,(ROOT/'data/marker_numeric_fixtures.jsonl').open()) if r['tag']==job['model']['tag']]
        if job.get('smoke'):rows=rows[:2]
        return {'|'.join([r['sid'],str(r['seed']),str(r['h']),str(r['k'])]) for r in rows}
    if kind in ['byte_defense','sa_functional']:
        from .full_model import selected,SHAPES
        labels=['chained_reencode','identity_epoch','mid_session','shared_prefix'] if kind=='byte_defense' else ['original','refresh']
        return {r['sid']+'|'+shape+'|'+label for r in selected(job) for shape in SHAPES for label in labels}
    if kind in ['spectrum','matched']:
        from .extended import candidates
        return {r['sid'] for r in candidates(job)}
    rows=samples_for(job)
    if kind=='cross_key':return {f'{r["sid"]}-{key}' for r in rows for key in job.get('key_seeds',[42,1234,5678,9012,34567])}
    return {r['sid'] for r in rows}


def check_score(score,truth,block,heads=None):
    if 'heads' not in score:
        assert score.get('calibration_failed_heads') and score['correct']==0
        return
    assert score['total']==len(truth)*(heads if heads is not None else len(score['heads']))
    head_ids=[h['head'] for h in score['heads']]
    assert len(head_ids)==len(set(head_ids)), 'Duplicate scored heads'
    if heads is not None:assert set(head_ids)==set(range(heads)), 'Missing scored heads'
    total=correct=0
    for h in score['heads']:
        bags=h['bags'];assert len(bags)==len(truth)//block
        assert all(len(bag)==block or (not h.get('calibration_complete',True) and not bag) for bag in bags)
        counts=[sum((Counter(bag)&Counter(truth[k*block:(k+1)*block])).values()) for k,bag in enumerate(bags)]
        assert counts==h['block_correct'] and sum(counts)==h['correct'] and h['total']==len(truth)
        total+=h['total'];correct+=h['correct']
    assert total==score['total'] and correct==score['correct']


def verify(job,folder):
    observations=load(Path(folder)/'records.jsonl');rows=[r for r in observations if r['status']=='measured']
    ids=[r['sid'] for r in rows];expected=expected_keys(job)
    assert len(ids)==len(set(ids)), 'Duplicate measured keys'
    assert set(ids)==expected, f'Missing {len(expected-set(ids))}, extra {len(set(ids)-expected)}'
    environment=json.loads((Path(folder)/'environment.json').read_text()) if (Path(folder)/'environment.json').exists() else {}
    head_count=environment.get('heads')
    for row in rows:
        if job.get('device_class')=='cuda':
            context=Path(folder)/'invocations'/row['invocation_id']/'environment.json'
            assert context.exists(),'Missing immutable invocation environment'
            env=json.loads(context.read_text());assert env['device'].startswith('cuda'),'CPU observations cannot complete a GPU job'
            assert env['model_dtype']=='torch.'+job['dtype']
            if (ROOT/'SCIENTIFIC_LOCK.json').exists():assert row['science_version']==json.loads((ROOT/'SCIENTIFIC_LOCK.json').read_text())['version']
        kind=job['kind']
        if kind=='probe':
            assert len(row['input_ids'])==row['budget'] and 0<=row['prefix_tokens']<=row['budget']
            assert len({h['head'] for h in row['heads']})==len(row['heads'])==head_count
            for h in row['heads']:
                assert h['coverage']==(h['states']==16)
                if h['coverage']:
                    d=h['diagnostics'];assert h['parameter_ok']==(d['cos_a']>=.999 and d['S_min_signed_cos']>=.999 and d['S_bijection'])
            assert row['coverage']==all(h['coverage'] for h in row['heads'])
            assert row['calibration']==all(h['parameter_ok'] and h['decoder_ok'] for h in row['heads'])
        elif kind in ['latency','calibration_cost']:
            assert row['seconds']>=0
            if kind=='latency' and row['operation']!='prefill':
                assert abs(row['seconds_per_1000_tokens']-row['seconds']*1000/512)<1e-10
                assert abs(row['milliseconds_per_GiB']-row['seconds']*1000/(row['cache_bytes']/(1<<30)))<1e-8
        elif kind=='main':
            assert set(row['seeds'])=={'5001','5002'}
            for score in row['seeds'].values():check_score(score,row['truth'],job['block'],head_count)
        elif kind in ['lifecycle','sa_rotation']:
            names={'same_epoch','stale_after_rotation'}|({'recalibrated'} if kind=='lifecycle' else set())
            assert set(row['conditions'])==names
            for score in row['conditions'].values():check_score(score,row['truth'],16)
        elif kind=='cross_key':check_score(row['scores'],row['truth'],16)
        elif kind=='fused':check_score(row['scores'],row['truth'],16)
        elif kind=='ratio':
            check_score(row['scores']['same_epoch'],row['truth'],16)
            if 'heads' in row['scores']['adapted']:check_score(row['scores']['adapted'],row['truth'],16)
        elif kind=='privacy':
            from .studies import topk
            assert len(row['candidate_tokens'])==100 and 0<=row['true_idx']<100
            for label,record in [('same_epoch',row['observed']),('wrong_secret',row['wrong_secret'])]:
                check_score(record,row['truth'],16,head_count)
                bags=[Counter(sum(h['bags'],[])) for h in record['heads']]
                scores=[sum(2*sum((bag&Counter(c)).values())/(sum(bag.values())+len(c)) if sum(bag.values())+len(c) else 0. for bag in bags)/len(bags) for c in row['candidate_tokens']]
                assert scores==row['identification'][label]['scores']
                assert row['identification'][label]['topk']=={str(k):topk(scores,row['true_idx'],k) for k in [1,5,10]}
        elif kind=='direction':
            assert sorted(row['candidate_order'])==[0,1] and len(row['observations'])==2
            for observation in row['observations']:
                target=observation['target'];assert observation['truth']==row['candidate_tokens'][target]
                for global_only in [False,True]:
                    seen=[sum(observation['bags'],[])] if global_only else observation['bags']
                    values=[]
                    for candidate in row['candidate_order']:
                        tokens=row['candidate_tokens'][candidate]
                        offered=[tokens] if global_only else [tokens[i:i+16] for i in range(0,len(tokens),16)]
                        values.append(2*sum(sum((Counter(a)&Counter(b)).values()) for a,b in zip(seen,offered))-sum(map(len,seen))-sum(map(len,offered)))
                    winners=[i for i,v in enumerate(values) if v==max(values)]
                    result=observation['comparisons']['global_bag' if global_only else 'block_bags']
                    assert result['scores']==values
                    assert result['accuracy_credit']==(1/len(winners) if row['candidate_order'].index(target) in winners else 0.)
        elif kind=='ordering' and not row.get('calibration_failed'):
            assert set(row['conditions'])=={'end2end','oracle_ms','no_cross_block','random_order'}
            for name,record in row['conditions'].items():
                pred=record['prediction'];truth=row['truth'];assert len(pred)==len(truth)
                assert record['position_correct']==sum(a==b for a,b in zip(pred,truth))
                bags=row['oracle_bags'] if name=='oracle_ms' else row['bags']
                assert all(Counter(pred[i:i+16])==Counter(bags[i//16]) for i in range(0,len(pred),16))
        elif kind in ['byte_defense','sa_functional','large_byte']:
            if row.get('duplicate_plain_sanity') is not None:assert row['duplicate_plain_sanity']
            if kind=='byte_defense':
                assert len(row.get('segments',[]))+row.get('dependent_consumer_steps_not_run',0)==row['expected_consumer_steps']
                if 'checks' in row:assert row['functional_pass']==all(row['checks'].values())
            elif kind=='sa_functional':
                assert row['M_fixed'] and len(row['segments'])==3
                assert row['functional_pass']==all(s['functional_pass'] for s in row['segments'])
            else:assert len(row['segments'])==3 and row['functional_pass']==all(row['checks'].values())
            for segment in row.get('segments',[]):
                diff=segment['logits']
                assert abs(abs(diff['worst_observed']-diff['worst_plain'])-diff['maximum_absolute_error'])<1e-5
                assert diff['argmax_equal']==(diff['observed_argmax']==diff['plain_argmax'])
        elif kind in ['spectrum','matched']:
            assert len(row['candidates'])==100 and 0<=row['true_idx']<100
            for methods in row['seeds'].values():
                for result in methods.values():
                    scores=result['scores'];assert len(scores)==100 and result['chosen']==max(range(100),key=lambda i:scores[i])
                    assert result['hit']==(result['chosen']==row['true_idx'])
        elif kind=='official_utility':
            if job['benchmark']=='mmlu':assert row['correct']==(row['prediction']==row['truth'])
            else:
                from .utility import normalize
                assert row['correct']==any(normalize(row['prediction'])==normalize(t) for t in row['truths'])
        elif kind=='marker_numerics':
            assert row['official_forward_used']
            for result in row['conditions'].values():assert result['correct_rows']==sum(a==b for a,b in zip(result['predictions'],row['truth']))
    if job['kind']=='sa_functional':
        by={r['sid']:r for r in rows}
        for key,row in by.items():
            if key.endswith('|original'):assert row['paired_permutations']==by[key[:-8]+'refresh']['paired_permutations']
    result=dict(job_id=job['id'],complete=True,expected=len(expected),measured=len(rows),normal_negative_outcomes=sum(r.get('functional_pass') is False for r in rows),predictions_and_denominators_checked=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'protocol/jobs.json');p.add_argument('--output',type=Path,default=result_root());p.add_argument('--job');p.add_argument('--completed-only',action='store_true');a=p.parse_args()
    reports=[]
    for job in json.loads(a.manifest.read_text()):
        if a.job and job['id']!=a.job:continue
        folder=a.output/job['id']
        if a.completed_only and not (folder/'summary.json').exists():continue
        reports.append(verify(job,folder))
    print(json.dumps(reports,indent=2))
