"""Verify the frozen figure inputs using only saved numbers and Python stdlib."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def quantile(values, q):
    ordered=sorted(values); location=(len(ordered)-1)*q
    lo=math.floor(location); hi=math.ceil(location)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(location-lo)


def inspect(root):
    data=read(root/'results/visual_summary_20261102/data.json')
    manifest=[json.loads(line) for line in (root/'results/main_public/samples_main_public.jsonl').open() if line.strip()]
    expected={r['sid'] for r in manifest if r['or_subset']}; assert len(expected)==602
    local=data['semantic_local'];assert len(local)==602 and {r['sid'] for r in local}==expected
    for condition,report in data['semantic_summary']['conditions'].items():
        values=[r['conditions'][condition] for r in local]
        for key,value in [('mean',statistics.mean(values)),('median',statistics.median(values)),('p25',quantile(values,.25)),('p75',quantile(values,.75))]:
            assert abs(value-report[key])<1e-12
    for model,records in data['api_scores'].items():
        assert len(records)==602 and {r['sid'] for r in records}==expected
        report=data['api_summary']['models'][model]
        for key,field,scale in [('position_accuracy','macro_position_percent',100),('multiset_recall','macro_multiset_recall_percent',100),('multiset_precision','macro_multiset_precision_percent',100),('embedding_cosine','mean_embedding_cosine',1)]:
            assert abs(scale*statistics.mean(r[key] for r in records)-report[field])<1e-10
    privacy=data['privacy_cases'];assert len(privacy)==len({r['sid'] for r in privacy})==100
    for kind,report in data['privacy_summary']['by_type'].items():
        group=[r for r in privacy if r['etype']==kind]
        assert len(group)==report['n'] and abs(statistics.mean(r['topk_expected']['1'] for r in group)-report['top1'])<1e-12
    assert abs(statistics.mean(r['topk_expected']['1'] for r in privacy)-.975)<1e-12
    assert sum(r['unique_top1'] for r in privacy)==95
    for model,report in data['epochs'].items():
        assert report['n']==500 and len(report['by_key'])==5
        for cell in report['by_key'].values():
            assert abs(100*cell['correct']/cell['total']-cell['percent'])<1e-10
    reference=read(root/'results/new_evidence_20261102/summary.json')
    assert data['scaling']==reference['E4'] and data['budget_comparison']==reference['E2']
    for point in data['scaling']:
        assert sum(point[k] for k in ['sample_ids_improved','sample_ids_unchanged','sample_ids_degraded'])==100
    assert len(data['cache_timing'])==8
    for cell in data['cache_timing'].values():
        values=cell['samples_seconds_per_1000_tokens'];assert len(values)==100 and all(math.isfinite(x) and x>0 for x in values)
        for field,q in [('p50',.5),('p95',.95),('p99',.99)]:assert abs(quantile(values,q)-cell[field])<1e-12
    return {'figure_inputs_pass':True,'privacy_cases':100,'cosine_scores':6*602,'timed_calls':800,'scaling_points':6,'model_key_cells':15,'new_measurements':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    print(json.dumps(inspect(args.root),indent=2))
