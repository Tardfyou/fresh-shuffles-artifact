"""Inspect published fixed-cohort counters and protocol accounting offline."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from result_checks import suite,cross_key,ordering,direction,Pending,require
from battery_evidence import summarize


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.open() if line.strip()]


def inspect(root):
    results=root/'results/takeover_20261029/corrected';main=root/'results/main_public'
    subsets=read(main/'experiment_subsets.json');samples={r['sid']:r for r in rows(main/'samples_main_public.jsonl')}
    a=SimpleNamespace(results=results)
    lifecycle=suite(a,subsets,'lifecycle');rotation=suite(a,subsets,'sa_rotation');keys=cross_key(a,subsets)
    local=ordering(a,samples);pairs=direction(a,subsets);battery=summarize(results,Pending,require)
    for name,expected in [('fused',subsets['fused']),('need_ratio',subsets['need_ratio'])]:
        values=rows(results/(name+'.jsonl'))
        assert len(values)==len({r['sid'] for r in values})==len(expected)
        assert {r['sid'] for r in values}==set(expected) and all(r['status']=='ok' for r in values)
    privacy=rows(results/'privacy.jsonl');summary=read(results/'privacy_summary.json')
    assert len(privacy)==len({r['sid'] for r in privacy})==100
    assert sum(r['unique_top1'] for r in privacy)==summary['unique_top1']==95
    assert abs(sum(r['topk_expected']['1'] for r in privacy)/100-.975)<1e-12
    controls=rows(results/'privacy_controls.jsonl');assert len(controls)==100 and {r['sid'] for r in controls}=={r['sid'] for r in privacy}
    for model in ['Qwen2.5-0.5B','Llama-3.2-1B']:
        values=rows(results/f'spectrum_{model}.jsonl');assert len(values)==len({r['sid'] for r in values})==100
    return {'fixed_cohort_checks_pass':True,'lifecycle_ids':lifecycle['n'],'row_rotation_ids':rotation['n'],
            'cross_key_records':sum(r['n'] for r in keys.values()),'ordering_ids':local['n'],'entity_swap_pairs':pairs['n'],
            'privacy_cases':100,'fused_ids':300,'scaling_baseline_ids':100,'defense_battery_completed_stages':battery['stages'],
            'new_model_execution':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    print(json.dumps(inspect(args.root),indent=2))
