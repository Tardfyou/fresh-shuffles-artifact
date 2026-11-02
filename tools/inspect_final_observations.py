"""Inspect saved descriptive observations offline, without models or APIs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.open() if line.strip()]


def inspect(root):
    data=root/'results/final_record_observations_20261102';summary=read(data/'summary.json')
    history={r['sid']:r for r in rows(root/'results/main_public/phi3_mini/per_sample.jsonl')}
    observed=rows(data/'phi_seed_counters.jsonl');seen=set();correct=total=0;differences=[]
    for row in observed:
        key=(row['sid'],row['seed']);assert key not in seen;seen.add(key)
        old=history[row['sid']]['seeds'][row['seed']]
        assert row['total']==old['total'];correct+=row['correct'];total+=row['total']
        if row['correct']!=old['correct']:differences.append((row['sid'],row['seed'],row['correct']-old['correct']))
    assert seen=={(sid,seed) for sid in history for seed in ['seed5001','seed5002']}
    assert len(seen)==6000 and len(differences)==6 and sum(x[2] for x in differences)==0
    assert (correct,total)==(summary['phi_counter_audit']['correct'],summary['phi_counter_audit']['total'])
    grand_misses=sum(read(path)['total']-read(path)['correct'] for path in (root/'results/main_public').glob('*/summary.json'))
    covered=total-correct
    for tag in ['qwen25_05b','llama32_1b']:
        old=read(root/'results/main_public'/tag/'summary.json');covered+=old['total']-old['correct']
    assert (covered,grand_misses)==(471739,516666)
    assert summary['miss_coverage']['covered']==covered and abs(summary['miss_coverage']['fraction']-covered/grand_misses)<1e-15
    sessions=read(data/'calibration_sessions.json')['models']
    for model,record in sessions.items():
        values=[sum(rep[k] for k in ['probe_construction_s','probe_prefill_s','probe_protect_s','calibrate_s']) for rep in record['repetitions']]
        assert len(values)==5 and abs(statistics.median(values)-summary['session_costs'][model]['session_median_s'])<1e-12
        assert all(math.isfinite(x) and x>0 for x in values)
    attempts=rows(data/'attempt_labels.jsonl');assert len(attempts)==44
    assert all(row['actual_submitted_tokens_recorded'] is False and row['actual_used_character_recorded'] is False and row['recorded_after_state_count'] is None for row in attempts)
    assert sum(row['requested_label_matches_primary'] for row in attempts)==7
    grouped=defaultdict(list)
    for row in attempts:
        assert row['requested_label_matches_primary']==(row['requested_character']==row['related_primary_character'])
        grouped[row['model']].append(row)
    for model,values in grouped.items():
        record=summary['attempt_label_accounting'][model]
        assert len(values)==record['attempt_rows']
        heads={row['head'] for row in values};assert len(heads)==record['triggered_heads']
        resolved={row['head'] for row in values if not row['requested_label_matches_primary'] and row['reported_threshold_clearance']}
        assert len(resolved)==record['reported_threshold_cleared_heads']
    for model,record in summary['candidate_identification'].items():
        values=rows(data/'candidate_identification'/(model+'.jsonl'))
        assert len(values)==len({row['sid'] for row in values})==record['target_ids']==100
        for rule,observation in record['rules'].items():
            hits=[]
            for seed in ['5001','5002']:
                assert all(row[f'{rule}_seed{seed}']['hit']==(row[f'{rule}_seed{seed}']['chosen']==row['true_idx']) for row in values)
                hits.append(sum(row[f'{rule}_seed{seed}']['hit'] for row in values))
            assert hits==observation['hits_per_100_by_seed']
    point=rows(data/'scaling_point_scores.jsonl');expected=set(read(root/'results/main_public/experiment_subsets.json')['need_ratio'])
    assert len(point)==100 and {row['sid'] for row in point}==expected
    sums={condition:{field:sum(row['scores'][condition][field] for row in point) for field in ['correct','total']} for condition in ['same_epoch','adapted']}
    assert sums==summary['scaling_point']['scores']
    assert summary['scaling_point']['general_method_validity_accepted'] is False
    return {'saved_numeric_checks_pass':True,'phi_seed_records':6000,'historical_differing_cells':6,'coverage_percent':100*covered/grand_misses,'request_label_rows':44,'same_label_requests':7,'candidate_target_ids_per_model':100,'scaling_prompt_ids':100,'scope':'Record/number checks only; not input-object identity, mechanism or method correctness verification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();print(json.dumps(inspect(args.root),indent=2))
