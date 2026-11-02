"""Independently reconcile distributed saved observations; no model or API calls."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def median_hist(hist):
    n = sum(count for _, count in hist)
    ranks = [(n-1)//2, n//2]
    values = []
    offset = 0
    for value, count in sorted(hist):
        values += [value for rank in ranks if offset <= rank < offset+count]
        offset += count
    assert len(values) == 2
    return statistics.mean(values)


def inspect(root):
    data = root/'results/new_evidence_20261102'
    summary = read(data/'summary.json')
    subsets = read(root/'results/main_public/experiment_subsets.json')
    counts = Counter()
    for model, heads in [('Qwen2.5-0.5B', 2), ('Llama-3.2-1B', 8), ('phi3_mini', 32)]:
        record = read(data/f'e1_a5_full_ladder_{model}.json')
        metrics = read(data/f'e1_m_true_error_{model}.json')
        sessions = {'abxyz1e0': record['primary_session'], **record['alternate_sessions']}
        unique_inputs = set()
        for name, session in sessions.items():
            assert len(session['probe_token_ids']) == session['n_tokens']
            unique_inputs.add(tuple(session['probe_token_ids']))
            assert len(session['heads']) == heads
            numerator = denominator = 0
            for h, cell in session['heads'].items():
                gt = cell['ground_truth']
                expected = gt['cos_a'] >= .999 and gt['S_min_signed_cos'] >= .999 and gt['S_bijection']
                assert cell['parameter_ok'] == expected
                counts['a_S_criterion_passes'] += expected
                measured = metrics['sessions'][name]['heads'][h]
                angles = measured['angle_errs_wrapped']
                assert len(angles) == measured['angle_pairs_total'] == measured['angle_pairs_used']
                assert all(math.isfinite(a) and -math.pi < a <= math.pi for a in angles)
                assert abs(max(map(abs, angles))-measured['angle_err_abs_max']) <= 1.1e-8
                assert abs(statistics.mean(map(abs, angles))-measured['angle_err_abs_mean']) <= 1.1e-8
            for block in session['content_recovery']['blocks']:
                for cell in block['heads'].values():
                    numerator += cell['correct']; denominator += cell['total']
            assert abs(numerator/denominator-session['content_recovery']['mean_correct_frac']) < .00000051
            counts['content_correct'] += numerator; counts['content_total'] += denominator
        counts['unique_input_head_observations'] += len(unique_inputs)*heads
    assert (counts['a_S_criterion_passes'], counts['unique_input_head_observations'], counts['content_correct'], counts['content_total']) == (126, 116, 10020, 10080)
    for model in ['Qwen2.5-0.5B', 'Llama-3.2-1B']:
        records = rows(data/f'e2_a6_budget_cross_{model}.jsonl')
        assert len(records) == len({r['sid'] for r in records}) == 100
        previous = rows(root/'results/final_record_observations_20261102/candidate_identification'/(model+'.jsonl'))
        assert {r['sid'] for r in records} == {r['sid'] for r in previous}
        for rule, seed_hits in summary['E2'][model]['hits_by_seed'].items():
            for seed, hits in seed_hits.items():
                assert all(r['seeds'][seed][rule]['hit'] == (r['seeds'][seed][rule]['chosen'] == r['true_idx']) for r in records)
                assert sum(r['seeds'][seed][rule]['hit'] for r in records) == hits
    for tag in ['qwen25_05b', 'llama32_1b']:
        records = rows(data/f'e3b_flip_{tag}_v2.jsonl')
        assert len(records) == len({(r['sid'], r['seed'], r['h'], r['k']) for r in records})
        for record in records:
            assert len(record['variants']) == 9
            for variant in record['variants'].values():
                assert len(variant['pred']) == 16 and variant['pred'] == record['variants']['base']['pred']
                assert sum((Counter(variant['pred']) & Counter(record['true'])).values()) == variant['correct']
                counts['prediction_arrays'] += 1
    assert counts['prediction_arrays'] == 2754
    histograms = read(data/'gap_histograms.json')['models']
    for tag, model in histograms.items():
        hist = model['absent_token_block_gap_counts']; expected = summary['E3a'][tag]
        assert sum(n for _, n in hist) == expected['distributions']['gap_fully_missed_tokens']['n_distinct_token_block_entries']
        assert median_hist(hist) == expected['distributions']['gap_fully_missed_tokens']['median']
        assert sum(n for gap, n in hist if gap >= .0101) == expected['absent_distinct_entries_with_stored_gap_ge_0.0101']
    for point in summary['E4']:
        records = rows(data/f"e4_{point['model']}_r{point['ratio']}.jsonl")
        assert len(records) == len({r['sid'] for r in records}) == 100
        assert {r['sid'] for r in records} == set(subsets['need_ratio'])
        assert all(r['status'] == 'ok' for r in records)
        for condition in ['same_epoch', 'adapted']:
            observed = {field: sum(r['scores'][condition][field] for r in records) for field in ['correct', 'total']}
            assert observed == point[condition]
        deltas = [r['scores']['adapted']['correct']-r['scores']['same_epoch']['correct'] for r in records]
        assert sum(deltas) == point['net_head_token_gain']
        assert sum(delta < 0 for delta in deltas) == point['sample_ids_degraded']
        counts['scaling_points'] += 1
    return {'saved_record_checks_pass': True, 'counts': dict(counts),
            'scope': 'Offline record checks; no model inference or mechanism proof.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2))
