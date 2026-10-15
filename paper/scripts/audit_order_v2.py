#!/usr/bin/env python3
"""Reconcile retained ordering metrics; no model calls or inference reruns."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evidence' / 'order_v2'
CONDITIONS = ['end2end', 'random_order', 'oracle_ms', 'no_cross_block']
METRIC_FIELDS = ['position_accuracy', 'exact_match', 'block_exact', 'block_total',
                 'trigram_overlap', 'multiset_conservation', 'pa_by_block',
                 'n_true_tokens', 'n_pred_tokens']


def read_jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def aggregate(rows):
    result = {}
    for condition in CONDITIONS:
        values = [row[condition] for row in rows]
        result[condition] = {
            'macro_position_accuracy': statistics.mean(v['position_accuracy'] for v in values),
            'exact_records': sum(v['exact_match'] for v in values),
            'exact_blocks': sum(v['block_exact'] for v in values),
            'blocks': sum(v['block_total'] for v in values),
            'tokens': sum(v['n_true_tokens'] for v in values),
            'conservation_min': min(v['multiset_conservation'] for v in values),
            'conservation_max': max(v['multiset_conservation'] for v in values),
        }
    return result


def summarize():
    tokenization = json.loads((DATA / 'TOKENIZATION_AUDIT.json').read_text())
    tokens = {row['id']: row for row in tokenization['rows']}
    result = {'scope': 'Descriptive re-analysis of retained summaries and source; no model reruns',
              'provenance': 'Retrospective delivery snapshot and retrospective tokenization',
              'phase_files': {}, 'warnings': []}
    phase_metrics, phase_samples = {}, {}
    for phase, expected in [('main', 600), ('boundary', 300)]:
        samples = read_jsonl(f'samples_{phase}.jsonl')
        metrics = read_jsonl(f'metrics_{phase}.jsonl')
        ids = [m['id'] for m in metrics]
        assert len(ids) == len(set(ids)) == expected
        assert set(ids) == {s['id'] for s in samples}
        smap = {s['id']: s for s in samples}
        phase_metrics[phase], phase_samples[phase] = metrics, smap
        files = {}
        for kind in ['samples', 'metrics', 'predictions', 'costs']:
            rows = read_jsonl(f'{kind}_{phase}.jsonl')
            counts = Counter(row['id'] for row in rows)
            files[kind] = {'physical_rows': len(rows), 'unique_ids': len(counts),
                           'duplicate_ids': sum(n > 1 for n in counts.values()),
                           'statuses': dict(Counter(row.get('status', 'completed') for row in rows))}
        result['phase_files'][phase] = files
        for m in metrics:
            for cond in CONDITIONS:
                v = m[cond]
                assert 0 <= v['position_accuracy'] <= 1
                assert v['n_true_tokens'] == v['n_pred_tokens'] == 16 * v['block_total']
                assert len(tokens[m['id']]['tokens']) == v['n_true_tokens']
                assert len(v['pa_by_block']) == v['block_total']
                assert v['block_exact'] <= v['block_total']
        unique_spans = {tuple(tokens[m['id']]['tokens']) for m in metrics}
        result[phase] = {
            'records': expected, 'unique_evaluated_spans': len(unique_spans),
            'conditions': aggregate(metrics),
            'oracle_equal_recorded_metric_vectors': sum(
                all(m['end2end'][key] == m['oracle_ms'][key] for key in METRIC_FIELDS)
                for m in metrics),
            'source_inferred_global_bag_match_records': sum(
                m['oracle_ms']['multiset_conservation'] == 1.0
                and m['oracle_ms']['n_true_tokens'] == m['oracle_ms']['n_pred_tokens']
                and m['oracle_ms']['n_true_tokens'] < 10000 for m in metrics),
            'actual_block_counts': dict(sorted(Counter(m['end2end']['block_total'] for m in metrics).items())),
        }
    main = phase_metrics['main']
    samples = phase_samples['main']
    cells = []
    for category in ['conversation', 'narrative', 'structured']:
        for nominal in [1, 2, 4, 8]:
            selected = [m for m in main if m['category'] == category
                        and samples[m['id']]['target_blocks'] == nominal]
            assert len(selected) == 50
            counts = Counter(m['end2end']['block_total'] for m in selected)
            cells.append({'category': category, 'nominal_minimum_blocks': nominal,
                          'records': len(selected), 'actual_block_counts': dict(sorted(counts.items())),
                          'conditions': aggregate(selected)})
    result['main']['nominal_cells'] = cells
    result['main']['nominal_length_overshoots'] = sum(
        m['end2end']['block_total'] != samples[m['id']]['target_blocks'] for m in main)
    result['main']['unique_exact_spans'] = len({tuple(tokens[m['id']]['tokens'])
                                               for m in main if m['end2end']['exact_match']})
    boundary = phase_metrics['boundary']
    bmap = {m['id']: m for m in boundary}
    pairs = defaultdict(dict)
    for s in phase_samples['boundary'].values():
        if s['category'] == 'boundary_pair':
            pairs[s['pair_id']][s['direction']] = s['id']
    assert len(pairs) == 100 and all(set(p) == {'a', 'b'} for p in pairs.values())
    pair_audit, pair_signatures = [], set()
    for pair_id, pair in sorted(pairs.items()):
        a, b = [tokens[pair[d]]['tokens'] for d in ['a', 'b']]
        global_equal = Counter(a) == Counter(b)
        block_equal = len(a) == len(b) and all(Counter(a[i:i+16]) == Counter(b[i:i+16])
                                               for i in range(0, len(a), 16))
        pair_audit.append({'pair_id': pair_id, 'global_multiset_equal': global_equal,
                           'all_block_multisets_equal': block_equal})
        pair_signatures.add(tuple(sorted((tuple(a), tuple(b)))))
    for category in ['boundary_pair', 'random_seq']:
        selected = [m for m in boundary if m['category'] == category]
        result['boundary'][category] = {'records': len(selected), 'conditions': aggregate(selected)}
    directions = {}
    for direction in ['a', 'b']:
        selected = [bmap[pair[direction]] for pair in pairs.values()]
        directions[direction] = {'records': len(selected), 'conditions': aggregate(selected)}
    supplied = json.loads((DATA / 'boundary_direction_analysis.json').read_text())
    assert supplied['direction_bias']['pa_dir_a'] == round(directions['a']['conditions']['end2end']['macro_position_accuracy'], 4)
    assert supplied['direction_bias']['random_order'] == round(directions['a']['conditions']['random_order']['macro_position_accuracy'], 4)
    result['boundary']['paired_audit'] = {
        'pairs': len(pairs), 'unique_unordered_token_sequence_pairs': len(pair_signatures),
        'global_multiset_equal_pairs': sum(p['global_multiset_equal'] for p in pair_audit),
        'all_block_multisets_equal_pairs': sum(p['all_block_multisets_equal'] for p in pair_audit),
        'direction_groups': directions, 'per_pair': pair_audit,
        'supplied_direction_analysis_metric': 'position accuracy grouped by direction; no direction classifier',
        'supplied_random_reference_scope': 'direction-a records only',
    }
    dev = read_jsonl('metrics_dev.jsonl')
    result['dev'] = {'records': len(dev), 'oracle_equal_recorded_metric_vectors': sum(
        all(m['end2end'][key] == m['oracle_ms'][key] for key in METRIC_FIELDS) for m in dev)}
    result['warnings'] = [
        'Conservation compares sorted output with recovered input. In oracle mode the output is a truth-bag permutation, providing indirect whole-span overlap evidence, not per-block evidence.',
        'Equal recorded end2end/oracle scores do not prove equal multisets or equal output sequences.',
        'No per-token predictions or recovered counters retained for any phase in this delivery.',
        'Nominal bucket labels are minimum generation targets; 109/600 actual spans overshoot.',
        'Main contains 581 distinct evaluated token spans, not 600 unique texts.',
        'Boundary paired inputs agree globally but differ in blockwise bags in every pair.',
        'Position accuracy by direction is not factual direction-identification accuracy.',
        'The supplied 6.69% random comparison selects direction-a only; both directions give 7.12%.',
        'Random shuffling restarts its RNG stream across resumes; report retained descriptive outcomes.',
        'No independent pre-run registration; source snapshots and protocol are retrospective.',
        'The analytic random_baseline_pa source uses the whole span, not per-block expectation for multi-block inputs.',
        'Two configured ordering passes reset context; no previous-pass ordering is fed back.',
    ]
    result['global_bag_inference'] = (
        'Source inspection: oracle_ms supplies true per-block Counters to a count-consuming beam; '
        'evaluate then compares its output with the recovered bags pooled over all blocks. '
        'Unit conservation with equal lengths and spans shorter than 10000 tokens rules out '
        'one missing token being rounded away. This implies whole-span bag agreement under '
        'the inspected implementation; it is not a retained raw counter comparison or a '
        'per-block agreement measurement.')
    return result


if __name__ == '__main__':
    result = summarize()
    out = ROOT / 'review' / 'order_v2_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['main', 'boundary']}, indent=2))
    print('Main macro PA:', result['main']['conditions']['end2end']['macro_position_accuracy'])
    print('Main exact records:', result['main']['conditions']['end2end']['exact_records'])
    print('Main nominal overshoots:', result['main']['nominal_length_overshoots'])
    print('Blockwise-identical boundary pairs:', result['boundary']['paired_audit']['all_block_multisets_equal_pairs'])
    print(out)
