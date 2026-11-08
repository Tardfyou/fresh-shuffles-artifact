"""Recompute the manuscript's selected model scope from saved records only."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def summarize(root, output):
    results = root/'results'
    main = results/'main_public'
    samples = [json.loads(line) for line in (main/'samples_main_public.jsonl').open()]
    order = {r['sid']: i for i, r in enumerate(samples)}
    assert len(samples) == len(order) == 3000
    groups = {}
    for r in samples:
        groups.setdefault(r['len_bucket'], []).append(order[r['sid']])
    assert sorted(map(len, groups.values())) == [600]*5
    cs = np.zeros(3000, dtype=np.int64)
    ns = np.zeros(3000, dtype=np.int64)
    configs = {}
    original = Counter()
    for path in sorted(main.glob('*/per_sample.jsonl')):
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == len({r['sid'] for r in rows}) == 3000
        assert {r['sid'] for r in rows} == set(order)
        summary = read(path.parent/'summary.json')
        c = n = 0
        heads = int(summary['heads_with_secrets'].split('/')[0])
        for r in rows:
            assert set(r['seeds']) == {'seed5001', 'seed5002'}
            for cell in r['seeds'].values():
                assert cell['total'] == (r['ntok']//summary['b'])*summary['b']*heads
                assert 0 <= cell['correct'] <= cell['total']
                c += cell['correct']
                n += cell['total']
                if path.parent.name != 'phi3_mini':
                    cs[order[r['sid']]] += cell['correct']
                    ns[order[r['sid']]] += cell['total']
        assert (c, n) == (summary['correct'], summary['total'])
        original['correct'] += c
        original['total'] += n
        if path.parent.name == 'phi3_mini':
            continue
        configs[path.parent.name] = dict(model=summary['model'], block_size=summary['b'], heads=heads,
                                        prompts=3000, correct=c, total=n, misses=n-c,
                                        recovery_percent=100*c/n)
    assert (original['correct'], original['total']) == (153274758, 153791424)
    assert len(configs) == 11
    c, n = int(cs.sum()), int(ns.sum())
    assert (c, n) == (91528679, 91611072)
    rng = np.random.default_rng(20261029)
    rates = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups.values()])
        rates.append(float(cs[idx].sum()/ns[idx].sum()))
    ci = (100*np.quantile(rates, [.025, .975])).tolist()
    evidence = read(results/'new_evidence_20261102/summary.json')
    e1 = {k: v for k, v in evidence['E1'].items() if 'phi' not in k.lower()}
    sessions = [s for model in e1.values() for s in model['sessions'].values()]
    e1_totals = dict(session_head_observations=sum(v['session_head_observations'] for v in e1.values()),
                     unique_input_head_observations=sum(v['unique_input_head_observations'] for v in e1.values()),
                     cached_event_rows=sum(v['cached_attempt_rows'] for v in e1.values()),
                     same_input_events=sum(v['same_as_primary_input_event_rows'] for v in e1.values()),
                     different_input_events=sum(v['different_from_primary_input_event_rows'] for v in e1.values()),
                     content_correct=sum(s['content_correct'] for s in sessions),
                     content_total=sum(s['content_head_token_opportunities'] for s in sessions))
    visual = read(results/'visual_summary_20261102/data.json')
    epochs = {k: v for k, v in visual['epochs'].items() if 'phi' not in k.lower()}
    assert len(epochs) == 2
    supplementary = read(results/'supplementary_reporting_values.json')
    diagnostics = [r for r in supplementary['calibration_diagnostics'] if 'phi' not in r['model'].lower()]
    assert len(diagnostics) == 11
    extrema = dict(minimum_state_separation=min(float(r['minimum_state_separation']) for r in diagnostics),
                   minimum_matched_S_cosine=min(float(r['minimum_matched_S_cosine']) for r in diagnostics),
                   maximum_w_relative_error=max(float(r['maximum_w_relative_error']) for r in diagnostics))
    localization = {}
    for tag in ['qwen25_05b', 'llama32_1b']:
        row_path = results/'saved_record_review_20261101'/tag/'failed_block_diagnostics.jsonl.gz'
        cell = Counter()
        seen = set()
        with gzip.open(row_path, 'rt') as stream:
            for line in stream:
                saved = json.loads(line)
                r = saved['block']
                key = (saved['sid'], saved['seed'], r['h'], r['k'])
                assert key not in seen
                seen.add(key)
                assert len(r['rows']) == 16 and 0 <= r['correct'] < 16 and 0 <= r['j'] < 16
                pred = [row['p'] for row in r['rows']]
                gold = {g['tok'] for g in r['gold']}
                novel = [i for i, t in enumerate(pred) if t not in gold]
                absent = [g for g in r['gold'] if g['tok'] not in pred]
                cell['failed_blocks'] += 1
                cell['missed_head_tokens'] += 16-r['correct']
                cell['single_miss_blocks'] += r['correct'] == 15
                cell['novel_prediction_rows'] += len(novel)
                cell['novel_prediction_rows_at_selected_marker'] += sum(i == r['j'] for i in novel)
                cell['blocks_with_one_novel_prediction_at_selected_marker'] += novel == [r['j']]
                cell['single_error_blocks_with_one_novel_prediction_at_selected_marker'] += (r['correct'] == 15 and novel == [r['j']])
                cell['absent_gold_types'] += len(absent)
                cell['absent_gold_best_row_at_selected_marker'] += sum(g['row'] == r['j'] for g in absent)
        assert cell['missed_head_tokens'] == configs[tag]['misses']
        localization[tag] = cell
    result = dict(date='2026-11-08', scope='Nine models, eleven configurations; fixed 3000 prompts, all heads and both permutation seeds.',
                  scope_change=dict(excluded_model='Phi-3-mini', reason='Author-selected reduction of the extended-model evaluation scope.',
                                    original_records_retained=True, exclusion_is_not_a_data_quality_finding=True),
                  model_count=9, configuration_count=11, configurations=configs,
                  correct=c, total=n, misses=n-c, recovery_percent=100*c/n,
                  prompt_stratified_bootstrap_95_percent=ci,
                  bootstrap=dict(replicates=2000, seed=20261029, unit='Whole prompt, paired across configurations, heads and permutation seeds; fixed keys.'),
                  error_followup=dict(configurations=['qwen25_05b', 'llama32_1b'], covered_misses=37466,
                                      total_misses=n-c, coverage_percent=100*37466/(n-c),
                                      row_localization=localization,
                                      limitation='Selected marker-cancellation row is not the recorded true marker row; localization does not establish a unique physical cause.'),
                  E1=e1, E1_totals=e1_totals, epochs=epochs, calibration_extrema=extrema,
                  original_full_scope_counts=dict(original), new_model_or_decoder_execution=False)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['correct', 'total', 'misses', 'recovery_percent', 'prompt_stratified_bootstrap_95_percent', 'E1_totals', 'calibration_extrema']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, default=Path('results/reporting_scope_20261108'))
    a = p.parse_args()
    summarize(a.root, a.out)
