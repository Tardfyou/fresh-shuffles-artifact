"""Check saved marker precision diagnostics without inference or model loading."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import torch


def inspect(root):
    data = root/'results/marker_numerical_diagnostics_20261109'
    rows = [json.loads(line) for line in (data/'checks.jsonl').open()]
    fixtures = [json.loads(line) for line in (data/'fixtures.jsonl').open()]
    protocol = json.loads((data/'PROTOCOL.json').read_text())
    summary = json.loads((data/'summary.json').read_text())
    assert len(rows) == len(fixtures) == 446
    seen = set()
    totals = Counter()
    by_tag = Counter()
    for row, fixture in zip(rows, fixtures):
        key = (row['tag'], row['sid'], row['seed'], row['head'], row['block'])
        assert key not in seen
        seen.add(key)
        assert key == (fixture['tag'], fixture['sid'], fixture['seed'], fixture['h'], fixture['k'])
        assert row['true'] == fixture['true'] and row['prior_correct'] == fixture['orig_correct']
        assert row['official_forward_bit_equal']
        assert row['candidate_ids'] == sorted(set(row['true']+fixture['variants']['base']['pred']))
        assert sum((Counter(row['true']) & Counter(fixture['variants']['base']['pred'])).values()) == row['prior_correct']
        by_tag[row['tag']] += 1
        trace = row['marker_addition_trace']
        value = torch.tensor(trace['bf16_value_after_M'], dtype=torch.float64)
        marker = torch.tensor(trace['stored_marker'], dtype=torch.float64)
        rounded = torch.tensor(trace['bf16_sum'], dtype=torch.float64)
        assert torch.equal((value+marker).to(torch.bfloat16).double(), rounded)
        assert int((value != 0).sum()) == trace['nonzero_value_coordinates']
        assert int(((value != 0) & (rounded == marker)).sum()) == trace['swallowed_nonzero_coordinates']
        for condition, record in row['conditions'].items():
            pred = record['fixed_candidate_predictions']
            correct = sum(t == p for t,p in zip(row['true'], pred))
            assert len(pred) == 16 and correct == record['correct_rows']
            mr = row['reconstructed_true_marker_row']
            assert record['marker_row_correct'] == (pred[mr] == row['true'][mr])
            for i,(actual,other,margin) in enumerate(zip(record['true_score'], record['best_other_score'], record['margin'])):
                assert abs((actual-other)-margin) < 1e-12
                if margin > 1e-12:
                    assert pred[i] == row['true'][i]
                elif margin < -1e-12:
                    assert pred[i] != row['true'][i]
            assert all(math.isfinite(e) and e >= 0 for e in record['relative_vector_error'])
            totals[condition] += 16-correct
            if condition == 'fp64_sanity':
                assert correct == 16 and max(record['relative_vector_error']) < 1e-10
        totals['official_bitwise_matches'] += row['official_forward_bit_equal']
    assert dict(by_tag) == protocol['expected_by_tag'] == {'qwen25_05b': 66, 'llama32_1b': 380}
    assert totals['official_bitwise_matches'] == 446
    assert totals['bf16_known_key'] == 40
    assert totals['bf16_no_marker'] == totals['fp32_forward'] == totals['fp64_sanity'] == 0
    for tag in by_tag:
        for label,failed in [('previously_failed', True), ('previously_correct', False)]:
            group = [r for r in rows if r['tag'] == tag and (r['prior_correct'] < 16) == failed]
            saved = summary[tag][label]
            assert len(group) == saved['blocks']
            for condition in group[0]['conditions']:
                marker_wrong = sum(not r['conditions'][condition]['marker_row_correct'] for r in group)
                other_wrong = sum(16-r['conditions'][condition]['correct_rows']-int(not r['conditions'][condition]['marker_row_correct']) for r in group)
                expected = saved[condition]
                assert (marker_wrong, other_wrong) == (expected['wrong_marker_rows'], expected['wrong_other_rows'])
                error = statistics.median(r['conditions'][condition]['relative_vector_error'][r['reconstructed_true_marker_row']] for r in group)
                assert abs(error-expected['median_relative_error_at_marker']) < 1e-12
    return dict(saved_checks_pass=True, blocks=446, official_bitwise_matches=446,
                bf16_candidate_misranked_positions=40, fp32_candidate_misranked_positions=0,
                bf16_without_marker_misranked_positions=0, model_or_inference_calls=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    print(json.dumps(inspect(p.parse_args().root), indent=2))
