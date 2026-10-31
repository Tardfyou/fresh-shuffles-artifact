"""Verify retained GPT ordering responses and summarize the selected cohort offline."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

MODELS = ('gpt-6.1-sol', 'gpt-6-luna')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(results, manifest):
    from transformers import AutoTokenizer
    protocol = json.loads((results/'api_protocol.json').read_text())
    assert tuple(protocol['models']) == MODELS
    samples = read(manifest)
    expected = {r['sid'] for r in samples if r['or_subset']}
    inputs = read(results/protocol['inputs'])
    by_id = {r['sid']: r for r in inputs}
    assert len(samples) == 3000 and len(expected) == len(inputs) == len(by_id) == 602
    assert set(by_id) == expected
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)
    summaries = {}
    prompts = {}
    audit = {}
    for model in MODELS:
        raw_path = results/'api_ready'/f'raw_{model}.jsonl'
        scores_path = results/f'api_{model}_scores.jsonl'
        raw_before, scores_before = raw_path.read_bytes(), scores_path.read_bytes()
        raw, scores = read(raw_path), read(scores_path)
        scored = {r['sid']: r for r in scores}
        assert len(raw) == len({r['sid'] for r in raw}) == len(scores) == len(scored) == 602
        assert {r['sid'] for r in raw} == set(scored) == expected
        values = []
        settings = {}
        for response in raw:
            sid = response['sid']
            assert response['model'] == model and response['status'] == 'ok'
            if sid in prompts:
                assert response['prompt'] == prompts[sid]
            else:
                prompts[sid] = response['prompt']
            config = response['settings']
            assert config['max_tokens'] == 4096 and config['temperature'] == 0.0
            assert config['reasoning_effort'] == protocol['model_settings'][model]['reasoning_effort']
            settings[json.dumps(config, sort_keys=True)] = config
            truth = by_id[sid]['true_tokens']
            text = response['predicted_text']
            prediction = tokenizer(text).input_ids if text else []
            record = scored[sid]
            offered = Counter({int(k): v for k, v in by_id[sid]['recovered_ms'].items()})
            overlap = sum((Counter(truth) & Counter(prediction)).values())
            correct = sum(a == b for a, b in zip(truth, prediction))
            recall = overlap/len(truth)
            precision = overlap/len(prediction) if prediction else 0.0
            assert record['model'] == model and record['prediction'] == prediction
            assert record['tokens'] == len(truth) and record['position_correct'] == correct
            assert record['multiset_correct'] == overlap
            assert abs(record['position_accuracy']-correct/len(truth)) < 1e-12
            assert abs(record['multiset_recall']-recall) < 1e-12
            assert abs(record['multiset_precision']-precision) < 1e-12
            assert record['exact_match'] == (prediction == truth)
            assert record['supplied_multiset_conserved'] == (Counter(prediction) == offered)
            assert record['raw_source'] == str(raw_path.relative_to(results))
            cosine = record['embedding_cosine']
            assert math.isfinite(cosine) and -1.000001 <= cosine <= 1.000001
            if not text:
                assert cosine == 0.0
            values.append(record)
        summaries[model] = dict(
            status='completed', prompts=602,
            macro_position_percent=100*sum(r['position_accuracy'] for r in values)/602,
            position_correct=sum(r['position_correct'] for r in values),
            tokens=sum(r['tokens'] for r in values),
            macro_multiset_recall_percent=100*sum(r['multiset_recall'] for r in values)/602,
            macro_multiset_precision_percent=100*sum(r['multiset_precision'] for r in values)/602,
            mean_embedding_cosine=sum(r['embedding_cosine'] for r in values)/602,
            exact_spans=sum(r['exact_match'] for r in values),
            conserved_supplied_multisets=sum(r['supplied_multiset_conserved'] for r in values),
            encoder='sentence-transformers/all-MiniLM-L6-v2', encoder_max_tokens=256,
            settings=list(settings.values()))
        audit[model] = dict(
            complete_ids=602, raw_predictions_and_counters_verified=True,
            empty_outputs_retained=sum(not r['predicted_text'].strip() for r in raw),
            budget_limited_outputs_retained=sum(bool(r['budget_exhausted']) for r in values),
            raw_and_score_bytes_unchanged=(raw_path.read_bytes() == raw_before and
                                           scores_path.read_bytes() == scores_before))
        assert audit[model]['raw_and_score_bytes_unchanged']
    summary = dict(complete=True, expected_prompts=602, expected_model_outputs=1204,
                   cohort=protocol['cohort'], models=summaries, inputs=protocol['inputs'],
                   scoring='All 602 complete responses per GPT model count, including incorrect, budget-limited and empty outputs. No response filtering or new requests.')
    report = dict(passed=True, models=audit, expected_model_outputs=1204,
                  all_model_prompts_match=True, new_model_or_API_requests=False,
                  embeddings='Existing stored per-prompt cosine values checked for bounds, empty-output handling and aggregation; no new encoder inference.')
    return summary, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--write-summary', action='store_true')
    args = parser.parse_args()
    summary, report = verify(args.results, args.manifest)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    if args.write_summary:
        temporary = args.results/'api_summary.new.json'
        temporary.write_text(json.dumps(summary, indent=2)+'\n')
        temporary.replace(args.results/'api_summary.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
