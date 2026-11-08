"""Align saved public row labels to deterministic known protection layouts.

No model inference, parameter recovery, or attack execution. Correct-block
records alone determine row-label alignment; all saved failed blocks are checked.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoTokenizer


def audit(root, cache, output):
    results = root/'results'
    manifest = {r['sid']: r for r in map(json.loads, (results/'main_public/samples_main_public.jsonl').open())}
    report = {}
    for tag, name, H in [('qwen25_05b', 'Qwen/Qwen2.5-0.5B', 2),
                          ('llama32_1b', 'unsloth/Llama-3.2-1B', 8)]:
        snapshot = next((cache/('models--'+name.replace('/', '--'))/'snapshots').iterdir())
        tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        main = {r['sid']: r for r in map(json.loads, (results/'main_public'/tag/'per_sample.jsonl').open())}
        tokens = {sid: tok(r['text']).input_ids for sid, r in manifest.items()}
        assert set(tokens) == set(main) and len(tokens) == 3000
        assert all(len(ids) == main[sid]['ntok'] for sid, ids in tokens.items())
        selected = [json.loads(l) for l in (results/'new_evidence_20261102'/f'e3b_flip_{tag}_v2.jsonl').open()]
        permutations = {}
        def perm(sid, seed, block):
            key = (sid, seed)
            if key not in permutations:
                generator = torch.Generator().manual_seed(seed)
                permutations[key] = torch.argsort(torch.rand(main[sid]['blocks'], 16, generator=generator), dim=1).tolist()
            return permutations[key][block]
        scores = [np.zeros((16, 16), dtype=np.int64) for _ in range(H)]
        support = Counter()
        for r in selected:
            if r['orig_correct'] != 16:
                continue
            support[r['h']] += 1
            p = perm(r['sid'], r['seed'], r['k'])
            for i, predicted in enumerate(r['variants']['base']['pred']):
                for j, source_position in enumerate(p):
                    scores[r['h']][i, j] += predicted == r['true'][source_position]
        mappings = []
        minimum_assignment_gaps = []
        for matrix in scores:
            ii, jj = linear_sum_assignment(-matrix)
            best = int(matrix[ii, jj].sum())
            gaps = []
            for i, j in zip(ii, jj):
                prohibited = matrix.copy()
                prohibited[i, j] = -1000000
                ai, aj = linear_sum_assignment(-prohibited)
                gaps.append(best-int(prohibited[ai, aj].sum()))
            mappings.append(jj.tolist())
            minimum_assignment_gaps.append(min(gaps))
        g = torch.Generator().manual_seed(1234)
        marker = []
        for _ in range(H):
            torch.randint(0, 16, (1,), generator=g)  # Key marker consumes a draw.
            marker.append(torch.randint(0, 16, (1,), generator=g).item())
        counts = Counter()
        by_head = defaultdict(Counter)
        missing_types = Counter()
        source = results/'saved_record_review_20261101'/tag/'failed_block_diagnostics.jsonl.gz'
        seen = set()
        with gzip.open(source, 'rt') as stream:
            for saved in map(json.loads, stream):
                b = saved['block']
                sid = saved['sid']
                seed = int(str(saved['seed']).replace('seed', ''))
                key = (sid, seed, b['h'], b['k'])
                assert key not in seen
                seen.add(key)
                assert minimum_assignment_gaps[b['h']] > 0, 'Row mapping is not unique for a failing head'
                original_position = perm(sid, seed, b['k'])[mappings[b['h']][b['j']]]
                predicted = [r['p'] for r in b['rows']]
                truth = tokens[sid][16*b['k']:16*(b['k']+1)]
                missing = Counter(truth)-Counter(predicted)
                n = sum(missing.values())
                assert n == 16-b['correct']
                right = original_position == marker[b['h']]
                counts['failed_blocks'] += 1
                counts['misses'] += n
                counts['correct_marker_hypothesis_blocks'] += right
                counts['wrong_marker_hypothesis_blocks'] += not right
                counts['marked_token_type_missing_blocks'] += truth[marker[b['h']]] in missing
                by_head[str(b['h'])]['failed_blocks'] += 1
                by_head[str(b['h'])]['wrong_marker_hypothesis_blocks'] += not right
                missing_types.update(missing)
        assert counts['misses'] == sum(c['total']-c['correct'] for r in main.values() for c in r['seeds'].values())
        report[tag] = dict(counts=counts, by_head=dict(by_head), marker_rows=marker,
                           row_mappings=mappings, mapping_correct_block_support=dict(support),
                           assignment_uniqueness_minimum_gaps=minimum_assignment_gaps,
                           alignment_count_matrices=[m.tolist() for m in scores],
                           all_3000_tokenized_lengths_match=True, checkpoint_revision=snapshot.name,
                           most_common_missing_types=[dict(token=t, piece=tok.decode([t]), misses=c) for t,c in missing_types.most_common(12)])
        print(tag, dict(counts), 'assignment gaps', minimum_assignment_gaps, flush=True)
    result = dict(results=report,
                  provenance='New deterministic reconstruction from frozen implementation and CPU seeds, aligned using saved correct blocks; not historical per-block marker logs.',
                  new_model_inference_or_parameter_recovery=False, torch=torch.__version__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.cache, a.out)
