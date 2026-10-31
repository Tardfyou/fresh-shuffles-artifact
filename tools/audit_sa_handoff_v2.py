"""Independently check known-key cache measurements, inputs and controls offline."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys


def records(path):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith('#run '):
            continue
        result.append(json.loads(line))
    return result


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify(root, manifest, subsets, regenerate=True):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    torch.set_num_threads(4)
    source = root/'source/exp_sa_rotation_verify_v2.py'
    spec = importlib.util.spec_from_file_location('_known_key_measurement', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Definitions only; never call its launch/model functions.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/kvcloak'))
    inputs = records(root/'inputs.jsonl')
    originals = {r['sid']: r for r in records(manifest)}
    expected_ids = json.loads(subsets.read_text())['lifecycle']
    assert len(inputs) == len({r['sid'] for r in inputs}) == 100
    assert [r['sid'] for r in inputs] == expected_ids
    assert all(r['text'] == originals[r['sid']]['text'] for r in inputs)
    proto = json.loads((root/'PROTOCOL.json').read_text())
    models = {'Qwen2.5-0.5B': 'Qwen/Qwen2.5-0.5B', 'Llama-3.2-1B': 'unsloth/Llama-3.2-1B'}
    dtypes = ('float32', 'bfloat16')
    conditions = ('original', 'S_a_refresh')
    masks = ('no_pad', 'left8', 'right8', 'left16', 'right16')
    segments = [('seg1', 2, 0), ('seg2', 3, 2), ('seg3', 1, 5)]
    rows = [r for f in sorted(root.glob('workers/*/checks.jsonl')) for r in records(f)]
    requests = [r for f in sorted(root.glob('workers/*/requests.jsonl')) for r in records(f)]
    expected = {'|'.join(x) for x in itertools.product(expected_ids, models, dtypes, conditions, masks, [s[0] for s in segments])}
    keys = [r['key'] for r in rows]
    assert len(keys) == len(set(keys)) == len(expected) == 12000 and set(keys) == expected
    by_request = {r['key']: r for r in requests}
    assert len(by_request) == len(requests) == 4000
    assert {r['key'].rsplit('|', 1)[0] for r in rows} == set(by_request)
    assert all(r['status'] == 'measured' and r['attempt'] == 1 for r in rows + requests)
    launch = json.loads((root/'launcher_run.json').read_text())
    assert len(launch['exit_codes']) == 32 and all(v == 0 for v in launch['exit_codes'].values())
    index = json.loads((root/'fixtures/index.json').read_text())['entries']
    assert len(index) == len({(r['sid'], r['model'], r['mask']) for r in index}) == 1000
    fixtures = {}
    mixed = {}
    configs = {}
    input_by_id = {r['sid']: r for r in inputs}
    for short, full in models.items():
        tokenizer = AutoTokenizer.from_pretrained(full, local_files_only=True)
        config = AutoConfig.from_pretrained(full, local_files_only=True)
        prefix = tokenizer(module.PREFIX_TEXT).input_ids[:16]
        assert len(prefix) == 16
        dims = (config.num_hidden_layers, config.num_key_value_heads,
                getattr(config, 'head_dim', None) or config.hidden_size//config.num_attention_heads)
        theta = json.loads((root/'fixtures'/f'theta_{short}.json').read_text())
        base = module.build_base_config(*dims, theta) if regenerate else None
        base_fingerprint = module.config_fingerprint(base) if regenerate else None
        for sid in expected_ids:
            body = tokenizer(input_by_id[sid]['text']).input_ids[:134]
            cont = body[-6:]
            real_history = prefix + body[:-6]
            if regenerate:
                refresh, m_fixed, changed = module.build_refresh_config(base, *dims, theta, module.request_seed_for(sid, short))
                assert m_fixed['all'] and changed['all']
                configs[(short, sid)] = (base_fingerprint, module.config_fingerprint(refresh))
            for name in masks:
                pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                n = 0 if name == 'no_pad' else int(name.replace('left', '').replace('right', ''))
                if name.startswith('left'):
                    ids = prefix + [pad]*n + body[:-6]
                    mask = [1]*16 + [0]*n + [1]*(len(body)-6)
                    pos = [sum(mask[:i+1])-1 for i in range(len(mask))]
                else:
                    ids = real_history + [pad]*n
                    mask = [1]*len(real_history) + [0]*n
                    pos = list(range(len(ids)))
                item = next(e for e in index if (e['sid'], e['model'], e['mask']) == (sid, short, name))
                fix = json.loads((root/'fixtures'/item['file']).read_text())
                assert fix['eligible'] and fix['prefix_ids'] == prefix
                assert (fix['ids'], fix['mask'], fix['pos'], fix['cont']) == (ids, mask, pos, cont)
                identity = digest(dict(ids=ids, mask=mask, pos=pos, cont=cont))
                assert item['input_sha256'] == fix['input_sha256'] == identity
                assert fix['hist_len'] == len(ids) and fix['cont_len'] == len(cont) == 6
                fixtures[(sid, short, name)] = fix
                mixed[(sid, short, name)] = any(len(set(mask[i:i+16])) > 1 for i in range(0, len(mask), 16))
        for req in [r for r in requests if r['model'] == short]:
            fix = fixtures[(req['sid'], short, req['mask'])]
            assert req['input_sha256'] == fix['input_sha256']
            assert req['hist_len'] == fix['hist_len'] and req['cont_len'] == 6
            assert req['prep_seed'] == module.PREP_SEED and req['shuffle_seed'] == module.SHUFFLE_SEED
            assert req['m_fixed']['all'] and req['s_a_changed']['all']
            if regenerate:
                base_fp, refresh_fp = configs[(short, req['sid'])]
                assert req['base_config_fingerprint'] == base_fp
                assert req['config_fingerprint'] == (base_fp if req['condition'] == 'original' else refresh_fp)
            mutation = req['mutation']
            assert mutation['n_entries'] == len(mutation['detail_op']) == len(mutation['detail_plain']) == 2*dims[0]
            assert mutation['plain_baseline_unpolluted'] and all(e['bytes_equal'] for e in mutation['detail_plain'])
            assert mutation['operator_input_mutated_n'] == sum(not e['bytes_equal'] for e in mutation['detail_op'])
            # Recompute the actual draw schedule without invoking the cache operator.
            blocks = math.ceil(fix['hist_len']/16)
            torch.manual_seed(module.SHUFFLE_SEED)
            state = hashlib.sha256()
            for _ in range(dims[0]):
                state.update(torch.argsort(torch.rand(blocks, 16), dim=1).to(torch.int64).numpy().tobytes())
            assert req['perm_hash'] == state.hexdigest()
            assert req['perm_calls'] == dims[0] and req['perm_shapes'] == [[blocks, 16]]*dims[0]
    by_pair = defaultdict(dict)
    cells = defaultdict(list)
    for row in rows:
        fix = fixtures[(row['sid'], row['model'], row['mask'])]
        seg_name, n, offset = next(s for s in segments if s[0] == row['seg'])
        assert row['seg_len_declared'] == row['seg_len_actual'] == n
        assert row['seg_tokens'] == fix['cont'][offset:offset+n]
        assert row['seg_abs_range'] == [fix['hist_len']+offset, fix['hist_len']+offset+n]
        assert row['cursor_before'] == fix['hist_len']+offset and row['cont_start'] == fix['hist_len']
        assert row['input_sha256'] == fix['input_sha256'] and row['sanity_max'] == 0.0
        assert all(math.isfinite(row[k]) for k in ('prot_vs_plain_max', 'prot_vs_plain_mean', 'plain_vs_full_max', 'prot_vs_full_max'))
        passed = (row['prot_vs_plain_max'] <= module.THRESHOLDS[row['dtype']]
                  and row['prot_vs_plain_argmax_all'] and row['shape_match'] and row['finite'])
        assert row['functional_pass'] == passed
        by_pair[(row['sid'], row['model'], row['dtype'], row['mask'], seg_name)][row['condition']] = row
        cells[(row['model'], row['dtype'], row['condition'], row['mask'])].append(row)
    summary = json.loads((root/'summary.json').read_text())['cells']
    assert len(cells) == 40
    for key, rr in cells.items():
        saved = summary['|'.join(key)]
        assert saved['n'] == len(rr) == 300 and saved['pass'] == sum(r['functional_pass'] for r in rr)
        assert saved['max'] == max(r['prot_vs_plain_max'] for r in rr)
    request_pairs = defaultdict(dict)
    for req in requests:
        request_pairs[(req['sid'], req['model'], req['dtype'], req['mask'])][req['condition']] = req
    assert len(request_pairs) == 2000 and all(p['original']['perm_hash'] == p['S_a_refresh']['perm_hash'] for p in request_pairs.values())
    discordant = [p for p in by_pair.values() if p['original']['functional_pass'] != p['S_a_refresh']['functional_pass']]
    alignment = {}
    for category in (False, True):
        rr = [r for r in rows if r['dtype'] == 'float32' and mixed[(r['sid'], r['model'], r['mask'])] == category]
        alignment['mixed' if category else 'unmixed'] = dict(records=len(rr), passed=sum(r['functional_pass'] for r in rr),
            min_error=min(r['prot_vs_plain_max'] for r in rr), max_error=max(r['prot_vs_plain_max'] for r in rr))
    return dict(record_integrity_pass=True, measurement_protocol_accepted=True, accepted_for_paper=True,
                expected_records=12000, measured=12000, error=0, not_run=0, source_prompts=100,
                fixture_arrays_independently_reconstructed=1000, full_configurations_regenerated=200 if regenerate else 0,
                independent_request_pairs=2000, paired_shuffles_verified=2000, segment_pairs=6000,
                functional_pass=sum(r['functional_pass'] for r in rows), functional_fail=sum(not r['functional_pass'] for r in rows),
                pass_by_condition={c: sum(r['functional_pass'] for r in rows if r['condition'] == c) for c in conditions},
                discordant_pass_pairs=len(discordant),
                original_pass_refresh_fail=sum(p['original']['functional_pass'] and not p['S_a_refresh']['functional_pass'] for p in by_pair.values()),
                original_fail_refresh_pass=sum(not p['original']['functional_pass'] and p['S_a_refresh']['functional_pass'] for p in by_pair.values()),
                fp32_alignment=alignment,
                max_plain_full_fp32=max(r['plain_vs_full_max'] for r in rows if r['dtype'] == 'float32'),
                fp32_plain_full_argmax_disagreements=sum(not r['plain_vs_full_argmax_all'] for r in rows if r['dtype'] == 'float32'),
                source_review='Matched full/plain-cached/protected paths, supplied masks/positions, sequential branches, fixed M and recorded configuration/permutation controls inspected in measured source.',
                reporting_corrections=['4,000 directed permutation comparisons represent 2,000 independent request pairs.',
                    'Original and refresh cells are not completely symmetric; report the paired disagreements.',
                    'The saved FP32 full/cached maximum differs from the handoff summary; use the recomputed value.',
                    'Mixed-mask association and native-bf16 failures are observations, not standalone causal proofs or unconditional rotation-equivalence claims.'],
                scope='One protection/recovery of prefill followed by three consumer segments; no repeated encryption, serving scheduler, device transfer or hardware-attestation validation.',
                new_model_or_API_requests=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--subsets', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--no-config-regeneration', action='store_true')
    a = p.parse_args()
    result = verify(a.input, a.manifest, a.subsets, not a.no_config_regeneration)
    a.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
