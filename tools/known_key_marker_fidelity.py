"""Known-key arithmetic diagnostics on public, already saved token arrays.

No parameter recovery, marker search, or open-vocabulary decoding is performed.
All candidate token IDs are fixed by previously saved public observations.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def run(project, cache, output):
    assert (output/'PROTOCOL.json').exists()
    assert not (output/'checks.jsonl').exists()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(project/'experiments/kvcloak'))
    from get_kvcloak_config import get_kvcloak_config
    from kvcloak import KVCloak
    from run_suite import NATURAL

    source = project/'github-public-artifact/results/new_evidence_20261102'
    main = project/'github-public-artifact/results/main_public'
    environments = {}
    start = time.monotonic()
    with (output/'checks.jsonl').open('w') as stream:
        for tag, name in [('qwen25_05b', 'Qwen/Qwen2.5-0.5B'),
                          ('llama32_1b', 'unsloth/Llama-3.2-1B')]:
            snapshots = list((cache/('models--'+name.replace('/', '--'))/'snapshots').iterdir())
            assert len(snapshots) == 1
            snapshot = snapshots[0]
            config = read(snapshot/'config.json')
            H = config['num_key_value_heads']
            D = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
            protocol = read(output/'PROTOCOL.json')
            if 'fixtures_file' in protocol:
                rows = [r for r in map(json.loads, (output/protocol['fixtures_file']).open()) if r['tag'] == tag]
                assert len(rows) == protocol['expected_by_tag'][tag]
            else:
                rows = [json.loads(line) for line in (source/f'e3b_flip_{tag}_v2.jsonl').open()]
                assert len(rows) == {'qwen25_05b': 66, 'llama32_1b': 240}[tag]
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            natural = [tokenizer(text, return_tensors='pt').input_ids[0] for text in NATURAL]
            ids = sorted({t for r in rows for t in r['true']+r['variants']['base']['pred']}
                         | {t for sentence in natural for t in sentence.tolist()})
            index = {token: i for i, token in enumerate(ids)}
            with safe_open(snapshot/'model.safetensors', framework='pt') as handle:
                embedding = handle.get_tensor('model.embed_tokens.weight')[ids].float()
                norm = handle.get_tensor('model.layers.0.input_layernorm.weight').float()
                weight = handle.get_tensor('model.layers.0.self_attn.v_proj.weight').float()
                bias_key = 'model.layers.0.self_attn.v_proj.bias'
                bias = handle.get_tensor(bias_key).float() if bias_key in handle.keys() else None
            def value(x):
                x = x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+config['rms_norm_eps'])
                return torch.nn.functional.linear(x*norm, weight, bias).reshape(-1, H, D)
            vectors = value(embedding)
            vmax = torch.zeros(H)
            for sentence in natural:
                vv = value(embedding[[index[t] for t in sentence.tolist()]])
                vmax = torch.maximum(vmax, vv.abs().amax(dim=(0, 2)))
            theta = {'layer_0': {'value_max_values': vmax.tolist(), 'key_max_values': [0.]*H}}
            torch.manual_seed(42)
            cfg = get_kvcloak_config(1, H, D, 16, theta, 1., 1., 2.)
            cloak = KVCloak(cfg, torch.bfloat16, False, False, True)
            torch.manual_seed(1234)
            cloak._prepare_device_tensors(torch.device('cpu'))
            params = cloak._device_cache[torch.device('cpu')]['obf'][0]
            S = params['S_v_batch'].reshape(H, 16, 16)
            M = params['M_v_batch'].reshape(H, D, D)
            A = params['A_v_batch'].reshape(H, 16, D)
            marker_rows = [int((a.abs().sum(-1)>0).nonzero().flatten().item()) for a in A]
            historical = {r['sid']: r for r in map(json.loads, (main/tag/'per_sample.jsonl').open())}
            permutations = {}
            environments[tag] = dict(checkpoint=name, revision=snapshot.name, heads=H, dimension=D,
                                     marker_rows=marker_rows, threshold_value_max=vmax.tolist(),
                                     marker_norms=A.double().norm(dim=-1).amax(-1).tolist(),
                                     projection='FP32 first-layer input RMSNorm and public Value projection from selected checkpoint tensors; no full model or calibrator run.',
                                     implementation='Official KVCloak prepares BF16 S/M/A; all diagnostic inverse matrices use those stored values in FP64.')
            for ordinal, r in enumerate(rows):
                h = r['h']
                key = (r['sid'], r['seed'])
                if key not in permutations:
                    generator = torch.Generator().manual_seed(r['seed'])
                    nb = historical[r['sid']]['blocks']
                    permutations[key] = torch.argsort(torch.rand(nb, 16, generator=generator), dim=1)
                P = torch.eye(16, dtype=torch.bfloat16)[permutations[key][r['k']]]
                Sp = S[h]@P
                invS = torch.linalg.inv(Sp.double())
                invM = torch.linalg.inv(M[h].double())
                V = vectors[[index[t] for t in r['true']], h]
                Vm = V.to(torch.bfloat16)@M[h]
                marker = A[h]
                bf16_pre = Vm+marker
                bf16_cipher = Sp@bf16_pre
                if protocol.get('validate_official_forward'):
                    from transformers.cache_utils import DynamicCache
                    nb = historical[r['sid']]['blocks']
                    cache_v = torch.zeros(1, H, nb*16, D, dtype=torch.float32)
                    cache_v[0, h, r['k']*16:(r['k']+1)*16] = V
                    cache_k = torch.zeros_like(cache_v)
                    known_cache = DynamicCache.from_legacy_cache(((cache_k, cache_v),))
                    torch.manual_seed(r['seed'])
                    native = cloak.obfuscate(known_cache)[0][1][0, h, r['k']*16:(r['k']+1)*16]
                    assert torch.equal(native, bf16_cipher.float()), (tag, ordinal, 'official forward differs')
                no_marker_cipher = Sp@Vm
                fp32_cipher = Sp.float()@(V@M[h].float()+marker.float())
                fp64_cipher = Sp.double()@(V.double()@M[h].double()+marker.double())
                outcomes = {
                    'fp64_sanity': (invS@fp64_cipher-marker.double())@invM,
                    'fp32_forward': (invS@fp32_cipher.double()-marker.double())@invM,
                    'bf16_known_key': (invS@bf16_cipher.double()-marker.double())@invM,
                    'bf16_no_marker': (invS@no_marker_cipher.double())@invM,
                    'bf16_marker_before_row_mixing': (bf16_pre.double()-marker.double())@invM,
                    'bf16_no_marker_before_row_mixing': Vm.double()@invM,
                }
                candidate_ids = sorted(set(r['true']+r['variants']['base']['pred']))
                candidates = vectors[[index[t] for t in candidate_ids], h].double()
                normalized_candidates = torch.nn.functional.normalize(candidates, dim=-1)
                true_idx = torch.tensor([candidate_ids.index(t) for t in r['true']])
                surplus = list((Counter(r['variants']['base']['pred'])-Counter(r['true'])).elements())
                record = dict(tag=tag, ordinal=ordinal, sid=r['sid'], seed=r['seed'], head=h, block=r['k'],
                              true=r['true'], prior_correct=r['variants']['base']['correct'],
                              reconstructed_true_marker_row=marker_rows[h], candidate_ids=candidate_ids,
                              prior_surplus_token_ids=surplus, true_vector_norms=V.double().norm(dim=-1).tolist(),
                              conditions={})
                mr = marker_rows[h]
                record['marker_addition_trace'] = dict(
                    original_value=V[mr].tolist(),
                    bf16_value_after_M=Vm[mr].float().tolist(),
                    stored_marker=marker[mr].float().tolist(),
                    bf16_sum=bf16_pre[mr].float().tolist(),
                    nonzero_value_coordinates=int((Vm[mr] != 0).sum()),
                    swallowed_nonzero_coordinates=int(((Vm[mr] != 0) & (bf16_pre[mr] == marker[mr])).sum()),
                    marker_to_value_norm_ratio=float(marker[mr].double().norm()/V[mr].double().norm()),
                )
                if protocol.get('validate_official_forward'):
                    record['official_forward_bit_equal'] = True
                for condition, reconstructed in outcomes.items():
                    scores = torch.nn.functional.normalize(reconstructed, dim=-1)@normalized_candidates.T
                    true_score = scores[torch.arange(16), true_idx]
                    competing = scores.clone()
                    competing[torch.arange(16), true_idx] = -torch.inf
                    alternative_score = competing.max(-1).values
                    chosen = scores.argmax(-1)
                    predicted = [candidate_ids[j] for j in chosen.tolist()]
                    delta = reconstructed-V.double()
                    record['conditions'][condition] = dict(
                        relative_vector_error=(delta.norm(dim=-1)/V.double().norm(dim=-1).clamp_min(1e-30)).tolist(),
                        true_score=true_score.tolist(), best_other_score=alternative_score.tolist(),
                        margin=(true_score-alternative_score).tolist(), fixed_candidate_predictions=predicted,
                        correct_rows=sum(t == p for t, p in zip(r['true'], predicted)),
                        marker_row_correct=predicted[marker_rows[h]] == r['true'][marker_rows[h]])
                assert record['conditions']['fp64_sanity']['correct_rows'] == 16
                assert max(record['conditions']['fp64_sanity']['relative_vector_error']) < 1e-10
                stream.write(json.dumps(record, separators=(',', ':'))+'\n')
                stream.flush()
            print(tag, 'completed', len(rows), 'known-key blocks', flush=True)
            del embedding, vectors, weight
    (output/'environment.json').write_text(json.dumps(dict(torch=torch.__version__, threads=8,
                                                          models=environments, elapsed_s=time.monotonic()-start), indent=2)+'\n')
    print('known-key arithmetic diagnostics complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.project, a.cache, a.out)
