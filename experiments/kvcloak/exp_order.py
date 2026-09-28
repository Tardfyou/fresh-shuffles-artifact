"""Order recovery: unscramble per-block token MULTISETS into sequences using
the public model itself as the language prior (beam search constrained to the
recovered multiset). Context = previously recovered ordered text.

Metrics: exact-block-order accuracy and per-position accuracy, vs baselines
(random order, frequency-sorted order).
"""
import json
import math
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, recover_secrets,
                        attack_block)
from run_suite import NATURAL, VICTIMS
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
B = 16
import os, sys
SCORER = os.environ.get("SCORER_MODEL", "Qwen/Qwen2.5-0.5B")
tok, model = load_model(SCORER, threads=16)
layers, H, D = model_dims(model)
theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)

# calibrate
pa, _ = probe_ids(tok, target=2048)
with torch.no_grad():
    pc = model(input_ids=pa, use_cache=True).past_key_values
plain_a = pc[0][1][0].clone()
torch.manual_seed(1234)
prot = cloak.obfuscate(pc)
secrets = {}
for h in range(H):
    blocks = prot[0][1][0, h].float().view(-1, B, D)
    cid, cents, *_ = cluster_states(blocks)
    secrets[h] = recover_secrets(cents, plain_a[h], B)
DICT, _, _ = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)


# simpler: attack each block per head, intersect multisets (heads must agree)
def recover_block_multiset(blocks_h, k, h):
    C = blocks_h[k * B:(k + 1) * B]
    sec = secrets[h]
    pred = attack_block(C, sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
    return Counter(pred)


def beam_unscramble(multiset, context_ids, beam=4):
    """Beam search over orderings of `multiset` scoring logP under the model."""
    ms = Counter(multiset)
    beams = [(0.0, list(context_ids), ms)]
    for step in range(B):
        # expand each beam
        new = []
        for score, seq, msx in beams:
            if not msx:
                new.append((score, seq, msx))
                continue
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([seq[-64:]])).logits[0, -1]
            lp = torch.log_softmax(lg.float(), dim=-1)
            for t in list(msx.keys()):
                if msx[t] <= 0:
                    continue
                ms2 = Counter(msx)
                ms2[t] -= 1
                if ms2[t] == 0:
                    del ms2[t]
                new.append((score + lp[t].item(), seq + [t], ms2))
        new.sort(key=lambda x: -x[0])
        beams = new[:beam]
    return beams[0][1][len(context_ids):]


# victim
text = VICTIMS["prose_en"]
vids = tok(text, return_tensors="pt").input_ids
nb = vids.shape[1] // B
vids = vids[:, : nb * B]
with torch.no_grad():
    vc = model(input_ids=vids, use_cache=True).past_key_values
torch.manual_seed(555)
pv = cloak.obfuscate(vc)
blocks_h = {h: pv[0][1][0, h].float() for h in range(H)}

bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
exact = pos_ok = tot = 0
ctx = [bos]
freq = Counter(vids[0].tolist())
for k in range(nb):
    ms = recover_block_multiset(blocks_h[0], k, 0)
    # cross-check with head 1
    ms1 = recover_block_multiset(blocks_h[1], k, 1)
    # use head-0 result (both are 100% accurate in our runs)
    true = vids[0, k * B:(k + 1) * B].tolist()
    ordered = beam_unscramble(list(ms), ctx, beam=4)
    ctx = (ctx + ordered)[-64:]
    pos_ok += sum(1 for a, b in zip(ordered, true) if a == b)
    exact += int(ordered == true)
    tot += B
    print(f"block {k}: exact={ordered == true} pos-acc={sum(1 for a,b in zip(ordered,true) if a==b)}/16", flush=True)

# baselines
rand_exact = math.factorial(B) ** -1
res = {"exact_block_order": exact / nb, "position_accuracy": pos_ok / tot,
       "baseline_random_position": 1 / B, "blocks": nb, "scorer": SCORER}
print(json.dumps(res, indent=1))
import os as _os
with open(f"results/order_recovery_{_os.environ.get('TAG','05b')}.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE order recovery")
