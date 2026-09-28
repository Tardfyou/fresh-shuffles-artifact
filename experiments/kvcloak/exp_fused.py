"""Fused-mode attack with STRICT permission isolation (review §3.1):
attacker NEVER touches the fused model. All attacker assets come from:
  - the PUBLIC original model (dictionary, probe plaintexts)
  - the recovered secrets (M̂ from own-probe (v, w) pairs)
Fused deployment applies V' = V M per head (fusion_llama). The attacker's
public-model dictionary D is composed with recovered M̂ per head: D̂ = D M̂.
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
from fusion import fusion_llama

torch.manual_seed(42)
torch.set_num_threads(16)
B = 16
res = {"mode": "fused, attacker isolated (public model only)"}

# ---- defender side: fuse a private copy ----
tok, defender = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(defender)
pids, prefix_len, pchar = probe_ids(tok, target=2600,
                                    max_positions=getattr(defender.config, "max_position_embeddings", None))
theta = build_theta(defender, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
fusion_llama(defender, cfg)                      # deployment step: M folded into weights
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=True, need_ratio=False, add_a=True)
NB = pids.shape[1] // B
nblk_pad = None
with torch.no_grad():
    pc = defender(input_ids=pids, use_cache=True).past_key_values
fused_probe_v0 = pc[0][1][0].clone()             # defender-side artifact; used ONLY to derive what the attacker can also compute
torch.manual_seed(1234)
prot = cloak.obfuscate(pc)
nblk_pad = prot[0][1].shape[2] // B
first_blk = (prefix_len + B - 1) // B

# ---- attacker side: fresh process-equivalent (public model, no fusion) ----
del defender, pc
import gc
gc.collect()
tok2, public = load_model("Qwen/Qwen2.5-0.5B", threads=16)
with torch.no_grad():
    pub_pc = public(input_ids=pids, use_cache=True).past_key_values
plain_probe = pub_pc[0][1][0].clone()            # attacker computes own plaintexts from PUBLIC model
assert not torch.allclose(plain_probe[0], fused_probe_v0[0]), "fusion should change V (sanity)"
DICT, _, _ = build_dictionary(public, H, D)      # public-model dictionary only

secrets = {}
for h in range(H):
    blocks = prot[0][1][0, h].float().view(nblk_pad, B, D)[first_blk:NB]
    cid, cents, *_ = cluster_states(blocks)
    if cid != B:
        continue
    sec = recover_secrets(cents, plain_probe[h], B)
    # w here = M^T v computed by recover_secrets with the PUBLIC v.
    # In fused mode the ciphertext data component is u w^T with w = M v (V' = V M),
    # so recompute w from the fused-consistent relation: attacker uses own probe
    # block states directly: w_est = row mean of (C - marker) projected; simpler:
    # w = M^T v  <=>  fused w' = M^T (M v)? No -- fused V' rows = v M, so
    # C data part = u (v M)^T i.e. w_fused = M^T v (same!). Check magnitude:
    w_pub = sec["w"]
    # sanity: ||w_pub|| should equal ||v|| (M orthogonal); verify against public v
    v_pub = plain_probe[h][0]
    if abs(w_pub.norm().item() - v_pub.norm().item()) / v_pub.norm().item() > 0.05:
        # fused convention differs; rebuild w from projection of states
        P = torch.eye(D) - torch.outer(sec["a_hat"], sec["a_hat"]) / (sec["a_hat"] ** 2).sum()
        rights = []
        for c in cents:
            U_, S_, Vh_ = torch.linalg.svd(c @ P)
            rights.append(Vh_[0] * S_[0])
        w_pub = torch.stack(rights).mean(0)
        sec["w"] = w_pub
    secrets[h] = sec
    res[f"head{h}_states"] = cid

# build attacker dictionary composed with recovered M̂:  D̂ = D M̂ (per head)
# M̂ recovered from (v_public, w) pair per head via strided angles
def rot_from_pair(v, w):
    D2 = v.numel() // 2
    ang = torch.zeros(D2)
    for i in range(D2):
        va, vb = v[i].item(), v[i + D2].item()
        wa, wb = w[i].item(), w[i + D2].item()
        if va * va + vb * vb < 1e-8:
            continue
        ang[i] = math.atan2(wb, wa) - math.atan2(vb, va)
    from attack_lib import rot_strided
    return rot_strided(ang, v.numel())

# With column-vector notation, w = M.T @ v for cache rows transformed by V M.
# rot_from_pair follows attack_lib.rot_strided's rotation convention.
# The fixed-marker diagnostic below does not select or validate angle signs.
accs = {}
for h in range(H):
    if h not in secrets:
        continue
    sec = secrets[h]
    M_hat = rot_from_pair(plain_probe[h][0], sec["w"])
    Dhat = DICT[:, h] @ M_hat
    Dn_hat = torch.nn.functional.normalize(Dhat, dim=-1)
    # Legacy diagnostic: subtract marker column 0 from every sampled state.
    # This is not a full probe-decoding validation; most states move the marker.
    blocks = prot[0][1][0, h].float().view(nblk_pad, B, D)[first_blk:NB]
    cid, cents, *_ = cluster_states(blocks)
    ok = 0
    for j in range(min(8, len(cents))):
        C = cents[j]
        Xn = torch.nn.functional.normalize(
            sec["s"] @ (C - torch.outer(sec["s"][0], sec["a_hat"])), dim=-1)
        sims = Xn @ Dn_hat.T
        pred = sims.argmax(1)
        # Count single-token consistency under this fixed marker hypothesis.
        ok += int(len(set(pred.tolist())) == 1)
    res[f"head{h}_probe_selfcheck"] = f"{ok}/8"
    accs[h] = (sec, Dn_hat)
    print(f"head {h}: probe self-check (single-token consistency) {ok}/8", flush=True)

# victims (defender-fused side regenerated deterministically)
torch.manual_seed(0)
tok3, defender2 = load_model("Qwen/Qwen2.5-0.5B", threads=16)
theta2 = build_theta(defender2, tok3, NATURAL)
torch.manual_seed(42)
cfg2 = get_kvcloak_config(layers, H, D, B, theta2, 1.0, 1.0, 2.0)
fusion_llama(defender2, cfg2)
cloak2 = KVCloak(cfg2, dtype=torch.bfloat16, fused=True, need_ratio=False, add_a=True)
for name in ("prose_en", "chinese", "python_code"):
    text = VICTIMS[name]
    vids = tok3(text, return_tensors="pt").input_ids
    nb = vids.shape[1] // B
    vids = vids[:, : nb * B]
    with torch.no_grad():
        vc = defender2(input_ids=vids, use_cache=True).past_key_values
    torch.manual_seed(7300)
    pv = cloak2.obfuscate(vc)
    cor = tot = 0
    for h in range(H):
        if h not in accs:
            continue
        sec, Dn_hat = accs[h]
        bp = pv[0][1][0, h].float()
        for k in range(nb):
            C = bp[k * B:(k + 1) * B]
            best = None
            for j in range(B):
                X = torch.nn.functional.normalize(
                    sec["s"] @ (C - torch.outer(sec["s"][j], sec["a_hat"])), dim=-1)
                sims = X @ Dn_hat.T
                sc = sims.max(1).values.sum().item()
                if best is None or sc > best[0]:
                    best = (sc, sims.argmax(1).tolist())
            cor += sum((Counter(best[1]) & Counter(vids[0, k * B:(k + 1) * B].tolist())).values())
            tot += B
    res[f"victim_{name}"] = round(cor / tot, 4)
    print(f"victim {name}: {cor}/{tot} = {cor/tot:.2%}", flush=True)

with open("results/fused_isolated.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE fused isolated")
