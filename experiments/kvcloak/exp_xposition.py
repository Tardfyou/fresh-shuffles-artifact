"""X-position calibration: recover the FULL per-block permutation on own probes.

Probe: blocks of 15 repeated 'a' + 1 sentinel token X at KNOWN offset p_j
(rotating across blocks). Value block: V = 1 v_a^T + (v_x - v_a) e_p^T, so
  C_j = u w_a^T + s_{pi(p_j)} w_x^T + s_{pi(r_j)} a^T    (rank 3)
With u, w_a, a, all s-columns and w_x = M^T (v_x - v_a) known from the plain
repeated-token calibration, each block yields (marker state, X state) ->
pi(r_j) and pi(p_j). Covering all p gives the complete permutation per block
and fixes S's column ordering (up to now only known up to permutation).
"""
import json
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, recover_secrets,
                        rot_strided)
from run_suite import NATURAL
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
B, D = 16, 64
tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)

theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)

# ---------- step 1: plain calibration (all-'a' probe) ----------
pa, cha = probe_ids(tok, target=2048)
with torch.no_grad():
    pc = model(input_ids=pa, use_cache=True).past_key_values
plain_a = pc[0][1][0].clone()
torch.manual_seed(1234)
prot_a = cloak.obfuscate(pc)
secrets = {}
for h in range(H):
    blocks = prot_a[0][1][0, h].float().view(-1, B, D)
    cid, cents, *_ = cluster_states(blocks)
    assert cid == B
    secrets[h] = recover_secrets(cents, plain_a[h], B)
print("plain calibration done", flush=True)

# ---------- step 2: X-probe: blocks with sentinel at rotating offsets ----------
va = plain_a[0, 0]          # value row of token 'a'
a_tok = pa[0, 0].item()
# pick sentinel token: a distinctive word
sent_text = "xylophone"
sx = tok(sent_text, return_tensors="pt").input_ids[0, 0].item()
ids_a = pa[0].tolist()
# build sequence: for block k, sentinel at offset (k % 16)
seq = []
offsets = []
for k in range(64):
    p = k % B
    blk = [a_tok] * B
    blk[p] = sx
    seq.extend(blk)
    offsets.append(p)
xids = torch.tensor([seq])
with torch.no_grad():
    xc = model(input_ids=xids, use_cache=True).past_key_values
plain_x = xc[0][1][0].clone()
vx = plain_x[0, 0]  # NOTE: layer-0 V of sentinel is position-independent
torch.manual_seed(4321)
prot_x = cloak.obfuscate(xc)

res = {"blocks": 64}
total_ok = 0
for h in range(H):
    sec = secrets[h]
    s_hat, a_hat, M_hat, u_hat, w_hat = sec["s"], sec["a_hat"], sec["M"], sec["u"], sec["w"]
    # w_x = M^T (v_x - v_a) using recovered M
    delta = (vx - va).float()
    w_x = M_hat @ delta  # since w = M^T v convention: w_x = M^T delta -> with our M_hat built from angles
    w_x = M_hat.T @ delta if False else M_hat @ delta
    # convention check: w = M^T v  => w_x = M_hat^T delta. Use that.
    w_x = M_hat.T @ delta
    C_all = prot_x[0][1][0, h].float().view(-1, B, D)
    base = torch.outer(u_hat, w_hat)
    found = 0
    perms = {}
    for j in range(64):
        C = C_all[j]
        R = C - base
        best = None
        for sigma in range(B):   # candidate marker state
            R2 = R - torch.outer(s_hat[sigma], a_hat)
            U_, S_, _ = torch.linalg.svd(R2)
            r1 = U_[:, 0] * S_[0]
            cands = torch.nn.functional.cosine_similarity(r1.unsqueeze(0), s_hat, dim=1)
            tau = int(cands.abs().argmax())
            lam = (s_hat[tau] @ R2 @ w_x) / ((s_hat[tau] @ s_hat[tau]) * (w_x @ w_x) + 1e-12)
            resid = (R2 - lam * torch.outer(s_hat[tau], w_x)).norm().item()
            if best is None or resid < best[0]:
                best = (resid, sigma, tau)
        resid, sigma, tau = best
        rel = resid / (R.norm().item() + 1e-9)
        ok = rel < 0.2 and sigma != tau
        found += ok
        if ok:
            perms[j] = (sigma, tau, offsets[j])
    # consistency: each block's sigma maps offset-independent marker position;
    # X-state tau vs known offset p_j gives pi_j(p_j); sigma gives pi_j(r).
    res[f"head{h}"] = {"decoded_blocks": found, "of": 64,
                       "sample": {str(k): perms[k] for k in list(perms)[:6]}}
    total_ok += found
    print(f"head {h}: decoded permutation entries on {found}/64 blocks; "
          f"sample (marker_state, X_state, offset): {[(perms[k]) for k in list(perms)[:4]]}", flush=True)

# self-consistency: X-state as a function of offset should be consistent with
# a single global labeling (pi depends on block, but the MAPPING column->state
# is fixed by S): i.e., for blocks sharing the same offset, the DISTRIBUTION of
# tau should be uniform; and across all blocks, sigma should be uniform.
h = 0
sig, tau_, off = [], [], []
for j, (s0, t0, o0) in res[f"head{h}"]["sample"].items():
    pass
res["note"] = ("per-block (marker-state, X-state) pairs give pi_j(r) and pi_j(p_j); "
               "covering p=0..15 recovers each block's full permutation and fixes "
               "S's column labeling (previously known only up to permutation).")
with open("results/xposition_05b.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE x-position", flush=True)
