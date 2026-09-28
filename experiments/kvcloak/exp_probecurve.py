"""Probe-length curve: P(collect all b states) and full-secret recovery vs
number of probe blocks N, over independent fresh-permutation draws.
Compares against coupon-collector bound."""
import json
import math

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        cluster_states, recover_secrets, ground_truth_errors)
from run_suite import NATURAL
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
B = 16
res = {}
tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)
pids, _pl, _ = probe_ids(tok, target=8192, max_positions=getattr(model.config, "max_position_embeddings", None))  # long probe; use prefixes of its blocks
theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
with torch.no_grad():
    pc = model(input_ids=pids, use_cache=True).past_key_values
plain_probe = pc[0][1][0].clone()
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)

NBLOCKS = [16, 24, 32, 48, 54, 64, 80, 96, 128, 160, 192, 256]
TRIALS = 10
Hb = sum(1.0 / k for k in range(1, B + 1))

out = []
for N in NBLOCKS:
    full_cnt = 0
    rec_ok = 0
    min_states = B
    for trial in range(TRIALS):
        ids = pids[:, : N * B]
        with torch.no_grad():
            pct = model(input_ids=ids, use_cache=True).past_key_values
        torch.manual_seed(10_000 + trial * 97 + N)
        prot = cloak.obfuscate(pct)
        head_full = True
        ok = True
        for h in range(H):
            blocks = prot[0][1][0, h].float().view(N, B, D)
            cid, cents, *_ = cluster_states(blocks)
            min_states = min(min_states, cid)
            if cid != B:
                head_full = False
                ok = False
                continue
            try:
                sec = recover_secrets(cents, plain_probe[h], B)
                errs = ground_truth_errors(sec, cfg[0][h][1], plain_probe[h])
                if errs["S_min_signed_cos"] < 0.999 or errs["cos_a"] < 0.999:
                    ok = False
            except Exception:
                ok = False
        full_cnt += head_full
        rec_ok += ok
    # coupon-collector prediction: P(all states) with N blocks
    # P(specific state missed) = (1-1/b)^N; union bound; exact via inclusion-exclusion approx
    p_miss = 1 - (1 - (1 - 1 / B) ** N) ** B
    out.append({"blocks": N, "P_all_states": full_cnt / TRIALS, "P_full_recovery": rec_ok / TRIALS,
                "min_states_seen": min_states, "coupon_pred_all": round(1 - p_miss, 3)})
    print(f"N={N:4d} blocks ({N*B:5d} tok): all-states {full_cnt}/{TRIALS} | full-recovery {rec_ok}/{TRIALS} | coupon-pred {1-p_miss:.3f}", flush=True)

res["probe_curve"] = out
res["theory"] = {"b": B, "bHb": round(B * Hb, 1), "note": "expected blocks to collect all states"}
with open("results/probecurve_05b.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE probecurve")
