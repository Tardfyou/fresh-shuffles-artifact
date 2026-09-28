"""Independent evaluation: frozen attack, held-out sessions, independent keys.

Protocol (frozen):
- 3 representative models: Qwen2.5-0.5B, Llama-3.2-1B, Phi-3-mini
- 5 independent key epochs per model
- 10 held-out LMSYS sessions per epoch (unique conversations)
- Report: calibration success, probe cost, conditional recovery, end-to-end
  (including calibration-failure heads as zero), per-epoch variance
"""
import json
import random
import time
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, recover_secrets,
                        attack_block)
from run_suite import NATURAL
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
B = 16
rng = random.Random(20260929)

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "qwen05"),
    ("unsloth/Llama-3.2-1B", "llama1b"),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
]
N_EPOCHS = 5
N_VICTIMS = 10

# held-out sessions from LMSYS (streaming, no overlap with anything used before)
from datasets import load_dataset

res_all = {}
for model_name, tag in MODELS:
    tok, model = load_model(model_name, threads=16)
    layers, H, D = model_dims(model)
    DICT, _, _ = build_dictionary(model, H, D)
    Dn = torch.nn.functional.normalize(DICT, dim=-1)
    maxpos = getattr(model.config, "max_position_embeddings", None)

    # fetch held-out sessions (fresh, not used in any prior experiment)
    ds = load_dataset("AarushSah/lmsys-chat-1m", split="train", streaming=True)
    sessions = []
    for r in ds:
        if r.get("language") not in ("English", None):
            continue
        t = r["conversation"][0]["content"].strip()
        if len(t) < 300 or len(t) > 1500:
            continue
        sessions.append(t)
        if len(sessions) >= N_VICTIMS:
            break
    print(f"[{tag}] {len(sessions)} held-out sessions", flush=True)

    epochs = []
    for ep in range(N_EPOCHS):
        t0 = time.time()
        theta = build_theta(model, tok, NATURAL)
        torch.manual_seed(99000 + ep * 7)
        cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
        cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)

        # calibration
        pids, prefix_len, _ = probe_ids(tok, target=2600, max_positions=maxpos)
        probe_tokens = pids.shape[1]
        with torch.no_grad():
            pc = model(input_ids=pids, use_cache=True).past_key_values
        NB = pids.shape[1] // B
        first_blk = (prefix_len + B - 1) // B
        plain_probe = pc[0][1][0, :, first_blk*B:NB*B].clone()
        torch.manual_seed(1234 + ep * 77)
        prot = cloak.obfuscate(pc)
        nblk_pad = prot[0][1].shape[2] // B
        secrets = {}
        for h in range(H):
            blocks = prot[0][1][0, h].float().view(nblk_pad, B, D)[first_blk:NB]
            cid, cents, *_ = cluster_states(blocks)
            if cid == B:
                secrets[h] = recover_secrets(cents, plain_probe[h], B)
        cal_success = len(secrets)
        cal_rate = cal_success / H

        # attack victims
        per_session = []
        for si, sess in enumerate(sessions):
            vids = tok(sess, return_tensors="pt").input_ids
            nb = vids.shape[1] // B
            if nb == 0:
                continue
            vids = vids[:, : nb * B]
            with torch.no_grad():
                vc = model(input_ids=vids, use_cache=True).past_key_values
            torch.manual_seed(777000 + ep * 100 + si)
            pv = cloak.obfuscate(vc)

            # conditional recovery (only calibrated heads)
            cond_cor = cond_tot = 0
            # end-to-end (failed heads count as zero)
            e2e_cor = e2e_tot = 0
            for h in range(H):
                sec = secrets.get(h)
                bp = pv[0][1][0, h].float()
                for k in range(nb):
                    true = vids[0, k*B:(k+1)*B].tolist()
                    if sec is not None:
                        pred = attack_block(bp[k*B:(k+1)*B], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
                        cond_cor += sum((Counter(pred) & Counter(true)).values())
                    cond_tot += B
                    e2e_tot += B  # failed heads contribute 0 to e2e_cor

            per_session.append({
                "session": si, "tokens": int(vids.shape[1]),
                "cond_acc": round(cond_cor / cond_tot, 4) if cond_tot else None,
            })

        # fresh-secrets control on 3 sessions
        torch.manual_seed(888000 + ep)
        cfg_ctrl = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
        cloak_ctrl = KVCloak(cfg_ctrl, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)
        ctrl_cor = ctrl_tot = 0
        for si in range(min(3, len(sessions))):
            vids = tok(sessions[si], return_tensors="pt").input_ids
            nb = vids.shape[1] // B
            if nb == 0:
                continue
            vids = vids[:, : nb * B]
            with torch.no_grad():
                vc = model(input_ids=vids, use_cache=True).past_key_values
            torch.manual_seed(999000 + ep * 10 + si)
            pv_ctrl = cloak_ctrl.obfuscate(vc)
            for h in range(H):
                sec = secrets.get(h)
                if sec is None:
                    continue
                bp = pv_ctrl[0][1][0, h].float()
                for k in range(nb):
                    pred = attack_block(bp[k*B:(k+1)*B], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
                    ctrl_cor += sum((Counter(pred) & Counter(vids[0, k*B:(k+1)*B].tolist())).values())
                    ctrl_tot += B

        elapsed = time.time() - t0
        cond_mean = sum(s["cond_acc"] for s in per_session if s["cond_acc"] is not None) / max(1, len(per_session))
        epoch_rec = {
            "epoch": ep, "cal_heads": cal_success, "cal_total": H,
            "cal_rate": round(cal_rate, 3), "probe_tokens": probe_tokens,
            "cond_recovery_mean": round(cond_mean, 4),
            "cond_recovery_min": min((s["cond_acc"] for s in per_session if s["cond_acc"] is not None), default=0),
            "ctrl_acc": round(ctrl_cor / ctrl_tot, 4) if ctrl_tot else None,
            "elapsed_s": round(elapsed, 1),
            "sessions": per_session,
        }
        epochs.append(epoch_rec)
        print(f"[{tag}] epoch {ep}: cal={cal_success}/{H}, cond={cond_mean:.4f}, ctrl={epoch_rec['ctrl_acc']}, {elapsed:.0f}s", flush=True)

    # aggregate per model
    cal_rates = [e["cal_rate"] for e in epochs]
    cond_means = [e["cond_recovery_mean"] for e in epochs]
    ctrls = [e["ctrl_acc"] for e in epochs if e["ctrl_acc"] is not None]
    res_all[tag] = {
        "model": model_name, "H": H, "D": D, "n_epochs": N_EPOCHS,
        "cal_rate_mean": round(sum(cal_rates) / len(cal_rates), 3),
        "cond_recovery_mean_of_epoch_means": round(sum(cond_means) / len(cond_means), 4),
        "cond_recovery_min_across_epochs": min(cond_means),
        "ctrl_mean": round(sum(ctrls) / len(ctrls), 4) if ctrls else None,
        "epochs": epochs,
    }
    print(f"[{tag}] SUMMARY: cal={res_all[tag]['cal_rate_mean']}, "
          f"cond={res_all[tag]['cond_recovery_mean_of_epoch_means']}, "
          f"ctrl={res_all[tag]['ctrl_mean']}", flush=True)
    del model
    import gc; gc.collect()

with open("results/independent_eval.json", "w") as f:
    json.dump(res_all, f, indent=1)
print("DONE independent eval")
