"""Privacy task v3: rigorous sensitive-value identification.

Fixes all v2 issues:
- 5 true values per type × 6 types = 30 entities (not 1 per type)
- Attacker uses ALL blocks from the full text (no true-value position leak);
  entity position is NOT provided to the attacker
- Candidate scores saved for ALL candidates with tie-group analysis
- Tie-break: count strictly-higher candidates (g) and tie-group size (m);
  report "unique-first rate" and "top-k with random tie-break"
- Baseline: random candidate selection matched to generation process
  (fixed candidate set, uniformly random true index)
- Entity completeness: verify entity tokens appear in the actual model input
  (pre-truncation), report if any are lost to truncation
- Longer context to ensure all entity types produce ≥3 blocks (48 tokens)
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

# ---------- entity generators ----------
def synth_ssn(): return f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"
def synth_credit():
    d = [rng.randint(0,9) for _ in range(15)]
    s = sum(d[-1::-2]) + sum(2*x % 10 for x in d[-2::-2])
    d.append((10 - s % 10) % 10)
    return "".join(map(str, d))
def synth_password(): return f"P@ss{rng.randint(1000,9999)}"
def synth_mrn(): return f"MRN-{rng.randint(100000,999999)}"
NAMES = ["Abernathy","Birnbaum","Castellano","Delacroix","Eisenhauer","Fitzgerald",
         "Greenspan","Hoshino","Ivanov","Jankowski","Kowalski","Lindqvist","Montague",
         "Nakamura","Olejniczak","Pemberton","Quintanilla","Rothschild","Sandoval",
         "Thackeray","Ustinov","Volkonsky","Wetherbee","Xanthopoulos","Yamamoto"] + \
        [f"Subject{rng.randint(100,999)}" for _ in range(80)]
rng.shuffle(NAMES)
def synth_name(): return rng.choice(NAMES)
STREETS = ["Alder","Birchwood","Chesterton","Dunhill","Elmcrest","Foxglove",
           "Glenhaven","Hawthorne","Ivywood","Juniper"]
def synth_address(): return f"{rng.randint(100,9999)} {rng.choice(STREETS)} Street"
GEN = {"ssn": synth_ssn, "credit": synth_credit, "password": synth_password,
       "mrn": synth_mrn, "name": synth_name, "address": synth_address}

# Long neutral context (no type-specific words)
PRE = "I would like to confirm the following information for verification purposes with the system administrator:"
POST = "Please ensure this is recorded correctly in the central database for future audit and compliance requirements."

# ---------- build: 5 true values per type + 95 distractors each ----------
N_TRUE = 5
N_CAND = 100  # total candidates per type

tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)
DICT, _, _ = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)

samples = []  # each: (type, true_val, context_text, all_token_ids, entity_token_set)
for etype, gen in GEN.items():
    # generate all candidates
    candidates = []
    seen = set()
    while len(candidates) < N_CAND:
        v = gen()
        if v not in seen:
            candidates.append(v)
            seen.add(v)
    # first 5 are true values
    for ti in range(N_TRUE):
        true_val = candidates[ti]
        text = f"{PRE} {true_val}. {POST}"
        ids = tok(text, return_tensors="pt").input_ids
        nb = ids.shape[1] // B
        # verify entity tokens present in input (pre-truncation)
        ent_alone = tok(true_val, return_tensors="pt").input_ids[0].tolist()
        all_ids_full = ids[0].tolist()
        present = sum(1 for t in set(ent_alone) if t in all_ids_full)
        total_uniq = len(set(ent_alone))
        # truncate for blocks
        ids_t = ids[:, :nb * B]
        all_ids = ids_t[0].tolist()
        # entity tokens in truncated input (for completeness check)
        present_trunc = sum(1 for t in set(ent_alone) if t in all_ids)
        samples.append({
            "type": etype, "true_val": true_val, "true_idx": ti,
            "candidates": candidates, "n_blocks": nb,
            "n_tokens_full": len(all_ids_full), "n_tokens_trunc": len(all_ids),
            "entity_uniq_tokens": total_uniq,
            "entity_present_full": present, "entity_present_trunc": present_trunc,
            "entity_complete": present_trunc == total_uniq,
        })
print(f"{len(samples)} samples built", flush=True)
incomplete = [s for s in samples if not s["entity_complete"]]
print(f"entity complete in truncated input: {len(samples)-len(incomplete)}/{len(samples)}", flush=True)

# ---------- calibrate attack ----------
theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)
pids, prefix_len, _ = probe_ids(tok, target=2600,
                                max_positions=getattr(model.config, "max_position_embeddings", None))
with torch.no_grad():
    pc = model(input_ids=pids, use_cache=True).past_key_values
NB = pids.shape[1] // B
first_blk = (prefix_len + B - 1) // B
plain_probe = pc[0][1][0, :, first_blk*B:NB*B].clone()
torch.manual_seed(1234)
prot = cloak.obfuscate(pc)
nblk_pad = prot[0][1].shape[2] // B
secrets = {}
for h in range(H):
    blocks = prot[0][1][0, h].float().view(nblk_pad, B, D)[first_blk:NB]
    cid, cents, *_ = cluster_states(blocks)
    if cid == B:
        secrets[h] = recover_secrets(cents, plain_probe[h], B)
print(f"calibration: {len(secrets)}/{H}", flush=True)

# ---------- attack + evaluation ----------
def score_candidate_tokens(cand_ids, multiset):
    """fraction of candidate's unique tokens present in multiset"""
    return sum(1 for t in cand_ids if multiset.get(t, 0) > 0) / len(cand_ids) if cand_ids else 0

# pre-tokenize all candidates ONCE (outside the scoring loop)
cand_token_cache = {}
for s in samples:
    for v in s["candidates"]:
        if v not in cand_token_cache:
            cand_token_cache[v] = list(set(tok(v, return_tensors="pt").input_ids[0].tolist()))

results = []
for si, s in enumerate(samples):
    true_val = s["true_val"]
    text = f"{PRE} {true_val}. {POST}"
    vids = tok(text, return_tensors="pt").input_ids[:, :s["n_blocks"]*B]
    nb = s["n_blocks"]

    with torch.no_grad():
        vc = model(input_ids=vids, use_cache=True).past_key_values
    torch.manual_seed(7000 + si)
    pv = cloak.obfuscate(vc)

    # attacker uses ALL blocks (no position knowledge)
    multiset = Counter()
    for h in range(H):
        if h not in secrets:
            continue
        sec = secrets[h]
        bp = pv[0][1][0, h].float()
        for k in range(nb):
            pred = attack_block(bp[k*B:(k+1)*B], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
            multiset.update(pred)

    # score all candidates
    scores = []
    for ci, v in enumerate(s["candidates"]):
        sc = score_candidate_tokens(cand_token_cache[v], multiset)
        scores.append((sc, ci, v))
    true_score = scores[s["true_idx"]][0]

    # tie analysis
    strictly_higher = sum(1 for sc, ci, v in scores if sc > true_score)
    tie_group = sum(1 for sc, ci, v in scores if sc == true_score)  # includes true

    # unique-first
    unique_first = strictly_higher == 0

    # top-k with random tie-break (expected success probability)
    def topk_prob(k):
        if strictly_higher >= k:
            return 0.0
        # true has rank strictly_higher + uniform position in tie_group
        return max(0.0, (k - strictly_higher) / tie_group)

    top1_prob = topk_prob(1)
    top5_prob = topk_prob(5)
    top10_prob = topk_prob(10)

    # baseline: random candidate selection (uniform from N_CAND)
    random_top1 = 1 / N_CAND
    random_top5 = 5 / N_CAND
    random_top10 = 10 / N_CAND

    results.append({
        "type": s["type"], "true_idx": s["true_idx"],
        "true_score": round(true_score, 4),
        "strictly_higher": strictly_higher,
        "tie_group_size": tie_group,
        "unique_first": unique_first,
        "top1_prob": round(top1_prob, 4),
        "top5_prob": round(top5_prob, 4),
        "top10_prob": round(top10_prob, 4),
        "entity_complete": s["entity_complete"],
        "n_blocks": nb,
    })
    print(f"  {s['type']:10s}[{s['true_idx']}] score={true_score:.3f} g={strictly_higher} m={tie_group} "
          f"uniq1st={unique_first} p@1={top1_prob:.2f} p@5={top5_prob:.2f} "
          f"complete={s['entity_complete']}", flush=True)

# ---------- aggregate ----------
n = len(results)
n_complete = sum(1 for r in results if r["entity_complete"])
res_complete = [r for r in results if r["entity_complete"]]

summary = {
    "n_entities": n, "n_types": len(GEN), "n_true_per_type": N_TRUE,
    "n_candidates": N_CAND,
    "entity_complete_rate": round(n_complete / n, 3),
    "attack_unique_first_rate": round(sum(1 for r in results if r["unique_first"]) / n, 3),
    "attack_top1_prob_mean": round(sum(r["top1_prob"] for r in results) / n, 4),
    "attack_top5_prob_mean": round(sum(r["top5_prob"] for r in results) / n, 4),
    "attack_top10_prob_mean": round(sum(r["top10_prob"] for r in results) / n, 4),
    "random_top1": round(1 / N_CAND, 4),
    "random_top5": round(5 / N_CAND, 4),
    "random_top10": round(10 / N_CAND, 4),
    "complete_only_unique_first": round(sum(1 for r in res_complete if r["unique_first"]) / max(1, len(res_complete)), 3),
    "complete_only_top5_prob": round(sum(r["top5_prob"] for r in res_complete) / max(1, len(res_complete)), 4),
    "note": "top-k prob = expected success under uniform random tie-breaking within equal-score group",
}
# per-type breakdown
by_type = {}
for r in results:
    t = r["type"]
    if t not in by_type:
        by_type[t] = []
    by_type[t].append(r)
for t, rs in by_type.items():
    summary[f"type_{t}"] = {
        "n": len(rs),
        "uniq1st": sum(1 for r in rs if r["unique_first"]),
        "top5_prob": round(sum(r["top5_prob"] for r in rs) / len(rs), 3),
        "complete": sum(1 for r in rs if r["entity_complete"]),
    }

print(json.dumps(summary, indent=1))
with open("results/privacy_v3.json", "w") as f:
    json.dump({"summary": summary, "per_entity": results}, f, indent=1)
print("DONE privacy v3")
