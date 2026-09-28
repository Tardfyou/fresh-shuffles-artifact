"""Lifecycle simulation: the cross-tenant confidentiality figure.

Epoch 1: attacker calibrates with one probe request; service then processes
10 DIFFERENT users' requests sequentially (fresh permutations each); attacker
demixes each with the same recovered keys.  -> key rotation (epoch 2, new
instance/secrets): old keys fail.  -> attacker recalibrates on epoch 2 with
one new probe: attack works again.
"""
import json
import math
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, recover_secrets,
                        attack_block)
from run_suite import NATURAL
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
B = 16
tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)
theta = build_theta(model, tok, NATURAL)
DICT, _, _ = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)

USERS = {
    "user1_medical": "Patient presents with persistent cough, mild fever, and shortness of breath lasting ten days. Chest imaging shows bilateral ground-glass opacities. Laboratory results indicate elevated inflammatory markers. Treatment plan includes antiviral therapy, supplemental oxygen, and monitoring of blood oxygen saturation levels every four hours.",
    "user2_legal": "The non-disclosure agreement shall remain in effect for a period of five years following the termination of employment. Confidential information includes, but is not limited to, source code, algorithm designs, customer lists, financial projections, and internal correspondence. Any breach shall be subject to liquidated damages.",
    "user3_finance": "The portfolio rebalancing strategy reduces equity exposure from seventy percent to fifty-five percent, increasing allocation to investment-grade bonds and money market instruments. Historical backtesting over twenty years shows improved Sharpe ratio with reduced maximum drawdown during recession periods.",
    "user4_engineering": "The bridge design uses a cable-stayed configuration with two pylons reaching one hundred eighty meters. Wind tunnel testing validated aerodynamic stability under typhoon conditions. The deck segments are prefabricated off-site and installed using a floating crane to minimize marine traffic disruption.",
    "user5_personal": "Hi mom, I finally moved into the new apartment last weekend. The kitchen is bigger than I expected and there is a small balcony facing south. I planted some herbs in pots. Come visit next month if you can, I will cook dinner. The cats are adjusting slowly but they like the window sill.",
    "user6_code": "def analyze_timestamps(events):\n    from collections import defaultdict\n    buckets = defaultdict(int)\n    for e in events:\n        hour = e.timestamp.hour\n        buckets[hour] += 1\n    peak = max(buckets.values())\n    return sorted(buckets.items()), peak\n\nif __name__ == '__main__':\n    import random, datetime\n    base = datetime.datetime(2026, 1, 1)\n    events = [type('E', (), {'timestamp': base + datetime.timedelta(minutes=random.random() * 1440)})() for _ in range(1000)]\n    print(analyze_timestamps(events))",
    "user7_travel": "The itinerary covers Kyoto in early April for cherry blossom season. Day one: Fushimi Inari shrine at sunrise, then Gion district in the afternoon. Day two: Arashiyama bamboo grove and the monkey park. Day three: day trip to Nara to see the deer and Todaiji temple. Budget approximately eight hundred dollars excluding flights.",
    "user8_academic": "We evaluate on three benchmarks following prior work. Table 2 reports accuracy and F1. Our method improves over the strongest baseline by 3.2 points on average. Ablation studies in Table 3 show that removing the reranking component reduces performance by 1.8 points, confirming its contribution. Statistical significance is assessed via paired bootstrap tests.",
    "user9_recipe": "Start by preheating the oven to 200 degrees Celsius. Cream together 200 grams of softened butter with 150 grams of brown sugar until light and fluffy. Beat in two eggs one at a time, then fold in 280 grams of flour, half a teaspoon of baking soda, and 200 grams of dark chocolate chunks. Bake for 12 minutes.",
    "user10_secrets": "My password recovery answers: first pet was named Biscuit, born in Springfield. Mother's maiden name Whitfield. First school was Jefferson Elementary. Favorite book is the Old Man and the Sea. The safe combination is 47-22-9 and the account number ends in 8853.",
}


def calibrate(cloak, pids, plain_probe, seed):
    with torch.no_grad():
        pc = model(input_ids=pids, use_cache=True).past_key_values
    torch.manual_seed(seed)
    prot = cloak.obfuscate(pc)
    secrets = {}
    for h in range(H):
        blocks = prot[0][1][0, h].float().view(-1, B, D)
        cid, cents, *_ = cluster_states(blocks)
        if cid == B:
            secrets[h] = recover_secrets(cents, plain_probe[h], B)
    return secrets


def attack_request(prot, vids, secrets):
    nb = vids.shape[1] // B
    cor = tot = 0
    for h in range(H):
        if h not in secrets:
            return None
        sec = secrets[h]
        bp = prot[0][1][0, h].float()
        for k in range(nb):
            pred = attack_block(bp[k * B:(k + 1) * B], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
            cor += sum((Counter(pred) & Counter(vids[0, k * B:(k + 1) * B].tolist())).values())
            tot += B
    return round(cor / tot, 4)


pids, _pl, _ = probe_ids(tok, target=2560, max_positions=getattr(model.config, "max_position_embeddings", None))
with torch.no_grad():
    pc0 = model(input_ids=pids, use_cache=True).past_key_values
plain_probe = pc0[0][1][0].clone()

log = []
# ---------------- EPOCH 1 ----------------
torch.manual_seed(42)
cfg1 = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
cloak1 = KVCloak(cfg1, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)
log.append(("epoch1", "attacker probe (calibration)", "-"))
sec1 = calibrate(cloak1, pids, plain_probe, seed=1001)
log.append(("epoch1", f"recovered {len(sec1)}/{H} heads", "-"))
seed = 2000
for name, text in USERS.items():
    vids = tok(text, return_tensors="pt").input_ids
    nb = vids.shape[1] // B
    vids = vids[:, : nb * B]
    with torch.no_grad():
        vc = model(input_ids=vids, use_cache=True).past_key_values
    torch.manual_seed(seed := seed + 1)
    prot = cloak1.obfuscate(vc)
    acc = attack_request(prot, vids, sec1)
    log.append(("epoch1", name, acc))

# ---------------- key rotation -> EPOCH 2 ----------------
torch.manual_seed(4242)
cfg2 = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
cloak2 = KVCloak(cfg2, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)
post = tok(USERS["user10_secrets"], return_tensors="pt").input_ids
post = post[:, : post.shape[1] // B * B]
with torch.no_grad():
    vc = model(input_ids=post, use_cache=True).past_key_values
torch.manual_seed(3001)
prot = cloak2.obfuscate(vc)
acc = attack_request(prot, post, sec1)
log.append(("epoch2", "user10_secrets with STALE epoch-1 keys", acc))

# ---------------- recalibration on epoch 2 ----------------
sec2 = calibrate(cloak2, pids, plain_probe, seed=1002)
torch.manual_seed(3002)
with torch.no_grad():
    vc = model(input_ids=post, use_cache=True).past_key_values
prot = cloak2.obfuscate(vc)
acc = attack_request(prot, post, sec2)
log.append(("epoch2", "user10_secrets after ONE recalibration probe", acc))

print(f"{'phase':8s} | request | accuracy")
print("-" * 70)
for ph, name, acc in log:
    print(f"{ph:8s} | {name:44s} | {acc}")
with open("results/lifecycle_05b.json", "w") as f:
    json.dump([{"phase": p, "request": n, "acc": a} for p, n, a in log], f, indent=1)
print("DONE lifecycle")
