"""Official-entry reproduction: run the AUTHORS' pipeline end-to-end on real
LMSYS-Chat-1M conversations, then attack the files their code produced.

Steps:
1. fetch ~16 conversations from the public mirror of lmsys-chat-1m
2. build their cache layout: cache/bfloat16/lmsys-chat-1m_1k/<model>/<hash>/origin/past_key_values.pt
3. write theta json; run THEIR get_kvcloak_config.py main() (subprocess) to
   generate defense/config/kvcloak/b16_S1_M1_t2/<model>.pt
4. run THEIR kvcloak.py main() (subprocess) -- the official obfuscation entry
5. attacker: calibrate on own probe with the generated config; demix the
   protected files their pipeline wrote; score token multiset accuracy.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, recover_secrets,
                        attack_block)
from run_suite import NATURAL

SBX = "official_repro"
MODEL_NAME = "Qwen2.5-0.5B"
B = 16
os.makedirs(SBX, exist_ok=True)
os.chdir(SBX)

torch.manual_seed(0)
torch.set_num_threads(16)
tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)

# ---------- 1. conversations ----------
from datasets import load_dataset
ds = load_dataset("AarushSah/lmsys-chat-1m", split="train", streaming=True)
convs = []
for r in ds:
    if r.get("language") not in ("English", None):
        continue
    text = r["conversation"][0]["content"].strip()
    if len(text) < 400:
        continue
    convs.append({"hash": hashlib.sha256(text.encode()).hexdigest()[:32], "text": text})
    if len(convs) >= 16:
        break
print(f"fetched {len(convs)} conversations", flush=True)

# ---------- 2. origin caches in their layout ----------
cache_root = f"cache/bfloat16/lmsys-chat-1m_1k/{MODEL_NAME}"
truth = {}
for c in convs:
    ids = tok(c["text"], return_tensors="pt").input_ids[:, :1024]
    ids = ids[:, : ids.shape[1] // B * B]
    if ids.shape[1] < B:
        continue
    with torch.no_grad():
        pv = model(input_ids=ids, use_cache=True).past_key_values
    legacy = tuple((k.to(torch.bfloat16), v.to(torch.bfloat16)) for k, v in pv.to_legacy_cache())
    d = os.path.join(cache_root, c["hash"], "origin")
    os.makedirs(d, exist_ok=True)
    torch.save(legacy, os.path.join(d, "past_key_values.pt"))
    truth[c["hash"]] = ids
print(f"wrote {len(truth)} origin caches", flush=True)

# ---------- 3. theta json + THEIR config generation ----------
theta = build_theta(model, tok, [c["text"][:2000] for c in convs[:8]])
os.makedirs("defense/config/kvcloak/theta", exist_ok=True)
with open(f"defense/config/kvcloak/theta/{MODEL_NAME}.json", "w") as f:
    json.dump(theta, f)
from huggingface_hub import snapshot_download
model_path = snapshot_download("Qwen/Qwen2.5-0.5B")
cmd = [sys.executable, "../get_kvcloak_config.py", "--model-name", MODEL_NAME,
       "--model-path", model_path]
p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
print("their config gen rc:", p.returncode, p.stdout[-200:], p.stderr[-300:] if p.returncode else "", flush=True)
cfg_path = f"defense/config/kvcloak/b16_S1_M1_t2/{MODEL_NAME}.pt"
assert os.path.exists(cfg_path), "config not generated"

# ---------- 4. THEIR obfuscation entry ----------
cmd = [sys.executable, "../kvcloak.py", "--device", "cpu", "--model-name", MODEL_NAME,
       "--cache-path", cache_root, "--config-path", cfg_path]
p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
print("their obfuscation rc:", p.returncode, (p.stdout[-300:] if p.returncode else "ok"), p.stderr[-400:] if p.returncode else "", flush=True)
protected_files = sorted(glob.glob(f"{cache_root}/*/kvcloak/past_key_values.pt"))
print(f"protected files produced: {len(protected_files)}", flush=True)

# ---------- 5. attack ----------
cfg = torch.load(cfg_path)
from kvcloak import KVCloak
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)
pids, _pl, _ = probe_ids(tok, target=2560, max_positions=getattr(model.config, "max_position_embeddings", None))
with torch.no_grad():
    pc = model(input_ids=pids, use_cache=True).past_key_values
plain_probe = pc[0][1][0].clone()
torch.manual_seed(1001)
prot = cloak.obfuscate(pc)
secrets = {}
for h in range(H):
    blocks = prot[0][1][0, h].float().view(-1, B, D)
    cid, cents, *_ = cluster_states(blocks)
    assert cid == B, f"calibration incomplete: {cid}"
    secrets[h] = recover_secrets(cents, plain_probe[h], B)
DICT, _, _ = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)

accs = {}
for pf in protected_files:
    hsh = pf.split("/")[-3]
    ids = truth[hsh]
    nb = ids.shape[1] // B
    kv = torch.load(pf, weights_only=True)
    cor = tot = 0
    for h in range(H):
        sec = secrets[h]
        V = kv[0][1][0, h].float()
        for k in range(nb):
            C = V[k * B:(k + 1) * B]
            pred = attack_block(C, sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
            cor += sum((Counter(pred) & Counter(ids[0, k * B:(k + 1) * B].tolist())).values())
            tot += B
    accs[hsh[:8]] = round(cor / tot, 4)
print("attack on THEIR pipeline output:", accs, flush=True)
vals = list(accs.values())
res = {"files": len(protected_files), "accs": accs,
       "min": min(vals), "mean": round(sum(vals) / len(vals), 4)}
os.chdir("..")
with open("results/official_repro.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"OFFICIAL-ENTRY RESULT: {res['files']} files, min {res['min']}, mean {res['mean']}", flush=True)
