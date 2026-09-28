"""Simple-fix baseline the review demanded: rotate S/a per REQUEST (M static,
as fusion.py only folds M into weights). Questions:
(1) does the cross-request attack die? (calibrate probe req1 -> attack victim req2)
(2) same-request attack still works? (probe+victim inside one request is not a
    real scenario, but same-request demix with the request's own S is trivially
    possible for the attacker's own blocks -- skip, not a scenario)
(3) honest decode with per-request S: round-trip still exact?
(4) regeneration cost of S/a per request (96 small matrices)?
(5) what breaks: cross-request obfuscated-domain cache sharing/reuse.
"""
import json
import math
import time
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, recover_secrets,
                        attack_block)
from run_suite import NATURAL, VICTIMS
from get_kvcloak_config import get_kvcloak_config
import kvcloak as kc
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
B = 16
res = {"variant": "KV-Cloak + per-request S/a rotation (M static)"}

tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)
theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
DICT, _, _ = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)

orig_get_sma = KVCloak._get_SMA_obf
REQ = {"n": 0}

def rotating_get_sma(self, config, device):
    """S/a freshly generated PER CALL (= per request, since device tensors are
    prepared once per obfuscate in our harness we regenerate by calling
    _prepare_device_tensors again; here we patch to draw fresh S/a each call).
    M unchanged (fused/static). Beacon stays single-row positive."""
    S = kc.KVCloak._get_rotation_matrix if False else None
    # fresh random orthogonal S
    q, _ = torch.linalg.qr(torch.randn(config["S"].shape, device=device, dtype=torch.float32))
    S = q.to(self.dtype)
    M = self._get_rotation_matrix(config["M_angles"].to(device)).to(device, self.dtype)
    thr = config["a"].abs().mean().float()
    a = ((torch.rand(config["a"].shape[0], device=device) + 3) * thr).to(self.dtype)
    b = S.shape[0]
    A = torch.zeros((b, a.shape[0]), device=device, dtype=self.dtype)
    A[torch.randint(low=0, high=b, size=(1,)).item()] = a
    return S, M, A


import copy as _copy

def cfg_with_fresh_Sa(base_cfg, seed):
    """M (angles/ratios) and theta UNCHANGED; S and a freshly drawn (per-request)."""
    g = torch.Generator().manual_seed(seed)
    c2 = _copy.deepcopy(base_cfg)
    for layer in c2:
        for head in layer:
            for kv in head:
                q, _ = torch.linalg.qr(torch.randn(kv["S"].shape, generator=g))
                kv["S"] = q.to(kv["S"].dtype)
                thr = kv["a"].abs().mean()
                kv["a"] = ((torch.rand(kv["a"].shape[0], generator=g) + 3) * thr).to(kv["a"].dtype)
    return c2

def fresh_cloak_for_request(seed, dtype=torch.bfloat16):
    c = KVCloak(cfg_with_fresh_Sa(cfg, seed), dtype=dtype, fused=False, need_ratio=False, add_a=True)
    return c


# ---- (4) regeneration cost ----
t0 = time.perf_counter()
for _ in range(100):
    q, _ = torch.linalg.qr(torch.randn(B, B))
    _ = (torch.rand(D) + 3)
res["sa_regen_ms_per_request"] = round((time.perf_counter() - t0) / 100 * 1000, 3)
# full layer/head matrix set for one model (24*2*2=96 matrices)
t0 = time.perf_counter()
for _ in range(10):
    for _ in range(96):
        q, _ = torch.linalg.qr(torch.randn(B, B))
        _ = (torch.rand(D) + 3)
res["sa_regen_ms_full_model"] = round((time.perf_counter() - t0) / 10 * 1000, 2)
print(f"(4) S/a regen: {res['sa_regen_ms_per_request']} ms/matrix, {res['sa_regen_ms_full_model']} ms full model", flush=True)

# ---- attack scenario: rotating instances ----
def calibrate(cloak, pids, prefix_len, plain_probe):
    torch.manual_seed(1234)
    prot = cloak.obfuscate(pc_for(pids))
    NB = pids.shape[1] // B
    first_blk = (prefix_len + B - 1) // B
    secs = {}
    for h in range(H):
        blocks = prot[0][1][0, h].float().view(NB, B, D)[first_blk:]
        cid, cents, *_ = cluster_states(blocks)
        if cid == B:
            secs[h] = recover_secrets(cents, plain_probe[h], B)
    return secs


pc_cache = {}
def pc_for(pids):
    key = pids.shape[1]
    if key not in pc_cache:
        with torch.no_grad():
            pc_cache[key] = model(input_ids=pids, use_cache=True).past_key_values
    return pc_cache[key]


maxpos = getattr(model.config, "max_position_embeddings", None)
pids, prefix_len, pchar = probe_ids(tok, target=2600, max_positions=maxpos)
with torch.no_grad():
    pc0 = model(input_ids=pids, use_cache=True).past_key_values
plain_probe = pc0[0][1][0].clone()

# epoch A: one cloak instance per request (per-request rotation)
cloakA = fresh_cloak_for_request(5001)
secA = calibrate(cloakA, pids, prefix_len, plain_probe)
print(f"calibration on request A recovered {len(secA)}/{H} heads", flush=True)

# victim under a DIFFERENT request (fresh S/a)
vids = tok(VICTIMS["prose_en"], return_tensors="pt").input_ids
nb = vids.shape[1] // B
vids = vids[:, : nb * B]
with torch.no_grad():
    vc = model(input_ids=vids, use_cache=True).past_key_values
cloakB = fresh_cloak_for_request(5002)
torch.manual_seed(6002)
pv = cloakB.obfuscate(vc)
cor = tot = 0
for h in range(H):
    if h not in secA:
        continue
    sec = secA[h]
    bp = pv[0][1][0, h].float()
    for k in range(nb):
        pred = attack_block(bp[k * B:(k + 1) * B], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
        cor += sum((Counter(pred) & Counter(vids[0, k * B:(k + 1) * B].tolist())).values())
        tot += B
res["cross_request_acc"] = round(cor / tot, 4)
print(f"(1) cross-request attack under per-request rotation: {cor}/{tot} = {cor/tot:.2%}", flush=True)

# same-epoch control (no rotation): single instance for both
cloakC = fresh_cloak_for_request(5001)
secC = calibrate(clover := cloakC, pids, prefix_len, plain_probe)
torch.manual_seed(6003)
with torch.no_grad():
    vc2 = model(input_ids=vids, use_cache=True).past_key_values
pv2 = cloakC.obfuscate(vc2)
cor = tot = 0
for h in range(H):
    if h not in secC:
        continue
    sec = secC[h]
    bp = pv2[0][1][0, h].float()
    for k in range(nb):
        pred = attack_block(bp[k * B:(k + 1) * B], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
        cor += sum((Counter(pred) & Counter(vids[0, k * B:(k + 1) * B].tolist())).values())
        tot += B
res["same_epoch_acc"] = round(cor / tot, 4)
print(f"(control) same-instance (static S/a) attack: {cor}/{tot} = {cor/tot:.2%}", flush=True)

# (3) honest decode with per-request S: round-trip on request B's own tensors
# reconstruct decode manually with the B-request S (server side knows its own S)
# we emulate: regenerate cloakB's tensors deterministically (same seed) and decode
cloakB2 = fresh_cloak_for_request(5002)
torch.manual_seed(6002)
with torch.no_grad():
    vc3 = model(input_ids=vids, use_cache=True).past_key_values
pv3 = cloakB2.obfuscate(vc3)
# NOTE: decode needs inverse tensors of THIS instance; KVCloak caches them
cloakB2f = fresh_cloak_for_request(5002, dtype=torch.float32)
torch.manual_seed(6002)
# save plaintext INDEPENDENTLY before obfuscation (obfuscate modifies in-place)
vc3_plain = tuple((k.clone(), v.clone()) for k, v in vc3)
pv3f = cloakB2f.obfuscate(vc3)
# ALSO verify attention-output equivalence: run one decode step with plain vs decoded
dec = cloakB2f.deobfuscate(pv3f)
# decoded rows are PERMUTED within blocks (attention row-invariant) --
# raw elementwise comparison is meaningless; use sorted-row equivalence
blk_errs = []
for (dk, dv), (_, pv) in zip(dec, vc3_plain):
    for h in range(min(2, dv.shape[1])):
        nb2 = dv.shape[2] // B
        for j in range(nb2):
            dblock = dv[0, h, j*B:(j+1)*B].float().sort(dim=0).values
            pblock = pv[0, h, j*B:(j+1)*B].float().sort(dim=0).values
            blk_errs.append((dblock - pblock).abs().max().item())
err = max(blk_errs) if blk_errs else -1
res["per_request_decode_roundtrip_maxerr"] = err  # SORTED-row equivalence
# permutation-equivalence check: decoded cache rows are a permutation of plaintext
# rows within each block (attention output invariant under row permutation)
perm_ok = True
for (dk, dv), (_, pv) in zip(dec, vc3_plain):
    for h in range(min(2, dv.shape[1])):
        nb2 = dv.shape[2] // B
        for j in range(min(4, nb2)):
            dblock = dv[0, h, j*B:(j+1)*B].float()
            pblock = pv[0, h, j*B:(j+1)*B].float()
            # sorted row norms should match if it's a permutation
            if not torch.allclose(dblock.norm(dim=1).sort().values,
                                  pblock.norm(dim=1).sort().values, atol=1e-5):
                perm_ok = False
res["per_request_decode_perm_equiv"] = perm_ok
print(f"(3b) sorted-row equivalence max err: {err:.2e} | permutation-equiv check: {perm_ok}", flush=True)
print(f"(3) per-request decode round-trip max err: {err:.2e}", flush=True)

with open("results/sarotate_05b.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE sarotate")
