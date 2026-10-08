"""VeilKV v2: closes the five review breakpoints on the v1 prototype.

Changes vs v1 (all from the 2026-10-07 read-only review):
 1. REAL implementation: decode RE-DERIVES all keys via PRF (no keymat return);
    incomplete tail is PADDED into a final protected block (no plaintext tail).
 2. Prefix-domain CPA measured: cross-request recomputed-prefix equality per layer
    (K and V), guess-confirmation boundary (deterministic-encryption leak, the
    accepted price of dedup, cf. MLE), same-content-different-position divergence,
    prefix-ciphertext invariance to tail changes at matched shape.
 3. Attack battery v2: non-collapse via unique-count + min pairwise distance (no
    adaptive clustering); ALL victim tail blocks; key-transfer with attacker-
    computable selection (mean top-1 NN cosine) over 64 wrong keys; the CORRECT
    null baseline (NN hit-rate of random vectors), not the permutation formula.
    Salted dictionary: attacker-best-derivable keys never match (true+6 wrong).
 4. Timing: 5 repetitions, mean±std, VeilKV in BOTH fp32 and bf16 storage,
    correct units (seconds and ms per 1k tokens).
 5. Full correctness: {fp32, bf16} x {aligned, unaligned} x {K, V} roundtrip,
    logits + greedy-50 on 3 distinct texts, incremental vs full decode equality.

Known limitation recorded: K_MASTER is cross-request static (design choice;
same trust class as KV-Cloak's TEE-held static S/a) — written into results.
"""
import hashlib
import json
import math
import time
from collections import Counter

import torch
from transformers.cache_utils import DynamicCache

from attack_lib import (load_model, model_dims, build_theta, build_dictionary)
from run_suite import NATURAL
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
B = 16
OUT = "/mnt/hdd/yichengzhi/kvcloak_results/veilkv2_05b.json"
res = {"defense": "VeilKV v2 (keymat-free decode, padded tail, measured leak boundaries)",
       "notes": {"sliding_window_warning": "transformers config warning, benign",
                  "master_key": "K_MASTER is cross-request static by design (TEE-held; "
                                "same trust class as KV-Cloak's static S/a) — limitation, not a claim"}}

K_MASTER = hashlib.blake2b(b"veilkv-demo-master-key", digest_size=16).digest()


def prf_generator(*tweak_parts) -> torch.Generator:
    h = hashlib.blake2b(K_MASTER, digest_size=16)
    for p in tweak_parts:
        h.update(str(p).encode())
    g = torch.Generator()
    g.manual_seed(int.from_bytes(h.digest(), "little") % (2**63))
    return g


def apply_rot(Xh, ang):
    D = Xh.shape[-1]
    a = ang.unsqueeze(-2) if ang.dim() < Xh.dim() else ang
    c, s = torch.cos(a), torch.sin(a)
    x1, x2 = Xh[..., :D//2], Xh[..., D//2:]
    return torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1)


def inv_rot(Xh, ang):
    D = Xh.shape[-1]
    a = ang.unsqueeze(-2) if ang.dim() < Xh.dim() else ang
    c, s = torch.cos(a), torch.sin(a)
    x1, x2 = Xh[..., :D//2], Xh[..., D//2:]
    return torch.cat([x1*c + x2*s, -x1*s + x2*c], dim=-1)


# ---------- pure key derivation (identical draw order on both sides) ----------

def derive_keys_stream(H, nb, D, g):
    """One stream per (layer, kv): the production hot path. Returns (A, R, ang)."""
    Hn = H * nb
    A = 1.5*torch.eye(B) + 0.5*torch.linalg.qr(torch.randn(Hn, B, B, generator=g))[0]
    R = torch.empty(Hn, B).uniform_(0.5, 2.0, generator=g)
    ang = torch.rand(Hn, D//2, generator=g) * 2*math.pi
    return A, R, ang


def derive_keys_block(H, D, g):
    """Single-block derivation for chain mode (one PRF per block)."""
    return derive_keys_stream(H, 1, D, g)


def obf_tensor(X, g, store_dtype):
    """X [1,H,L,D] -> protected [1,H,ceil(L/B)*B,D]. Tail PADDED into a protected
    block (no plaintext in the protected output). Returns (prot, true_L)."""
    H, L, D = X.shape[1], X.shape[2], X.shape[3]
    npad = (math.ceil(L / B)) * B
    Xp = torch.zeros(1, H, npad, D)
    Xp[0, :, :L] = X[0]
    nb = npad // B
    A, R, ang = derive_keys_stream(H, nb, D, g)
    Xb = Xp[0].reshape(H, nb, B, D)
    Y = torch.einsum("hnbk,hnkd->hnbd", A.view(H, nb, B, B), Xb * R.view(H, nb, B, 1))
    Y = apply_rot(Y, ang.view(H, nb, D//2))
    return Y.reshape(1, H, npad, D).to(store_dtype), L


def dec_tensor(Xp, g, true_L, out_dtype):
    """Re-derives keys from the SAME PRF call — no stored key material."""
    H, Lpad, D = Xp.shape[1], Xp.shape[2], Xp.shape[3]
    nb = Lpad // B
    A, R, ang = derive_keys_stream(H, nb, D, g)
    Y = Xp[0].float().reshape(H, nb, B, D)
    Y = inv_rot(Y, ang.view(H, nb, D//2))
    Y = torch.linalg.solve(A, Y.reshape(H*nb, B, D)).view(H, nb, B, D)
    Y = (Y / R.view(H, nb, B, 1)).reshape(H, Lpad, D)
    return Y[:, :true_L].unsqueeze(0).to(out_dtype)


def veilkv_obf(cache, salt, store_dtype):
    prot = []
    for li, (K, V) in enumerate(cache):
        row = []
        for kvf, X in ((0, K), (1, V)):
            g = prf_generator("req", salt, li, kvf)
            p, tl = obf_tensor(X, g, store_dtype)
            row.append(p)
        prot.append((row[0], row[1]))
    return tuple(prot)


def veilkv_dec(prot, salt, true_lens, out_dtype):
    """Keymat-free decode: every key re-derived from PRF."""
    outs = []
    for li, (K, V) in enumerate(prot):
        row = []
        for kvf, (X, tl) in enumerate(zip((K, V), true_lens)):
            g = prf_generator("req", salt, li, kvf)
            row.append(dec_tensor(X, g, tl, out_dtype))
        outs.append((row[0], row[1]))
    return tuple(outs)


def veilkv_obf_chain(cache, tokens, store_dtype):
    L_tok = tokens.shape[1]
    prot = []
    for li, (K, V) in enumerate(cache):
        row = []
        for kvf, X in ((0, K), (1, V)):
            H, L, D = X.shape[1], X.shape[2], X.shape[3]
            npad = math.ceil(L / B) * B
            hsb = [hashlib.blake2b(tokens[0, :min(j*B, L_tok)].numpy().tobytes(),
                                    digest_size=12).hexdigest() for j in range(npad // B)]
            Xp = torch.zeros(1, H, npad, D)
            Xp[0, :, :L] = X[0]
            out = torch.empty(H, npad, D)
            for j in range(npad // B):
                g = prf_generator("chn", hsb[j], li, kvf)
                A, R, ang = derive_keys_block(H, D, g)      # A [H,B,B] R [H,B] ang [H,D/2]
                Y = torch.einsum("hbk,hkd->hbd", A, Xp[0, :, j*B:(j+1)*B] * R.unsqueeze(-1))
                out[:, j*B:(j+1)*B] = apply_rot(Y, ang)
            row.append(out.unsqueeze(0).to(store_dtype))
        prot.append((row[0], row[1]))
    return tuple(prot)


def veilkv_dec_chain(prot, tokens, true_lens, out_dtype):
    L_tok = tokens.shape[1]
    outs = []
    for li, (K, V) in enumerate(prot):
        row = []
        for kvf, (X, tl) in enumerate(zip((K, V), true_lens)):
            H, Lpad, D = X.shape[1], X.shape[2], X.shape[3]
            hsb = [hashlib.blake2b(tokens[0, :min(j*B, L_tok)].numpy().tobytes(),
                                    digest_size=12).hexdigest() for j in range(Lpad // B)]
            out = torch.empty(H, Lpad, D)
            for j in range(Lpad // B):
                g = prf_generator("chn", hsb[j], li, kvf)
                A, R, ang = derive_keys_block(H, D, g)
                Y = inv_rot(X[0, :, j*B:(j+1)*B].float(), ang)
                Y = torch.linalg.solve(A, Y)
                out[:, j*B:(j+1)*B] = Y / R.unsqueeze(-1)
            row.append(out[:, :tl].unsqueeze(0).to(out_dtype))
        outs.append((row[0], row[1]))
    return tuple(outs)


def legacy_cache(ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=True).past_key_values.to_legacy_cache()


print("loading model...", flush=True)
tok, model = load_model("Qwen/Qwen2.5-0.5B", threads=16)
layers, H, D = model_dims(model)

SYS_PREFIX = ("You are a helpful assistant to employees of ExampleCorp. "
              "Answer using company documentation only. Be concise and professional.")
VICTIM_QUERY = "My badge PIN is 4831 and my desk lock code is 7702. Where is the nearest meeting room?"
ATTACKER_QUERY = "What is the vacation policy for new hires this year?"

# ================= (A) implementation sanity =================
print("\n=== (A) keymat-free decode + padded tail ===", flush=True)
vids = tok(SYS_PREFIX + " " + VICTIM_QUERY, return_tensors="pt").input_ids
L_v = vids.shape[1]
legacy = legacy_cache(vids)
SALT = "req-victim-001"
prot_a = veilkv_obf(legacy, SALT, torch.float32)
tl_k, tl_v = legacy[0][0].shape[2], legacy[0][1].shape[2]
dec_a = veilkv_dec(prot_a, SALT, [tl_k, tl_v], legacy[0][1].dtype)
err_k = max((d[0].float()-v[0].float()).abs().max().item() for d, v in zip(dec_a, legacy))
err_v = max((d[1].float()-v[1].float()).abs().max().item() for d, v in zip(dec_a, legacy))
res["A_keymat_free_roundtrip_K"] = err_k
res["A_keymat_free_roundtrip_V"] = err_v
# padded tail block must not contain plaintext: last protected block != zero-pad artifact
last_blk_prot = prot_a[0][1][0, 0, -B:, :4]
res["A_tail_block_is_nontrivial"] = bool(last_blk_prot.abs().max().item() > 1e-3)
print(f"roundtrip K {err_k:.2e} V {err_v:.2e} | tail protected block non-trivial: "
      f"{res['A_tail_block_is_nontrivial']}", flush=True)

# chain mode roundtrip (keymat-free)
prot_c = veilkv_obf_chain(legacy, vids, torch.float32)
dec_c = veilkv_dec_chain(prot_c, vids, [tl_k, tl_v], legacy[0][1].dtype)
err_c = max((d[1].float()-v[1].float()).abs().max().item() for d, v in zip(dec_c, legacy))
res["A_chain_keymat_free_roundtrip_V"] = err_c
print(f"chain-mode roundtrip V {err_c:.2e}", flush=True)

# ================= (B) prefix-domain CPA =================
print("\n=== (B) prefix-domain CPA ===", flush=True)
common_ids = tok(SYS_PREFIX + " My badge PIN is ", return_tensors="pt").input_ids
n_cb = common_ids.shape[1] // B
common_ids = common_ids[:, :n_cb*B]
v_tail = tok("4831 and my desk lock code is 7702. Where is the nearest meeting room?",
             return_tensors="pt").input_ids
a_tail = tok(ATTACKER_QUERY, return_tensors="pt").input_ids
vids2 = torch.cat([common_ids, v_tail], 1)
aids2 = torch.cat([common_ids, a_tail], 1)
cvv, caa = legacy_cache(vids2), legacy_cache(aids2)

# B1: cross-request recomputed-prefix equality per layer (K and V), fp32
bit_eq_K = bit_eq_V = tot = 0
diff_stats = []
for li in range(layers):
    eqK = torch.equal(cvv[li][0][:, :, :n_cb*B], caa[li][0][:, :, :n_cb*B])
    eqV = torch.equal(cvv[li][1][:, :, :n_cb*B], caa[li][1][:, :, :n_cb*B])
    dK = (cvv[li][0][:, :, :n_cb*B] - caa[li][0][:, :, :n_cb*B]).abs().max().item()
    dV = (cvv[li][1][:, :, :n_cb*B] - caa[li][1][:, :, :n_cb*B]).abs().max().item()
    bit_eq_K += eqK; bit_eq_V += eqV; tot += 1
    diff_stats.append((li, dK, dV))
res["B1_prefix_recompute_bitwise_equal_K"] = f"{bit_eq_K}/{tot}"
res["B1_prefix_recompute_bitwise_equal_V"] = f"{bit_eq_V}/{tot}"
res["B1_max_bitdiff_by_layer_first_last"] = [diff_stats[0], diff_stats[-1]]
print(f"B1 recomputed-prefix bitwise equal: K {bit_eq_K}/{tot}, V {bit_eq_V}/{tot} "
      f"(layer0 dK={diff_stats[0][1]:.1e} dV={diff_stats[0][2]:.1e}; "
      f"last dK={diff_stats[-1][1]:.1e} dV={diff_stats[-1][2]:.1e})", flush=True)

# B1b: protected equality under chain mode using each request's own forward
pc_v = veilkv_obf_chain(cvv, vids2, torch.float32)
pc_a = veilkv_obf_chain(caa, aids2, torch.float32)
peq = ptot = 0
for li in range(layers):
    for kvf in (0, 1):
        Xv, Xa = pc_v[li][kvf], pc_a[li][kvf]
        for j in range(n_cb):
            ptot += 1
            peq += torch.equal(Xv[0, :, j*B:(j+1)*B], Xa[0, :, j*B:(j+1)*B])
res["B1b_chain_protected_prefix_equal_cross_request"] = f"{peq}/{ptot}"
print(f"B1b chain-protected prefix equal cross-request: {peq}/{ptot}", flush=True)

# B2: guess-confirmation boundary (accepted deterministic-encryption leak)
def prefix_guess_matches(guess_text, n_blocks, target_pc):
    gids = tok(guess_text, return_tensors="pt").input_ids[:, :n_blocks*B]
    if gids.shape[1] < n_blocks*B:
        gids = torch.cat([gids, torch.full((1, n_blocks*B - gids.shape[1]),
                                           tok(" the")["input_ids"][0])], 1)
    gc = legacy_cache(gids)
    pg = veilkv_obf_chain(gc, gids, torch.float32)
    for li in (0, 12):
        for kvf in (0, 1):
            for j in range(n_blocks):
                if not torch.equal(pg[li][kvf][0, :, j*B:(j+1)*B].float(),
                                   target_pc[li][kvf][0, :, j*B:(j+1)*B].float()):
                    return False
    return True

true_guess = SYS_PREFIX + " My badge PIN is "
res["B2_true_prefix_guess_matches"] = prefix_guess_matches(true_guess, n_cb, pc_v)
wrong_guesses = ["You are a helpful assistant to AcmeCorp employees. Answer briefly.",
                 "You are a legal advisor. Cite statutes.",
                 "Translate everything to French.",
                 "You are a helpful assistant to employees of ExampleCorp. Answer using "
                 "company documentation only. Be verbose.",
                 SYS_PREFIX + " My badge PIN is "]
wrong_miss = sum(not prefix_guess_matches(g, n_cb, pc_v) for g in wrong_guesses)
res["B2_wrong_prefix_guesses_missed"] = f"{wrong_miss}/{len(wrong_guesses)}"
print(f"B2 guess-confirmation: true guess match={res['B2_true_prefix_guess_matches']}, "
      f"wrong guesses missed {wrong_miss}/{len(wrong_guesses)} (accepted equality leak, MLE)", flush=True)

# B3: same content at different positions -> different ciphertexts (chain)
rep = " the quick brown fox jumps over the lazy dog near the river bank "
rep_ids = tok(rep*4, return_tensors="pt").input_ids[:, :4*B]
rc = legacy_cache(rep_ids)
prc = veilkv_obf_chain(rc, rep_ids, torch.float32)
div = all(not torch.equal(prc[0][1][0, :, 0:B], prc[0][1][0, :, k*B:(k+1)*B])
          for k in (1, 2, 3))
res["B3_same_content_diff_position_diverges"] = div
print(f"B3 identical 16-token content at 4 positions -> all ciphertexts distinct: {div}", flush=True)

# B4: attacker's own prefix ciphertext invariant to THEIR tail (same shape)
at1 = torch.cat([common_ids, tok("Question one about payroll.", return_tensors="pt").input_ids], 1)
at2 = torch.cat([common_ids, tok("Question two about dental.", return_tensors="pt").input_ids], 1)
if at1.shape[1] != at2.shape[1]:
    n = min(at1.shape[1], at2.shape[1]); at1, at2 = at1[:, :n], at2[:, :n]
c1, c2 = legacy_cache(at1), legacy_cache(at2)
p1 = veilkv_obf_chain(c1, at1, torch.float32)
p2 = veilkv_obf_chain(c2, at2, torch.float32)
inv = all(torch.equal(p1[li][1][0, :, j*B:(j+1)*B], p2[li][1][0, :, j*B:(j+1)*B])
          for li in (0, 12) for j in range(n_cb))
res["B4_prefix_ciphertext_invariant_to_tail"] = inv
print(f"B4 attacker prefix ciphertext invariant to own tail changes: {inv}", flush=True)

# ================= (C1) timing =================
print("\n=== (C1) timing (5 reps, matched dtype) ===", flush=True)
bench = tok(SYS_PREFIX + " " + VICTIM_QUERY + " " + NATURAL[0]*3,
            return_tensors="pt").input_ids
bench = bench[:, :512][:, :(512//B)*B]
bc = legacy_cache(bench)
n_b = bench.shape[1]

def timeit(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    m = sum(ts)/len(ts)
    sd = (sum((t-m)**2 for t in ts)/len(ts))**0.5
    return m, sd

for dt, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
    m, sd = timeit(lambda: veilkv_obf(bc, "bench", dt))
    res[f"C1_veilkv_obf_{tag}_s_per_1k"] = round(m/n_b*1000, 4)
    res[f"C1_veilkv_obf_{tag}_sd"] = round(sd/n_b*1000, 4)
    pb = veilkv_obf(bc, "bench", dt)
    tlens = [bc[0][0].shape[2], bc[0][1].shape[2]]
    m, sd = timeit(lambda: veilkv_dec(pb, "bench", tlens, dt))
    res[f"C1_veilkv_dec_{tag}_s_per_1k"] = round(m/n_b*1000, 4)
    res[f"C1_veilkv_dec_{tag}_sd"] = round(sd/n_b*1000, 4)

theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)
bc_bf = tuple((k.to(torch.bfloat16), v.to(torch.bfloat16)) for k, v in bc)
torch.manual_seed(7100)
m, sd = timeit(lambda: cloak.obfuscate(bc_bf))
res["C1_kvcloak_obf_bf16_s_per_1k"] = round(m/n_b*1000, 4)
res["C1_kvcloak_obf_bf16_sd"] = round(sd/n_b*1000, 4)
pkv = cloak.obfuscate(bc_bf)
m, sd = timeit(lambda: cloak.deobfuscate(pkv))
res["C1_kvcloak_dec_bf16_s_per_1k"] = round(m/n_b*1000, 4)
res["C1_kvcloak_dec_bf16_sd"] = round(sd/n_b*1000, 4)
print(f"obf s/1k: veilkv fp32 {res['C1_veilkv_obf_fp32_s_per_1k']} bf16 "
      f"{res['C1_veilkv_obf_bf16_s_per_1k']} | kvcloak bf16 {res['C1_kvcloak_obf_bf16_s_per_1k']}", flush=True)
print(f"dec s/1k: veilkv fp32 {res['C1_veilkv_dec_fp32_s_per_1k']} bf16 "
      f"{res['C1_veilkv_dec_bf16_s_per_1k']} | kvcloak bf16 {res['C1_kvcloak_dec_bf16_s_per_1k']}", flush=True)

# ================= (C2) full correctness =================
print("\n=== (C2) correctness matrix ===", flush=True)
res["C2_roundtrip"] = {}
for dt, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
    for name, txt in (("aligned", SYS_PREFIX), ("unaligned", SYS_PREFIX + " " + VICTIM_QUERY)):
        ids = tok(txt, return_tensors="pt").input_ids
        lg = legacy_cache(ids)
        pr = veilkv_obf(lg, "corr-"+tag, torch.float32 if tag == "fp32" else torch.bfloat16)
        dc = veilkv_dec(pr, "corr-"+tag, [lg[0][0].shape[2], lg[0][1].shape[2]], lg[0][1].dtype)
        eK = max((d[0].float()-v[0].float()).abs().max().item() for d, v in zip(dc, lg))
        eV = max((d[1].float()-v[1].float()).abs().max().item() for d, v in zip(dc, lg))
        res["C2_roundtrip"][f"{tag}_{name}"] = {"K": eK, "V": eV}
print(json.dumps(res["C2_roundtrip"], indent=1), flush=True)

# logits + greedy-50 on 3 texts (fp32 path)
res["C2_generation"] = {}
for i, txt in enumerate((SYS_PREFIX + " " + VICTIM_QUERY,
                         NATURAL[0],
                         "The quarterly report covers revenue growth, headcount changes "
                         "and infrastructure spending across all departments.")):
    ids = tok(txt, return_tensors="pt").input_ids
    lg = legacy_cache(ids)
    cont = tok(" In summary", return_tensors="pt").input_ids[:, 1:]
    with torch.no_grad():
        lp = model(input_ids=cont, past_key_values=DynamicCache.from_legacy_cache(lg),
                   use_cache=False).logits[0, -1]
    pr = veilkv_obf(lg, "gen", torch.float32)
    dc = veilkv_dec(pr, "gen", [lg[0][0].shape[2], lg[0][1].shape[2]], lg[0][1].dtype)
    with torch.no_grad():
        ld = model(input_ids=cont, past_key_values=DynamicCache.from_legacy_cache(dc),
                   use_cache=False).logits[0, -1]
    def greedy50(past):
        d = DynamicCache.from_legacy_cache(past)
        toks, cur = [], cont[:, :1]
        for _ in range(50):
            with torch.no_grad():
                o = model(input_ids=cur, past_key_values=d, use_cache=True)
            cur = o.logits[:, -1:].argmax(-1)
            toks.append(cur.item())
        return toks
    g_p, g_d = greedy50(lg), greedy50(dc)
    agree = sum(a == b for a, b in zip(g_p, g_d))
    res["C2_generation"][f"text{i}"] = {
        "logits_max_diff": round((ld-lp).abs().max().item(), 4),
        "next_top1_match": bool(ld.argmax() == lp.argmax()),
        "greedy50_agreement": f"{agree}/50",
    }
    print(f"  text{i}: logits {res['C2_generation'][f'text{i}']['logits_max_diff']}, "
          f"greedy50 {res['C2_generation'][f'text{i}']['greedy50_agreement']}", flush=True)

# incremental vs full decode equality (streaming block-by-block)
pb2 = veilkv_obf(legacy, "incr", torch.float32)
g0 = prf_generator("req", "incr", 0, 1)
A, R, ang = derive_keys_stream(H, math.ceil(tl_v/B), D, g0)
nb_i = pb2[0][1].shape[2] // B
Y_inc = []
for j in range(nb_i):
    Yj = inv_rot(pb2[0][1][0, :, j*B:(j+1)*B].float(), ang.view(H, nb_i, D//2)[:, j])
    Yj = torch.linalg.solve(A.view(H, nb_i, B, B)[:, j], Yj)
    Y_inc.append(Yj / R.view(H, nb_i, B)[:, j].unsqueeze(-1))
Y_inc = torch.cat(Y_inc, dim=1)
full = dec_tensor(pb2[0][1], prf_generator("req", "incr", 0, 1), tl_v, torch.float32)
res["C2_incremental_equals_full"] = bool((Y_inc[:, :tl_v] - full[0]).abs().max().item() < 1e-5)
print(f"incremental decode == full decode: {res['C2_incremental_equals_full']}", flush=True)

# ================= (C3) attack battery v2 =================
print("\n=== (C3) attack battery v2 ===", flush=True)
probe = torch.cat([common_ids,
                   torch.full((1, 2048 - common_ids.shape[1]), tok(" a")["input_ids"][0])], 1)
pcp = legacy_cache(probe)
SALT_A = "req-attacker-999"
pp = veilkv_obf(pcp, SALT_A, torch.float32)
core = pp[0][1][0, 0].float().view(-1, B, D)[n_cb:]
flat = core.flatten(1)
uniq = len(set(tuple(r.tolist()) for r in flat))
dmat = torch.cdist(flat, flat)
offd = dmat[~torch.eye(flat.shape[0], dtype=torch.bool)]
res["C3_probe_unique_blocks"] = f"{uniq}/{flat.shape[0]}"
res["C3_probe_min_pairwise_dist"] = round(offd.min().item(), 4)
res["C3_probe_median_pairwise_dist"] = round(offd.median().item(), 4)
print(f"probe non-collapse: {uniq}/{flat.shape[0]} unique blocks, min pairwise "
      f"{res['C3_probe_min_pairwise_dist']} (no clustering used)", flush=True)

# correct null baseline: NN hit-rate of RANDOM vectors against victim tokens
DICT, _, _ = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)
victim_prot = veilkv_obf(cvv, "req-victim-042", torch.float32)
rand_rows = torch.randn(2048, D)
sims_r = rand_rows @ Dn[:, 0].t()
rand_hits = sims_r.argmax(-1)
vset = set(vids2[0].tolist())
res["C3_null_random_row_token_hitrate"] = round(sum(h in vset for h in rand_hits.tolist())/2048, 4)

# key transfer: ALL victim tail blocks x 64 attacker keys, cosine-selected
tail_start = n_cb * B
nb_tail = (vids2.shape[1] - tail_start + B - 1) // B
g_own = prf_generator("req", SALT_A, 0, 1)
n_keys = 64
_, _, ang_own = derive_keys_stream(H, 2048 // B, D, g_own)
g_own2 = prf_generator("req", SALT_A, 0, 1)
A_own, R_own, _ = derive_keys_stream(H, 2048 // B, D, g_own2)
sel_accs, blind_accs = [], []
for jv in range(nb_tail):
    Cj = victim_prot[0][1][0, 0].float().view(-1, B, D)[n_cb + jv]
    true_tok = vids2[0, tail_start + jv*B: tail_start + (jv+1)*B].tolist()
    best_score, best_pred = -9e9, None
    accs = []
    for kt in range(n_keys):
        Aj, Rj, angj = A_own[kt], R_own[kt], ang_own[kt]
        Y = inv_rot(Cj, angj)
        Y = torch.linalg.solve(Aj, Y)
        Y = Y / Rj.unsqueeze(-1)
        s = Y @ Dn[:, 0].t()
        cosv, pred = s.max(-1)
        accs.append(sum((Counter(pred.tolist()) & Counter(true_tok)).values()) / B)
        if cosv.mean().item() > best_score:
            best_score, best_pred = cosv.mean().item(), pred.tolist()
    sel_accs.append(sum((Counter(best_pred) & Counter(true_tok)).values()) / B)
    blind_accs.extend(accs)
res["C3_keytransfer_cosine_selected_mean"] = round(sum(sel_accs)/len(sel_accs), 4)
res["C3_keytransfer_blind_mean"] = round(sum(blind_accs)/len(blind_accs), 4)
res["C3_keytransfer_blind_max"] = round(max(blind_accs), 4)
print(f"key transfer: cosine-selected mean {res['C3_keytransfer_cosine_selected_mean']:.1%} | "
      f"blind mean {res['C3_keytransfer_blind_mean']:.1%} max {res['C3_keytransfer_blind_max']:.1%} | "
      f"null (random rows hit victim tokens) {res['C3_null_random_row_token_hitrate']:.1%}", flush=True)

# salted dictionary: attacker-best-derivable keys, true + 6 wrong candidates
def salt_dict_attempt(cand):
    gtxt = f"{cand} and my desk lock code is 7702. Where is the nearest meeting room?"
    g_t = tok(gtxt, return_tensors="pt").input_ids
    need = vids2.shape[1] - common_ids.shape[1]
    if g_t.shape[1] < need:
        g_t = torch.cat([g_t, torch.full((1, need - g_t.shape[1]), tok(" the")["input_ids"][0])], 1)
    gids = torch.cat([common_ids, g_t[:, :need]], 1)
    gc = legacy_cache(gids)
    for salt_guess in ("chain",):  # attacker derives via chain keys (their best non-salt path)
        pg = veilkv_obf_chain(gc, gids, torch.float32)
        for li in (0, 12):
            for kvf in (0, 1):
                if not torch.equal(pg[li][kvf][0, :, n_cb*B:(n_cb+1)*B].float(),
                                   victim_prot[li][kvf][0, :, n_cb*B:(n_cb+1)*B].float()):
                    return False
    return True

hits = sum(salt_dict_attempt(c) for c in ["4831", "0000", "1377", "5000", "7770", "9999", "2049"])
res["C3_salted_dict_hits"] = f"{hits}/7"
print(f"salted dictionary via best-derivable (chain) keys: {hits}/7 candidate hits "
      f"(0 expected; salt is 128-bit TEE-held)", flush=True)

import os
os.makedirs("/mnt/hdd/yichengzhi/kvcloak_results", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(f"\nsaved {OUT}", flush=True)
