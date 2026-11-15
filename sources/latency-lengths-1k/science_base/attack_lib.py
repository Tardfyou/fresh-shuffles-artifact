"""Attack library for KV-Cloak (official commit 6b40f36, unmodified).

Everything here is the experimentally-validated pipeline from exp1-4:
  probe -> obfuscate -> cluster states -> recover {a,u,w,S cols,M angles}
  -> demix unseen blocks -> dictionary nearest-neighbour token recovery.

Attacker knowledge: public model weights, protected caches of own requests,
their own probe text. Never touches plaintext victim caches (those are used
only as instrumentation for ground-truth scoring).
"""
import math
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvcloak import KVCloak
from get_kvcloak_config import get_kvcloak_config

B_DEFAULT = 16


# ---------------------------------------------------------------- model helpers
def load_model(name, threads=16):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return tok, model


def model_dims(model):
    cfg = model.config
    layers = cfg.num_hidden_layers
    q = getattr(cfg, "num_attention_heads")
    kv = getattr(cfg, "num_key_value_heads", None) or q
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // q
    return layers, kv, head_dim


def run_est_len(raw, target, max_positions):
    return False


def probe_ids(tok, target=1024, chars="abxyz1e0", max_positions=None):
    """RAW-ENTRY probe: return the tokenizer's exact output (BOS/prefix kept),
    plus the token index where a >=target single-token repeat run begins.
    Bounded search: chars that fragment at the first size are skipped.
    Returns (ids, repeat_start_idx, char)."""
    if max_positions is not None:
        target = min(target, max_positions - 64)
    for ch in chars:
        tries = 0
        n = max(4096, target)
        while tries < 5 and n <= 262144:
            tries += 1
            raw = tok(ch * n, return_tensors="pt").input_ids[0].tolist()
            if max_positions is not None and len(raw) > max_positions - 8:
                if run_est_len(raw, target, max_positions):
                    pass
                n2 = int(n * (max_positions - 16) / len(raw))
                if n2 >= 256 and n2 < n:
                    n = n2
                    continue
                break
            t = raw[-1]
            k = len(raw) - 1
            while k >= 0 and raw[k] == t:
                k -= 1
            prefix_len = k + 1
            runlen = len(raw) - prefix_len
            if runlen >= target and len(set(raw[:prefix_len])) <= 3:
                return torch.tensor([raw]), prefix_len, ch
            if len(set(raw)) > 6 or runlen < 256:
                break  # fragments; skip char
            if max_positions is not None and len(raw) >= max_positions - 8:
                break  # cannot grow within context
            n = min(int(n * 2.5), 262144)
    # fallback: direct token-id repeat (NOT raw-entry; logged as such)
    ids = tok(chars[0] * 8192, return_tensors="pt").input_ids
    t = ids[0, -1].item()
    return torch.tensor([[t] * (target // 16 * 16)]), 0, chars[0] + "(direct-id)"


def build_theta(model, tok, texts):
    """Per-layer/type/head max|cache| over natural texts (mirrors repo theta json)."""
    maxv = {}
    with torch.no_grad():
        for s in texts:
            pv = model(input_ids=tok(s, return_tensors="pt").input_ids, use_cache=True).past_key_values
            for li in range(len(pv)):
                K, V = pv[li]
                e = maxv.setdefault(li, {"key_max_values": torch.full((K.shape[1],), -1.0),
                                         "value_max_values": torch.full((V.shape[1],), -1.0)})
                e["key_max_values"] = torch.maximum(e["key_max_values"], K[0].abs().amax(dim=(1, 2)))
                e["value_max_values"] = torch.maximum(e["value_max_values"], V[0].abs().amax(dim=(1, 2)))
    return {f"layer_{li}": {k: v.tolist() for k, v in d.items()} for li, d in maxv.items()}


# ---------------------------------------------------------------- dictionary
def build_dictionary(model, H, D):
    """v(t) = v_proj( norm(emb(t)) ) + bias, per kv head. Norm auto-detected by
    comparing against a captured v_proj input; verified by caller."""
    layer0 = model.model.layers[0]
    emb = model.get_input_embeddings().weight
    v_proj = getattr(layer0.self_attn, "v_proj", None)
    qkv_proj = getattr(layer0.self_attn, "qkv_proj", None)  # Phi-3 fused
    with torch.no_grad():
        cand_norms = []
        for attr in ("input_layernorm", "layer_norm", "self_attn.attention_norm"):
            if hasattr(layer0, attr):
                cand_norms.append((attr, getattr(layer0, attr)))
        cand_norms.append(("identity", None))
        # capture reference: run one token through candidate pipelines and compare
        # against a hook-captured v_proj input for a short text
        # (we instead verify at cache level in verify_dictionary; here choose by
        #  trying all and scoring on 64 embeddings via a captured input)
        # capture input to v_proj via hook on the probe of 64 distinct-ish tokens
        captured = {}
        target_mod = v_proj if v_proj is not None else qkv_proj
        def hook(mod, inp, out):
            captured["x"] = inp[0].detach()
        h = target_mod.register_forward_hook(hook)
        ids = torch.arange(64).unsqueeze(0) % (emb.shape[0] - 1) + 1
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        h.remove()
        if v_proj is None:  # slice fused weights: q | k | v
            qn = model.config.num_attention_heads
            hidden = model.config.hidden_size
            hd = hidden // qn
            Wv = qkv_proj.weight[2 * qn * hd: 3 * qn * hd, :]
            bv = qkv_proj.bias[2 * qn * hd: 3 * qn * hd] if getattr(qkv_proj, "bias", None) is not None else None
        else:
            Wv, bv = v_proj.weight, getattr(v_proj, "bias", None)
        x_cap = captured["x"][0]  # (L, hidden) at layer 0 == norm(emb)
        best = None
        for attr, norm in cand_norms:
            with torch.no_grad():
                e = emb[ids[0]]
                xn = norm(e) if norm is not None else e
                err = (xn - x_cap).abs().max().item()
            if best is None or err < best[0]:
                best = (err, attr, norm)
        err, attr, norm = best
        rows = []
        bias = bv
        for i in range(0, emb.shape[0], 8192):
            e = emb[i:i + 8192]
            xn = norm(e) if norm is not None else e
            r = xn @ Wv.T
            if bias is not None:
                r = r + bias
            rows.append(r)
        DICT = torch.cat(rows).view(-1, H, D)
    return DICT, attr, err


def verify_dictionary(DICT, plain_v0, ids):
    """Instrumentation: dictionary rows should equal plaintext layer-0 V rows."""
    Dn = torch.nn.functional.normalize(DICT, dim=-1)
    ok = 0
    for h in range(plain_v0.shape[0]):
        p = torch.nn.functional.normalize(plain_v0[h], dim=-1)
        pred = (p @ Dn[:, h].T).argmax(1)
        ok += (pred == ids[0]).sum().item()
    return ok / (plain_v0.shape[0] * plain_v0.shape[1])


# ---------------------------------------------------------------- core attack
def rot_strided(angles, D):
    D2 = D // 2
    M = torch.zeros(D, D)
    for i, t in enumerate(angles.tolist()):
        c, s = math.cos(t), math.sin(t)
        M[i, i] = c
        M[i + D2, i + D2] = c
        M[i, i + D2] = s
        M[i + D2, i] = -s
    return M


def cluster_states(blocks):
    """blocks (NB, b, d) -> (n_states, centroids, intra, inter)."""
    flat = blocks.flatten(1)
    dist = torch.cdist(flat, flat)
    off = dist[~torch.eye(len(blocks), dtype=torch.bool)]
    pos = off[off > 0]
    if len(pos) == 0:
        thr = 1.0
    else:
        thr = min((pos.min() * pos.max()).sqrt(), pos.max() * 0.5)
    lab = torch.full((len(blocks),), -1)
    cid = 0
    for i in range(len(blocks)):
        if lab[i] >= 0:
            continue
        comp = ((dist[i] <= thr) & (lab < 0)).nonzero().flatten()
        if len(comp) == 0:
            comp = torch.tensor([i])
        lab[comp] = cid
        cid += 1
    cents = torch.stack([blocks[lab == c].mean(0) for c in range(cid)])
    intra = off[off <= thr].max().item() if (off <= thr).any() else 0.0
    inter = off[off > thr].min().item() if (off > thr).any() else float("inf")
    return cid, cents, intra, inter, lab


def recover_secrets(cents, plain_v0_head, b):
    """cents (b, b, d): one centroid per state. plain_v0_head (b, d): repeated-token
    plaintext block (known to attacker -- it is their own probe).
    Returns dict with a_hat, uhat, w_hat, s_hat, M_hat and diagnostics."""
    d = cents.shape[-1]
    vs, sigs = [], []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            U_, S_, Vh_ = torch.linalg.svd(cents[i] - cents[j])
            v1 = Vh_[0]
            if vs and torch.dot(v1, vs[0]) < 0:
                v1 = -v1
            vs.append(v1)
            sigs.append(S_[0].item() / math.sqrt(2.0))
    ahat = torch.stack(vs).mean(0)
    ahat = ahat * (1 if ahat.sum() > 0 else -1)
    pos_frac = (ahat > 0).float().mean().item()
    alpha = sum(sigs) / len(sigs)
    a_hat = ahat * alpha

    P = torch.eye(d) - torch.outer(ahat, ahat)
    lefts = []
    for c in cents:
        U_, S_, _ = torch.linalg.svd(c @ P)
        lefts.append(U_[:, 0] * S_[0].sqrt())
    uhat = torch.stack(lefts).mean(0)
    uhat = uhat / uhat.norm() * math.sqrt(b)

    v_row = plain_v0_head[0]

    def ws(u):
        w = ((u @ cents) - a_hat) / b
        w = w.mean(0)
        s = ((cents - torch.outer(u, w)) @ a_hat) / (a_hat ** 2).sum()
        return w, s

    w_hat, s_hat = ws(uhat)
    # sign of u by ||w|| ~= ||v|| (orthogonal M preserves norm)
    for sgn in (1.0, -1.0):
        cand_u = uhat * sgn
        cand_w, cand_s = ws(cand_u)
        if sgn == 1.0 or abs(cand_w.norm() - v_row.norm()) < abs(w_hat.norm() - v_row.norm()):
            uhat, w_hat, s_hat = cand_u, cand_w, cand_s

    # M angles from single known (v, w) pair, strided layout
    D2 = d // 2
    angles = torch.zeros(D2)
    used = 0
    for i in range(D2):
        va, vb = v_row[i].item(), v_row[i + D2].item()
        wa, wb = w_hat[i].item(), w_hat[i + D2].item()
        if va * va + vb * vb < 1e-8:
            continue
        angles[i] = math.atan2(wb, wa) - math.atan2(vb, va)
        used += 1
    M_hat = rot_strided(angles, d)
    return {"a_hat": a_hat, "u": uhat, "w": w_hat, "s": s_hat,
            "M": M_hat, "pos_frac": pos_frac, "alpha": alpha, "angle_pairs_used": used}


def attack_block(C, s_hat, a_hat, M_hat, Dn_h):
    """Demix one protected block; returns predicted token ids (list of b)."""
    best = None
    for j in range(s_hat.shape[0]):
        X = torch.nn.functional.normalize(
            s_hat @ (C - torch.outer(s_hat[j], a_hat)) @ M_hat.T, dim=-1)
        sims = X @ Dn_h.T
        sc = sims.max(1).values.sum().item()
        if best is None or sc > best[0]:
            best = (sc, sims.argmax(1).tolist())
    return best[1]


def ground_truth_errors(sec, conf_true, plain_v0_head):
    S_true = conf_true["S"].float()
    a_true = conf_true["a"].float()
    u_true = S_true @ torch.ones(S_true.shape[0])
    M_true = rot_strided(conf_true["M_angles"], a_true.numel())
    w_true = M_true.T @ plain_v0_head[0]
    cos = lambda x, y: (torch.dot(x, y) / (x.norm() * y.norm())).item()
    sc = torch.nn.functional.cosine_similarity(
        sec["s"].unsqueeze(1), S_true.t().unsqueeze(0), dim=2)
    return {
        "cos_a": abs(cos(sec["a_hat"], a_true)),
        "a_norm_relerr": abs(sec["alpha"] - a_true.norm()).item() / a_true.norm().item(),
        "cos_u": abs(cos(sec["u"], u_true)),
        "w_relerr": (sec["w"] - w_true).norm().item() / max(w_true.norm().item(), 1e-12),
        "S_min_signed_cos": sc.max(1).values.min().item(),
        "S_bijection": len(set(sc.argmax(1).tolist())) == sec["s"].shape[0],
    }
