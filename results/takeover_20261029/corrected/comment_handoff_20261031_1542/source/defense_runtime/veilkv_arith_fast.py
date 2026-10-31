"""VeilKV-Plus arithmetic-profile fast path.

Optimizes ONLY the mixing application; keys/AAD/MAC/chain semantics identical to
exp_veilkv_plus (see constraints 2026-10-08):
- HKDF stream: identical call, identical length, identical byte layout.
- Angle quantization grid (2^16) and pair/radius extraction rules: identical.
- cos/sin evaluated with torch on the same quantized angles (grid unchanged;
  apply/inverse share the same tensors, so the exact-inverse property holds).
- Row mixing materialized as Q = G_g...G_1 (batched over all blocks*heads of a
  (layer,kv) row); inverse Qinv = G_1^T...G_g^T — reverse-transposed factor order
  preserved by construction. No linalg.solve. No torch.Generator.
"""
import hashlib
import math
import struct

import torch

import exp_veilkv_plus as V

B = V.B
GIVENS_STEPS = V.GIVENS_STEPS
_TWO_PI = 2.0 * math.pi


def derive_params_fast(k: bytes, Hn: int, D: int, g: int, variant: dict):
    """Vectorized drop-in for V.derive_mix_params (same HKDF stream & rules)."""
    need = Hn * (g * 4 + D + B)
    stream = V.hkdf(k, b"mix", need + 64)
    return parse_params(torch.frombuffer(bytearray(stream),
                                         dtype=torch.uint8).unsqueeze(0),
                        Hn, D, g, variant)


def parse_params(s, Hn, D, g, variant):
    """Parse one or MANY (stacked) HKDF streams. s: [U, need] uint8 where each
    row is an independent hkdf(k_u, b"mix", need+64) output (Hn=1 per row)."""
    need = g * 4 + D + B
    gseg = s[:, : g * 4].reshape(Hn, g, 4).int()
    i_idx = (gseg[..., 0] % B)
    k_idx = (gseg[..., 1] % B)
    k_idx = torch.where(k_idx == i_idx, (i_idx + 1) % B, k_idx)
    q = gseg[..., 2] * 256 + gseg[..., 3]                       # [Hn, g]
    ang = q.float() / 65536.0 * _TWO_PI
    cos_r, sin_r = torch.cos(ang), torch.sin(ang)
    o = g * 4
    cseg = s[:, o: o + (D // 2) * 2].reshape(Hn, D // 2, 2).int()
    qc = cseg[..., 0] * 256 + cseg[..., 1]
    angc = qc.float() / 65536.0 * _TWO_PI
    cos_c, sin_c = torch.cos(angc), torch.sin(angc)
    o2 = o + (D // 2) * 2
    rseg = s[:, o2: o2 + B].reshape(Hn, B).float()
    radii = 0.75 + rseg / 255.0 * 0.75
    if variant.get("orthogonal_only"):
        radii = torch.ones_like(radii)
    if variant.get("radius_only"):
        i_idx = torch.zeros_like(i_idx)
        k_idx = torch.zeros_like(k_idx)
        cos_r = torch.ones_like(cos_r)
        sin_r = torch.zeros_like(sin_r)
    return dict(i_idx=i_idx.long(), k_idx=k_idx.long(), cos_r=cos_r, sin_r=sin_r,
                cos_c=cos_c, sin_c=sin_c, radii=radii)


def _givens_matrices(i_idx, k_idx, cos_r, sin_r, t):
    """G_t for every row of the batch: [Hn, B, B]."""
    Hn = i_idx.shape[0]
    G = torch.eye(B, dtype=cos_r.dtype).expand(Hn, B, B).clone()
    ar = torch.arange(Hn)
    c, s = cos_r[:, t], sin_r[:, t]
    G[ar, i_idx[:, t], i_idx[:, t]] = c
    G[ar, i_idx[:, t], k_idx[:, t]] = -s
    G[ar, k_idx[:, t], i_idx[:, t]] = s
    G[ar, k_idx[:, t], k_idx[:, t]] = c
    return G


def compose_Q(P):
    """Q = G_g ... G_1 for ALL units batched: 48 bmm steps total (was per-unit)."""
    Hn, g = P["i_idx"].shape
    Q = torch.eye(B, dtype=P["cos_r"].dtype).expand(Hn, B, B).clone()
    for t in range(g):
        Q = torch.bmm(_givens_matrices(P["i_idx"], P["k_idx"], P["cos_r"],
                                       P["sin_r"], t), Q)
    return Q


def compose_Qinv(P):
    """Qinv = G_1^T @ ... @ G_g^T for ALL units batched (no solve)."""
    Hn, g = P["i_idx"].shape
    Qinv = torch.eye(B, dtype=P["cos_r"].dtype).expand(Hn, B, B).clone()
    for t in range(g - 1, -1, -1):
        G = _givens_matrices(P["i_idx"], P["k_idx"], P["cos_r"], P["sin_r"], t)
        Qinv = torch.bmm(G.transpose(1, 2), Qinv)
    return Qinv


def _col_rot_batch(X, cos_c, sin_c):
    """X [Hn, B, D]; cos_c/sin_c [Hn, D/2]."""
    D = X.shape[-1]
    c = cos_c.unsqueeze(1)
    s = sin_c.unsqueeze(1)
    x1, x2 = X[..., :D // 2], X[..., D // 2:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def _col_rot_inv_batch(X, cos_c, sin_c):
    D = X.shape[-1]
    c = cos_c.unsqueeze(1)
    s = sin_c.unsqueeze(1)
    x1, x2 = X[..., :D // 2], X[..., D // 2:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], dim=-1)


# ---------------- fast arithmetic protect/recover (mirrors V semantics) ----------------

def _unit_streams(keys, li, kvf, H, D, bsz):
    """All (block, head) HKDF streams for one (li,kvf) row, stacked [nb*H, need]."""
    need = GIVENS_STEPS * 4 + D + B
    outs = []
    nb = len(keys)
    for j in range(nb):
        for hh in range(H):
            k_mix = V.hkdf(keys[j], b"mix-key" + struct.pack(">QQQ", li, kvf, hh)
                           + (b"\x00" if bsz == 1 else struct.pack(">Q", 0)), 32)
            outs.append(torch.frombuffer(
                bytearray(V.hkdf(k_mix, b"mix", need + 64)), dtype=torch.uint8))
    return torch.stack(outs)


def _keys_for_row(ctx, prot, li, kvf, nb, tdigs):
    keys, arts = [], []
    npb = ctx["n_prefix_blocks"]
    for j in range(nb):
        sclass = "public" if j < npb else "private"
        if sclass == "public" and not ctx["variant"].get("random_prefix"):
            art, key = V.prefix_key(ctx["model_dig"], ctx["tok_dig"], ctx["rope"],
                                    tdigs[j], j, li, kvf, "public",
                                    prot.dtype_name, ctx["epoch_id"])
        else:
            art = None
            principal = ctx["principal"] if not ctx["variant"].get("no_salt") \
                else ctx["principal"] + b"\x00noSalt"
            nonce = ctx["nonce"] if not ctx["variant"].get("no_salt") else b""
            key = V.tail_key(principal, nonce, j, li, 0, kvf, ctx["epoch_id"])
        keys.append(key)
        arts.append(art)
    return keys, arts


def protect_cache_arith_fast(cache, ctx, chain_seed=None):
    """Arithmetic-only fast protect. Same keys/AAD/MAC/chain semantics as
    V.protect_cache; mixing batched over all (block, head) units per (layer,kv)."""
    prof = ctx["profile"]
    assert prof == "arithmetic", "fast path is arithmetic-only"
    dt = ctx["storage_dtype"]
    prot = V.Protected(ctx["epoch_id"], prof, str(dt).split(".")[-1])
    npb = ctx["n_prefix_blocks"]
    tokens = ctx["tokens"]
    L = cache[0][1].shape[2]
    npad = math.ceil(L / B) * B
    nb = npad // B
    bsz = cache[0][1].shape[0]
    assert bsz == 1, "fast path validated for batch=1"
    tdigs = []
    for j in range(nb):
        seg = tokens[:, :min(j * B, tokens.shape[1])].numpy().tobytes() \
            if tokens.shape[1] > 0 else b""
        tdigs.append(hashlib.sha256(b"tok" + seg).digest())
    for li, (K, Vc) in enumerate(cache):
        for kvf, X in ((0, K), (1, Vc)):
            H, D = X.shape[1], X.shape[3]
            Xp = torch.zeros(bsz, H, npad, D)
            Xp[:, :, :L] = X.to(torch.float32)
            keys, arts = _keys_for_row(ctx, prot, li, kvf, nb, tdigs)
            S = _unit_streams(keys, li, kvf, H, D, bsz)          # [nb*H, need]
            Pb = parse_params(S, nb * H, D, GIVENS_STEPS, ctx["variant"])
            Qb = compose_Q(Pb)                                    # [nb*H, B, B]
            Xb = Xp[0].reshape(H, nb, B, D).permute(1, 0, 2, 3).reshape(nb * H, B, D)
            Y = torch.bmm(Qb, Xb) * Pb["radii"].unsqueeze(-1)
            Y = _col_rot_batch(Y, Pb["cos_c"], Pb["sin_c"])
            Yall = Y.reshape(nb, H, B, D).to(dt)                 # ONE batched cast
            ball = V._tensor_bytes(Yall)                          # ONE byte dump (j-major)
            bsize_blk = (H * B * D) * (2 if dt == torch.bfloat16 else 4)
            row = []
            prev_tag = b"\x00" * 16
            for j in range(nb):
                if j == npb and chain_seed and (li, kvf) in chain_seed:
                    prev_tag = chain_seed[(li, kvf)]
                sclass = "public" if j < npb else "private"
                aad = prot.aad(li, kvf, j, L, sclass, artifact_id=arts[j],
                               shape_d=D,
                               is_last_tail=(sclass == "private" and j == nb - 1))
                payload = Yall[j].unsqueeze(0)                    # zero-copy view
                pbytes = ball[j * bsize_blk:(j + 1) * bsize_blk]
                tag = V.mac_only(keys[j], aad, pbytes)
                row.append(dict(kind=sclass, artifact_id=arts[j], j=j, ct=payload,
                                tag=tag, prev_tag=prev_tag))
                prev_tag = tag
            prot.blocks[(li, kvf)] = row
    return prot


def recover_cache_arith_fast(prot, ctx, chain_seed=None):
    """Arithmetic-only fast recover: identical verification semantics to
    V.recover_cache; demix batched over all (block, head) units per (layer,kv)."""
    assert prot.profile == "arithmetic"
    out = []
    npb = ctx["n_prefix_blocks"]
    tokens = ctx["tokens"]
    nb = min(len(row) for row in prot.blocks.values())
    L = ctx["true_len"]
    npad = nb * B
    if npad < L:
        raise V.TamperError("block count inconsistent with true_length (truncation)")
    bsz = ctx.get("bsz", 1)
    assert bsz == 1, "fast path validated for batch=1"
    H, D = ctx["H"], ctx["D"]
    tdigs = []
    for j in range(nb):
        seg = tokens[:, :min(j * B, tokens.shape[1])].numpy().tobytes() \
            if tokens.shape[1] > 0 else b""
        tdigs.append(hashlib.sha256(b"tok" + seg).digest())
    for li in range(len(prot.blocks) // 2):
        rowout = []
        for kvf in (0, 1):
            row = prot.blocks[(li, kvf)]
            prev_tag = (chain_seed or {}).get((li, kvf), b"\x00" * 16)
            keys, _arts = _keys_for_row(ctx, prot, li, kvf, nb, tdigs)
            S = _unit_streams(keys, li, kvf, H, D, bsz)
            Pb = parse_params(S, nb * H, D, GIVENS_STEPS, ctx["variant"])
            Qinv = compose_Qinv(Pb)                              # [nb*H, B, B]
            for blk in row:
                if blk["prev_tag"] != prev_tag:
                    raise V.TamperError("chain break at block %d" % blk["j"])
                j = blk["j"]
                sclass = "public" if j < npb else "private"
                if blk["kind"] != sclass:
                    raise V.TamperError("policy class mismatch")
                if sclass == "public" and not ctx["variant"].get("random_prefix"):
                    if blk["artifact_id"] != _arts[j]:
                        raise V.TamperError("artifact substitution")
                    aad = prot.aad(li, kvf, j, L, sclass, artifact_id=_arts[j],
                                   shape_d=D, is_last_tail=False)
                else:
                    aad = prot.aad(li, kvf, j, L, sclass,
                                   is_last_tail=(j == len(row) - 1))
                prev_tag = blk["tag"]
            Ct = torch.stack([blk["ct"] for blk in row]).squeeze(2)  # [nb,H,B,D]
            ball = V._tensor_bytes(Ct)
            dtb = Ct.dtype
            bsize_blk = (H * B * D) * (2 if dtb == torch.bfloat16 else 4)
            for i_, blk in enumerate(row):
                j = blk["j"]
                sclass2 = "public" if j < npb else "private"
                if sclass2 == "public" and not ctx["variant"].get("random_prefix"):
                    aad2 = prot.aad(li, kvf, j, L, sclass2, artifact_id=_arts[j],
                                    shape_d=D, is_last_tail=False)
                else:
                    aad2 = prot.aad(li, kvf, j, L, sclass2,
                                    is_last_tail=(j == len(row) - 1))
                pb2 = ball[i_ * bsize_blk:(i_ + 1) * bsize_blk]
                if not V.hmac.compare_digest(V.mac_only(keys[j], aad2, pb2),
                                             blk["tag"]):
                    raise V.TamperError("MAC verification failed")
            Cb = Ct.reshape(nb * H, B, D)
            Y = _col_rot_inv_batch(Cb, Pb["cos_c"], Pb["sin_c"])
            Y = Y / Pb["radii"].unsqueeze(-1)
            Y = torch.bmm(Qinv, Y)
            Xfull = Y.reshape(nb, H, B, D).permute(1, 0, 2, 3).reshape(1, H, npad, D)
            rowout.append(Xfull[:, :, :L])
        out.append((rowout[0], rowout[1]))
    return tuple(out)
