"""Defensive codec extracted from the retained prototype; function bodies unchanged."""
import argparse

import hashlib

import hmac

import io

import json

import math

import os

import struct

import time

from collections import Counter

import torch

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

B = 16

HDD = "results/veilkv_plus"

OUTD = os.path.join(HDD, "results")

COSTD = os.path.join(HDD, "costs")

for d in (OUTD, COSTD):
    os.makedirs(d, exist_ok=True)

RES_PATH = os.path.join(OUTD, "veilkv_plus.json")

SAMPLES_PATH = os.path.join(OUTD, "veilkv_plus_samples.jsonl")

COSTS_PATH = os.path.join(COSTD, "veilkv_plus_costs.jsonl")

K_MASTER = hashlib.sha3_256(b"veilkv-plus-master").digest()

EPOCH = 3

def hkdf(ikm: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(b"\x00" * 32, ikm, hashlib.sha256).digest()
    okm, t, i = b"", b"", 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]

def aes_ctr(key: bytes, iv: bytes, data: bytes) -> bytes:
    c = Cipher(algorithms.AES(key), modes.CTR(iv))
    e = c.encryptor()
    return e.update(data) + e.finalize()

class TamperError(Exception):
    pass

def det_auth_enc(key: bytes, aad: bytes, plaintext: bytes):
    """AES-CTR + HMAC EtM; callers must enforce unique private request nonces."""
    stream = hkdf(key, b"split" + aad, 64)
    ek, mk, iv = stream[:16], stream[16:48], stream[48:64]
    ct = aes_ctr(ek, iv, plaintext)
    tag = hmac.new(mk, ct + b"|" + aad, hashlib.sha256).digest()[:16]
    return ct, tag

def mac_only(key: bytes, aad: bytes, data: bytes) -> bytes:
    """MAC over raw payload bytes (arithmetic profile stores mixed tensor, not AES ct)."""
    mk = hkdf(key, b"split" + aad, 64)[16:48]
    return hmac.new(mk, data + b"|" + aad, hashlib.sha256).digest()[:16]

def det_auth_dec(key: bytes, aad: bytes, ct: bytes, tag: bytes) -> bytes:
    stream = hkdf(key, b"split" + aad, 64)
    mk = stream[16:48]
    exp = hmac.new(mk, ct + b"|" + aad, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(exp, tag):
        raise TamperError("MAC verification failed")
    return aes_ctr(stream[:16], stream[48:64], ct)

def k_epoch(epoch_id: int) -> bytes:
    return hkdf(K_MASTER, b"epoch" + struct.pack(">Q", epoch_id))

def prefix_key(model_dig: bytes, tok_dig: bytes, rope: bytes, token_dig: bytes,
               block: int, layer: int, kvf: int, sclass: str, dtype: str,
               epoch_id: int) -> tuple:
    """v4: artifact identity includes layer & kv_type so K/V blocks are distinct
    artifacts (prevents same-key/same-AAD keystream reuse across K and V).
    artifact_id is PUBLIC (computable from public fields). K_art stays TEE-side."""
    artifact_id = hmac.new(b"artifact", model_dig + tok_dig + rope + token_dig +
                           struct.pack(">QQQQ", block, layer, kvf,
                                       1 if sclass == "public" else 0) +
                           dtype.encode(), hashlib.sha256).digest()[:16]
    k_art = hkdf(k_epoch(epoch_id), b"art" + artifact_id)
    return artifact_id, k_art

def tail_key(principal: bytes, nonce: bytes, block: int, layer: int, head: int,
             kvf: int, epoch_id: int) -> bytes:
    from corrected.freshness import scope
    info = scope(b"tail", principal, nonce, struct.pack(">QQQQ", block, layer, head, kvf))
    return hkdf(k_epoch(epoch_id), info)

GIVENS_STEPS = 3 * B

QANG = 1 << 16

def _qangle(byte2: bytes) -> float:
    return (byte2[0] << 8 | byte2[1]) / QANG * 2.0 * math.pi

def derive_mix_params(k: bytes, Hn: int, D: int, g: int, variant: dict):
    """From key bytes -> quantized rotation params. Deterministic across devices
    given the quantization grid (engineering premise; recorded as limitation)."""
    need = Hn * (g * 4 + D + B)                 # per step: pair(2B)+angle(2B); col angles D/2*2B; radii B
    stream = hkdf(k, b"mix", need + 64)
    pairs_i = torch.empty(Hn, g, dtype=torch.long)
    pairs_k = torch.empty(Hn, g, dtype=torch.long)
    cos_r = torch.empty(Hn, g)
    sin_r = torch.empty(Hn, g)
    o = 0
    for hn in range(Hn):
        for t in range(g):
            i = stream[o] % B
            kk = stream[o + 1] % B
            if kk == i:
                kk = (i + 1) % B
            pairs_i[hn, t], pairs_k[hn, t] = i, kk
            a = _qangle(stream[o + 2:o + 4])
            cos_r[hn, t], sin_r[hn, t] = math.cos(a), math.sin(a)
            o += 4
    col_ang = torch.empty(Hn, D // 2)
    cos_c = torch.empty(Hn, D // 2)
    sin_c = torch.empty(Hn, D // 2)
    for hn in range(Hn):
        for c in range(D // 2):
            a = _qangle(stream[o:o + 2])
            cos_c[hn, c], sin_c[hn, c] = math.cos(a), math.sin(a)
            o += 2
    radii = torch.empty(Hn, B)
    for hn in range(Hn):
        for rI in range(B):
            radii[hn, rI] = 0.75 + (stream[o] / 255.0) * 0.75          # [0.75,1.5], cond<=2
            o += 1
    if variant.get("orthogonal_only"):
        radii = torch.ones_like(radii)
    if variant.get("radius_only"):                                    # no row mixing
        pairs_i = pairs_i % 1  # all zeros
        pairs_k = (pairs_i + 1) % 1
        cos_r = torch.ones_like(cos_r)
        sin_r = torch.zeros_like(sin_r)
    return dict(pairs_i=pairs_i, pairs_k=pairs_k, cos_r=cos_r, sin_r=sin_r,
                cos_c=cos_c, sin_c=sin_c, radii=radii)

def _apply_row_rot(X: torch.Tensor, P: dict) -> torch.Tensor:
    """X [Hn, B, D]; apply g Givens steps then row radii. Vectorized over Hn."""
    Hn = X.shape[0]
    X = X.clone()
    g = P["pairs_i"].shape[1]
    D = X.shape[-1]
    for t in range(g):
        i = P["pairs_i"][:, t]                     # [Hn]
        k = P["pairs_k"][:, t]
        c = P["cos_r"][:, t].view(Hn, 1, 1)
        s = P["sin_r"][:, t].view(Hn, 1, 1)
        gi = i.view(Hn, 1, 1).expand(Hn, 1, D)
        gk = k.view(Hn, 1, 1).expand(Hn, 1, D)
        xi = X.gather(1, gi)
        xk = X.gather(1, gk)
        X = X.scatter(1, gi, c * xi - s * xk)
        X = X.scatter(1, gk, s * xi + c * xk)
    return X * P["radii"].unsqueeze(-1)

def _inv_row_rot(X: torch.Tensor, P: dict) -> torch.Tensor:
    """Explicit inverse: divide radii, reverse rotations with transposed angles."""
    Hn = X.shape[0]
    g = P["pairs_i"].shape[1]
    D = X.shape[-1]
    X = X / P["radii"].unsqueeze(-1)
    X = X.clone()
    for t in range(g - 1, -1, -1):
        i = P["pairs_i"][:, t]
        k = P["pairs_k"][:, t]
        c = P["cos_r"][:, t].view(Hn, 1, 1)
        s = P["sin_r"][:, t].view(Hn, 1, 1)
        gi = i.view(Hn, 1, 1).expand(Hn, 1, D)
        gk = k.view(Hn, 1, 1).expand(Hn, 1, D)
        xi = X.gather(1, gi)
        xk = X.gather(1, gk)
        X = X.scatter(1, gi, c * xi + s * xk)
        X = X.scatter(1, gk, -s * xi + c * xk)
    return X

def _apply_col_rot(X: torch.Tensor, P: dict) -> torch.Tensor:
    Hn, _, D = X.shape
    c = P["cos_c"].unsqueeze(1)                    # [Hn,1,D/2]
    s = P["sin_c"].unsqueeze(1)
    x1, x2 = X[..., :D // 2], X[..., D // 2:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

def _inv_col_rot(X: torch.Tensor, P: dict) -> torch.Tensor:
    Hn, _, D = X.shape
    c = P["cos_c"].unsqueeze(1)
    s = P["sin_c"].unsqueeze(1)
    x1, x2 = X[..., :D // 2], X[..., D // 2:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], dim=-1)

class Protected:
    """Per (layer, kv): list of blocks. Block = dict(payload, tag, chain_tag, header).
    header: {kind: prefix|tail, artifact_id?, block_index, aad fields...}.
    arithmetic profile: payload = float tensor [H,B,D] (+ separate MAC over bytes).
    TEE profile: payload = AES-CTR ciphertext bytes of the exact storage dtype."""

    def __init__(self, epoch_id, profile, dtype_name):
        self.epoch = epoch_id
        self.profile = profile
        self.dtype_name = dtype_name
        self.blocks = {}                            # (layer, kvf) -> [blockdict]

    def aad(self, li, kvf, j, true_len, sclass, artifact_id=None, shape_d=0,
            is_last_tail=False):
        # v4: PUBLIC prefix AAD binds ONLY block-level fields
        # (artifact_id ‖ block_index ‖ shape ‖ dtype ‖ epoch ‖ policy) — no
        # request-level true_length, enabling cross-request artifact reuse.
        # PRIVATE tail keeps full binding; true_length is bound ONLY on the final
        # tail block (truncation binding) per v4 contract.
        if sclass == "public":
            return ((artifact_id or b"\x00" * 16) + struct.pack(">Q", j) +
                    struct.pack(">QQ", B, shape_d) + self.dtype_name.encode() +
                    struct.pack(">Q", self.epoch) + b"public")
        tl = true_len if is_last_tail else -1
        return (struct.pack(">QQQq", li, kvf, j, tl) +
                str(sclass).encode() + self.dtype_name.encode() +
                struct.pack(">Q", self.epoch))

def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:                    # numpy has no bf16: reinterpret bits
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()

def protect_cache(cache, ctx, chain_seed=None) -> Protected:
    """ctx: dict(model_dig, tok_dig, rope, tokens(tensor), n_prefix_blocks,
    principal, nonce, epoch_id, profile in {tee, arithmetic},
    storage_dtype torch dtype, variant dict for ablations, page_map(out))."""
    prof = ctx["profile"]
    dt = ctx["storage_dtype"]
    prot = Protected(ctx["epoch_id"], prof, str(dt).split(".")[-1])
    npb = ctx["n_prefix_blocks"]
    tokens = ctx["tokens"]
    L = cache[0][1].shape[2]
    npad = math.ceil(L / B) * B
    nb = npad // B
    bsz = cache[0][1].shape[0]
    # canonical token digests per block (public info; decoder recomputes)
    tdigs = []
    for j in range(nb):
        seg = tokens[:, :min((j + 1) * B, tokens.shape[1])].numpy().tobytes() \
            if tokens.shape[1] > 0 else b""
        tdigs.append(hashlib.sha256(b"tok" + seg).digest())
    from corrected.freshness import KeyService, scope
    service = ctx.get("key_service") or KeyService(os.environ.get(
        "VEILKV_NONCE_DB", os.path.join(HDD, "freshness.sqlite")))
    if npb < 0 or npb > L // B:
        raise ValueError("only complete blocks may be public")
    if bsz != ctx.get("bsz", 1):
        raise ValueError("batch shape mismatch")
    if nb > npb:
        if not ctx.get("nonce") or not ctx.get("principal"):
            raise ValueError("private request requires principal and nonce")
        service.claim(scope(ctx["model_dig"], ctx["tok_dig"], ctx["rope"],
                            ctx["epoch_id"], ctx["principal"], ctx["nonce"]))
    for li, (K, V) in enumerate(cache):
        for kvf, X in ((0, K), (1, V)):
            H, D = X.shape[1], X.shape[3]
            Xp = torch.zeros(bsz, H, npad, D)
            Xp[:, :, :L] = X.to(torch.float32)
            row = []
            prev_tag = b"\x00" * 16
            for j in range(nb):
                # v4 chain junction: a REUSED prefix artifact's terminal tag seeds the
                # FIRST TAIL block (production: prefix is never re-encoded on a hit)
                if j == npb and chain_seed and (li, kvf) in chain_seed:
                    prev_tag = chain_seed[(li, kvf)]
                sclass = "public" if j < npb else "private"
                blk = Xp[:, :, j * B:(j + 1) * B]                    # [bsz,H,B,D]
                if sclass == "public" and not ctx["variant"].get("random_prefix"):
                    art, key = prefix_key(ctx["model_dig"], ctx["tok_dig"], ctx["rope"],
                                          tdigs[j], j, li, kvf, "public",
                                          prot.dtype_name, ctx["epoch_id"])
                    aad = prot.aad(li, kvf, j, L, sclass, artifact_id=art,
                                   shape_d=D, is_last_tail=False)
                else:
                    art = None
                    principal = ctx["principal"] if not ctx["variant"].get("no_salt") \
                        else ctx["principal"] + b"\x00noSalt"
                    nonce = ctx["nonce"] if not ctx["variant"].get("no_salt") else b""
                    key = tail_key(principal, nonce, j, li, 0, kvf, ctx["epoch_id"])
                    aad = prot.aad(li, kvf, j, L, sclass,
                                   is_last_tail=(j == nb - 1))
                # Authenticate chain links and dimensions, including shared-prefix junctions.
                aad += scope(b"chain-v1", prev_tag, bsz, H, D, len(cache), prot.profile)
                if art is not None:
                    service.public(scope(ctx["epoch_id"], art, aad), _tensor_bytes(blk.to(dt)))
                if prof == "tee":
                    payload = _tensor_bytes(blk.to(dt))
                    ct, tag = det_auth_enc(key, aad, payload)
                    row.append(dict(kind=sclass, artifact_id=art, j=j,
                                    ct=ct, tag=tag, prev_tag=prev_tag))
                else:
                    Y = torch.empty(bsz, H, B, D)
                    for hh in range(H):
                        k_mix = hkdf(key, b"mix-key" + struct.pack(">QQQ", li, kvf, hh)
                                     + (b"\x00" if bsz == 1 else struct.pack(">Q", 0)), 32)
                        P = derive_mix_params(k_mix, bsz, D, GIVENS_STEPS, ctx["variant"])
                        Xb = blk[:, hh]
                        Yh = _apply_row_rot(Xb, P)
                        Y[:, hh] = _apply_col_rot(Yh, P)
                    payload = Y.to(dt)
                    pbytes = _tensor_bytes(payload)
                    tag = mac_only(key, aad, pbytes)
                    row.append(dict(kind=sclass, artifact_id=art, j=j,
                                    ct=payload, tag=tag, prev_tag=prev_tag))
                prev_tag = row[-1]["tag"]
            prot.blocks[(li, kvf)] = row
    return prot

def recover_cache(prot: Protected, ctx, chain_seed=None) -> tuple:
    """TEE-side recovery; re-derives all keys. Raises TamperError on any violation."""
    dt_name = prot.dtype_name
    dt = torch.bfloat16 if dt_name == "bfloat16" else torch.float32
    out = []
    npb = ctx["n_prefix_blocks"]
    tokens = ctx["tokens"]
    from corrected.freshness import scope
    if prot.epoch != ctx["epoch_id"] or prot.profile != ctx["profile"]:
        raise TamperError("epoch/profile mismatch")
    if prot.dtype_name != str(ctx["storage_dtype"]).split(".")[-1]:
        raise TamperError("dtype mismatch")
    if len(prot.blocks) % 2 or not prot.blocks:
        raise TamperError("invalid layer/type layout")
    expected_keys = {(li, kvf) for li in range(len(prot.blocks)//2) for kvf in (0,1)}
    if set(prot.blocks) != expected_keys:
        raise TamperError("invalid layer/type layout")
    L = ctx["true_len"]
    nb = math.ceil(L / B)
    if any(len(row) != nb for row in prot.blocks.values()):
        raise TamperError("block count inconsistent with true length")
    if npb < 0 or npb > L // B:
        raise TamperError("invalid public block count")
    #                                    # decoder knows request length
    npad = nb * B
    if npad < L:
        raise TamperError("block count inconsistent with true_length (truncation)")
    bsz = ctx.get("bsz", 1)
    H, D = ctx["H"], ctx["D"]
    tdigs = []
    for j in range(nb):
        seg = tokens[:, :min((j + 1) * B, tokens.shape[1])].numpy().tobytes() \
            if tokens.shape[1] > 0 else b""
        tdigs.append(hashlib.sha256(b"tok" + seg).digest())
    for li in range(len(prot.blocks) // 2):
        rowK = []
        for kvf in (0, 1):
            row = prot.blocks[(li, kvf)]
            prev_tag = b"\x00" * 16
            Xp = torch.zeros(bsz, H, npad, D)
            for expected_j, blk in enumerate(row):
                if expected_j == npb and chain_seed and (li, kvf) in chain_seed:
                    prev_tag = chain_seed[(li, kvf)]
                if blk["j"] != expected_j:
                    raise TamperError("noncontiguous or duplicate block index")
                if blk["prev_tag"] != prev_tag:
                    raise TamperError("chain break at block %d" % blk["j"])
                j = blk["j"]
                sclass = "public" if j < npb else "private"
                if blk["kind"] != sclass:
                    raise TamperError("policy class mismatch")
                if sclass == "public" and not ctx["variant"].get("random_prefix"):
                    art, key = prefix_key(ctx["model_dig"], ctx["tok_dig"], ctx["rope"],
                                          tdigs[j], j, li, kvf, "public",
                                          prot.dtype_name, ctx["epoch_id"])
                    if blk["artifact_id"] != art:
                        raise TamperError("artifact substitution")
                    aad = prot.aad(li, kvf, j, L, sclass, artifact_id=art,
                                   shape_d=D, is_last_tail=False)
                else:
                    principal = ctx["principal"] if not ctx["variant"].get("no_salt") \
                        else ctx["principal"] + b"\x00noSalt"
                    nonce = ctx["nonce"] if not ctx["variant"].get("no_salt") else b""
                    key = tail_key(principal, nonce, j, li, 0, kvf, ctx["epoch_id"])
                    aad = prot.aad(li, kvf, j, L, sclass,
                                   is_last_tail=(j == len(row) - 1))
                aad += scope(b"chain-v1", prev_tag, bsz, H, D, len(prot.blocks)//2, prot.profile)
                if prot.profile == "tee":
                    pt = det_auth_dec(key, aad, blk["ct"], blk["tag"])
                    t = torch.frombuffer(bytearray(pt), dtype=dt).reshape(bsz, H, B, D)
                else:
                    pbytes = _tensor_bytes(blk["ct"])
                    if not hmac.compare_digest(mac_only(key, aad, pbytes), blk["tag"]):
                        raise TamperError("MAC verification failed")   # arithmetic
                    t = torch.empty(bsz, H, B, D)
                    for hh in range(H):
                        k_mix = hkdf(key, b"mix-key" + struct.pack(">QQQ", li, kvf, hh)
                                     + (b"\x00" if bsz == 1 else struct.pack(">Q", 0)), 32)
                        P = derive_mix_params(k_mix, bsz, D, GIVENS_STEPS, ctx["variant"])
                        Yh = blk["ct"][:, hh].float()
                        Yh = _inv_col_rot(Yh, P)
                        t[:, hh] = _inv_row_rot(Yh, P)
                Xp[:, :, j * B:(j + 1) * B] = t
                prev_tag = blk["tag"]
            rowK.append(Xp[:, :, :L])
        out.append((rowK[0], rowK[1]))
    return tuple(out)

def _digests(model, tok):
    cfg = model.config
    rope = json.dumps({k: getattr(cfg, k, None) for k in
                       ("rope_theta", "max_position_embeddings", "sliding_window")},
                      sort_keys=True).encode()
    return dict(model_dig=hashlib.sha256(json.dumps(
        {k: str(v) for k, v in cfg.to_dict().items() if k in
         ("architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "torch_dtype", "model_type")},
        sort_keys=True).encode()).digest(),
        tok_dig=hashlib.sha256(json.dumps(tok.init_kwargs, sort_keys=True,
                                          default=str).encode()).digest(),
        rope=hashlib.sha256(rope).digest())
