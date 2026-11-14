"""Frozen calibrator used for the retained 12-configuration main-table counts.

New corrected follow-ups use attack_lib.recover_secrets.
"""
import math
import torch
from attack_lib import rot_strided

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


