"""Recorded need_ratio=True variant with positive column scaling.

The official broadcast gives S_used = S0 D_r and M_used = M_rot D_m.
Probe states have the form C_j = u w^T + s_j a^T, where
u = S_used 1 and s_j is column j of S_used.

The routine estimates the marker direction from state differences, fits
orthogonality constraints by least squares and a scalar scan, and estimates
paired column rotations/scales from the public probe. Target decoding uses
the recovered columns as rows (a transpose, not a numerical inverse).
Under exact column recovery this leaves positive row scalings, which
normalized cosine matching ignores. Saved results evaluate this numerical
procedure; the orthogonal default-case theorem does not prove the variant.
"""
import json
import math
import time
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, cluster_states, rot_strided)
from run_suite import NATURAL, VICTIMS
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak

torch.manual_seed(42)
torch.set_num_threads(16)
MODEL = "Qwen/Qwen2.5-0.5B"
B, D = 16, 64
S_RATIO = M_RATIO = 1.25  # non-trivial scaling window [1/r, r]
res = {"mode": "need_ratio", "model": MODEL, "S_ratio": S_RATIO, "M_ratio": M_RATIO}

tok, model = load_model(MODEL, threads=16)
layers, H, D = model_dims(model)
pids, _pl, pchar = probe_ids(tok, target=1600, max_positions=getattr(model.config, "max_position_embeddings", None))
theta = build_theta(model, tok, NATURAL)
torch.manual_seed(42)
cfg = get_kvcloak_config(layers, H, D, B, theta, S_RATIO, M_RATIO, 2.0)

with torch.no_grad():
    pc = model(input_ids=pids, use_cache=True).past_key_values
plain_probe_v0 = pc[0][1][0].clone()

cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=True, add_a=True)
torch.manual_seed(1234)
prot = cloak.obfuscate(pc)
NB = pids.shape[1] // B


def fit_scaled_states(cents):
    """CORRECT semantics: torch broadcast S*S_ratios scales COLUMNS =>
    S_used = S0 D_r, states s_j = r_j * (orthonormal columns of S0).
    Identities: sum_j s_j = u;  L_j := (C_j - Cbar)@ahat = ||a|| (s_j - u/16);
    s_j = tau*uhat/16 + L_j with s_i.s_j = 0 (i!=j) -> quadratic in tau."""
    # ahat from pairwise diffs (sign-aligned)
    vs = []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            _, _, Vh_ = torch.linalg.svd(cents[i] - cents[j])
            v1 = Vh_[0]
            if vs and torch.dot(v1, vs[0]) < 0:
                v1 = -v1
            vs.append(v1)
    ahat = torch.stack(vs).mean(0)
    ahat = ahat * (1 if ahat.sum() > 0 else -1)
    ahat = ahat / ahat.norm()
    # u direction: projection rank-1 (left factor of u w^T after removing a)
    Cbar = cents.mean(0)
    U_, _, _ = torch.linalg.svd(Cbar)
    uhat = U_[:, 0]
    L = (cents - Cbar.unsqueeze(0)) @ ahat          # = ||a|| (s_j - u/16)
    # tau from orthogonality: (tau*u/16 + L_i) . (tau*u/16 + L_j) = 0, i != j
    # => tau^2 (u.u)/256 + tau/16 (u.(L_i+L_j)) + L_i.L_j = 0
    rows, rhs = [], []
    n = len(cents)
    for i in range(n):
        for j in range(i + 1, n):
            a1 = (uhat @ uhat).item() / 256.0
            a2 = (uhat @ (L[i] + L[j])).item() / 16.0
            a3 = (L[i] @ L[j]).item()
            rows.append([a1, a2])
            rhs.append(-a3)
    A = torch.tensor(rows)
    b = torch.tensor(rhs)
    sol = torch.linalg.lstsq(A, b).solution  # [tau^2, tau]
    taus = []
    for t2, t1 in [(sol[0].item(), sol[1].item())]:
        import math as _m
        disc = t1 * t1 + 4 * t2 * 0  # direct: we fit tau^2 and tau simultaneously
        taus.append(t1)
    # use tau estimate (least squares on mixed terms); refine by direct scan
    def ortherr(t):
        s = t * uhat.unsqueeze(0) / 16 + L
        G = s @ s.T
        off = G - torch.diag(torch.diagonal(G))
        return (off ** 2).sum().item() / (G.diagonal() ** 2).sum().item()
    cand = [sol[1].item()] + [x for x in torch.linspace(-3 * abs(sol[1].item()) - 5, 3 * abs(sol[1].item()) + 5, 401).tolist()]
    tau = min(cand, key=ortherr)
    s = tau * uhat.unsqueeze(0) / 16 + L
    u = tau * uhat
    # alpha under column scaling: ||s_i - s_j|| = sqrt(r_i^2 + r_j^2) with
    # r_j = ||s_j|| already recovered at TRUE scale via L
    r = s.norm(dim=1)
    num, den = 0.0, 0.0
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            sig = torch.linalg.svdvals(cents[i] - cents[j])[0].item()
            num += sig
            den += math.sqrt(r[i].item() ** 2 + r[j].item() ** 2)
    alpha = num / max(den, 1e-12)
    a = ahat * alpha
    # w consistent with u scale: C_j - s_j a^T = u w^T
    R = cents - torch.einsum("jb,d->jbd", s, a)
    w = (u @ R).mean(0) / (u @ u).item()
    return a, u, w, s, tau


DICT, attr, err = build_dictionary(model, H, D)
Dn = torch.nn.functional.normalize(DICT, dim=-1)
res["dict_selfcheck_probe"] = None

results_heads = {}
secrets = {}
for h in range(H):
    blocks = prot[0][1][0, h].float().view(NB, B, D)
    cid, cents, intra, inter, lab = cluster_states(blocks)
    res[f"head{h}_states"] = cid
    if cid != B:
        print(f"head {h}: only {cid} states, skip", flush=True)
        continue
    v_row = plain_probe_v0[h][0]
    t0 = time.time()
    a_hat, u_hat, w_hat, s_hat, tau = fit_scaled_states(cents)
    # ground truth for diagnostics
    conf = cfg[0][h][1]
    S_true = (conf["S"].float() * conf["S_ratios"]).T  # rows are columns of S0 D_r
    S_used_true = conf["S"].float() * conf["S_ratios"]  # column scaling (torch broadcast)
    a_true = conf["a"].float()
    cos = lambda x, y: (torch.dot(x, y) / (x.norm() * y.norm() + 1e-12)).item()
    # states (columns of S_used_true) vs rows of s_hat
    sc = torch.nn.functional.cosine_similarity(
        s_hat.unsqueeze(1), S_used_true.t().unsqueeze(0), dim=2)
    diag = {"cos_a": round(abs(cos(a_hat, a_true)), 6),
            "S_min_cos": round(sc.abs().max(1).values.min().item(), 6)}
    # No explicit D_r inverse: transpose demixing leaves positive row scalings.
    # M from (v, w'): angles by pair args; m_i by pair norms
    D2 = D // 2
    angles = torch.zeros(D2)
    m = torch.ones(D2)
    for i in range(D2):
        va, vb = v_row[i].item(), v_row[i + D2].item()
        wa, wb = w_hat[i].item(), w_hat[i + D2].item()
        if va * va + vb * vb < 1e-10:
            continue
        angles[i] = math.atan2(wb, wa) - math.atan2(vb, va)
        m[i] = math.hypot(wa, wb) / math.hypot(va, vb)
    M_rot = rot_strided(angles, D)
    # s_hat contains estimated columns as rows, in unknown column order.
    # Decoding below left-multiplies by s_hat, divides by paired column
    # scales, then right-multiplies by M_rot.T before cosine normalization.
    m_full = torch.cat([m, m])
    secrets[h] = (s_hat, a_hat, M_rot, m_full)  # demix via rows-of-states matrix
    diag["tau"] = round(float(tau), 6)
    res[f"head{h}_diag"] = diag
    print(f"head {h}: states={cid} cos_a={diag['cos_a']} S_min_cos={diag['S_min_cos']} "
          f"fit+Dr {time.time()-t0:.1f}s", flush=True)

# demix victims
res["victims"] = {}
for name, text in VICTIMS.items():
    vids = tok(text, return_tensors="pt").input_ids
    nb = vids.shape[1] // B
    if nb == 0:
        continue
    vids = vids[:, : nb * B]
    accs = {}
    for seed in (7001, 7002):
        with torch.no_grad():
            vc = model(input_ids=vids, use_cache=True).past_key_values
        torch.manual_seed(seed)
        pr = cloak.obfuscate(vc)
        cor = tot = 0
        for h in range(H):
            if h not in secrets:
                continue
            s_hat, a_hat, M_rot, m_full = secrets[h]
            bp = pr[0][1][0, h].float()
            for k in range(nb):
                C = bp[k * B:(k + 1) * B]
                best = None
                for j in range(B):
                    X = s_hat @ (C - torch.outer(s_hat[j], a_hat)) / m_full @ M_rot.T
                    Xn = torch.nn.functional.normalize(X, dim=-1)
                    sims = Xn @ Dn[:, h].T
                    sc_ = sims.max(1).values.sum().item()
                    if best is None or sc_ > best[0]:
                        best = (sc_, sims.argmax(1).tolist())
                true = vids[0, k * B:(k + 1) * B].tolist()
                cor += sum((Counter(best[1]) & Counter(true)).values())
                tot += B
        accs[f"seed{seed}"] = round(cor / tot, 4) if tot else None
    res["victims"][name] = accs
    print(f"victim {name}: {accs}", flush=True)

with open("results/ratio_05b.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print("DONE ratio", flush=True)
