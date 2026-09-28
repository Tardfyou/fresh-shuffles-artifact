"""Full-suite runner: one model end-to-end -> results/<tag>.json"""
import json
import sys
import time
from collections import Counter

import torch

from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                        build_dictionary, verify_dictionary, cluster_states,
                        recover_secrets, attack_block, ground_truth_errors)

NATURAL = [
    "The history of the Roman Empire spans centuries of political transformation.",
    "Machine learning systems require careful evaluation of training data quality.",
    "In 1969, humans first walked on the surface of the moon during Apollo 11.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The stock market reacted strongly to the central bank interest rate decision.",
    "Python is widely used for data analysis, web development, and automation.",
    "Ocean currents regulate global climate by redistributing heat across basins.",
    "A good textbook explains complex ideas through clear examples and exercises.",
]

VICTIMS = {
    "prose_en": (
        "The conference review process has long been criticized for its unpredictability, "
        "yet it persists as the primary mechanism for curating scientific output. Reviewers "
        "are assigned papers based on keyword matching and bid histories, and each "
        "submission receives independent assessments before a meta reviewer synthesizes a "
        "recommendation. Authors rebut, shepherds verify, and program chairs arbitrate. "
        "Along the way, artifacts are evaluated, code is executed, and claims are checked "
        "against reproducibility standards. Meanwhile, workshop venues absorb speculative "
        "work that is not yet ready for a full-length paper. Graduate students learn to "
        "frame contributions around threat models, attack surfaces, and defense "
        "mechanisms. Faculty balance novelty against rigor. Industry laboratories publish "
        "occasionally, guarding proprietary details while contributing measurements from "
        "production systems. The result is a slow, noisy, but ultimately trusted filter "
        "for research that shapes how the community thinks about privacy, integrity, and "
        "availability in large-scale machine learning systems."
    ),
    "python_code": (
        "import hashlib\n"
        "from dataclasses import dataclass\n\n"
        "@dataclass\nclass Record:\n    url: str\n    payload: bytes\n    seq: int\n\n"
        "def digest(rec: Record) -> str:\n    h = hashlib.blake2b(digest_size=16)\n"
        "    h.update(rec.url.encode())\n    h.update(rec.payload)\n"
        "    h.update(str(rec.seq).encode())\n    return h.hexdigest()\n\n"
        "def rolling_window(xs, k):\n    for i in range(len(xs) - k + 1):\n"
        "        yield tuple(xs[i:i+k])\n\n"
        "def shingles(text, k=5):\n    toks = text.lower().split()\n"
        "    return set(' '.join(w) for w in rolling_window(toks, k))\n\n"
        "class UnionFind:\n    def __init__(self):\n        self.parent = {}\n"
        "        self.size = {}\n    def find(self, x):\n"
        "        self.parent.setdefault(x, x)\n        self.size.setdefault(x, 1)\n"
        "        while self.parent[x] != x:\n            self.parent[x] = self.parent[self.parent[x]]\n"
        "            x = self.parent[x]\n        return x\n    def union(self, a, b):\n"
        "        ra, rb = self.find(a), self.find(b)\n        if ra == rb: return\n"
        "        if self.size[ra] < self.size[rb]: ra, rb = rb, ra\n"
        "        self.parent[rb] = ra\n        self.size[ra] += self.size[rb]\n"
    ),
    "chinese": (
        "隐私保护推理系统将大语言模型的不同部分放置在不同的执行环境中。可信环境持有秘密参数，"
        "不可信的加速器只能看到混淆之后的键值缓存。研究者的目标是让攻击者即使能够读取缓存内容，"
        "也无法还原用户的原始输入。为此，系统在缓存写入显存之前对其进行三重变换：行内打乱、"
        "稠密线性混合以及加性标记。打乱方式每个数据块都重新随机抽取，混合矩阵则长期保持不变。"
        "这样的设计在理论上能够抵抗差分攻击与碰撞攻击，因为攻击者无法把不同块之间的对应关系"
        "建立起来。然而，如果攻击者可以提交自己选择的文本并观察对应的受保护缓存，事情就完全"
        "不同了。重复字符构成的特殊输入会让块内所有行变得完全相同，此时打乱操作失去意义，"
        "剩余的变换结构就会通过有限的状态数量暴露出来。攻击者收集足够多的块之后，可以按照"
        "代数方法逐一恢复秘密参数，进而对其他用户的缓存内容进行解码。这说明任何为合法恢复"
        "而设计的辅助结构，都必须考虑被恶意利用的可能性。防御者应当在效率与安全之间寻找"
        "更加谨慎的平衡，例如引入真正的密钥轮换机制，或者放弃低支持度的恢复标记设计。"
    ),
    "dialogue": (
        "User: Can you help me understand how KV caching works in transformer inference?\n"
        "Assistant: Of course. During autoregressive generation, each new token only needs "
        "attention over previous tokens' keys and values, which stay fixed. Caching them "
        "avoids recomputing the whole prefix every step.\n"
        "User: Why does memory become a bottleneck?\n"
        "Assistant: The cache grows linearly with sequence length, times layers, times heads, "
        "times head dimension. For long contexts this can exceed the memory holding the "
        "weights themselves.\n"
        "User: What do serving systems do about it?\n"
        "Assistant: Paged attention, quantization, prefix sharing across requests, and "
        "offloading to CPU or disk. Each trades accuracy or latency for capacity.\n"
        "User: Is the cache sensitive data?\n"
        "Assistant: Very. Work has shown prompts can be reconstructed from leaked caches, "
        "which is exactly the threat model of obfuscation defenses.\n"
        "User: How are those defenses evaluated?\n"
        "Assistant: Usually against known inversion attacks on plaintext or lightly "
        "protected caches, plus adaptive chosen-plaintext variants.\n"
    ),
    "math": (
        "Let H denote the empirical second moment matrix of calibration activations, "
        "H = (2/n) X X^T. GPTQ adds damping, H~ = H + lambda I with "
        "lambda = 0.01 * mean(diag H), takes the upper Cholesky factor U of H~^{-1} "
        "satisfying U^T U = H~^{-1}, and processes columns sequentially: "
        "q_i = quant(w_i - sum_{j<i} e_j U_{j,i}), e_i = (w_i - q_i)/U_{i,i}. "
        "The residual satisfies W - Q = E U with E the rounding residual matrix, "
        "giving F = (W - Q) U^{-1} entries bounded by half the grid step for "
        "unsaturated columns. For a b x b orthogonal S and rank-one marker "
        "A = e_r a^T, repeated-token blocks satisfy V = 1 v^T, hence "
        "C_j = u w^T + s_j a^T with u = S 1, s_j = S e_j, w = M^T v. "
        "Differences C_j - C_k = (s_j - s_k) a^T are rank one, exposing a; "
        "projection onto the orthogonal complement of a exposes u and w; and "
        "since u^T s_j = 1 for orthogonal S, w = (u^T C_j - a^T)/b."
    ),
}


def run(model_name, tag, b=16, need_ratio=False, S_ratio=1.0, M_ratio=1.0, threads=16):
    t0 = time.time()
    res = {"model": model_name, "tag": tag, "b": b}
    tok, model = load_model(model_name, threads)
    layers, H, D = model_dims(model)
    res.update(layers=layers, kv_heads=H, head_dim=D)
    print(f"[{tag}] {model_name}: layers={layers} kv_heads={H} head_dim={D}", flush=True)

    # probe length scales with b: need ~b*H_b blocks (coupon collector), 2.5x margin
    Hb = sum(1.0 / k for k in range(1, b + 1))
    import os
    target = max(int(os.environ.get('PROBE_TOKENS', '1024')), int(2.5 * b * Hb) * b)
    maxpos = getattr(model.config, "max_position_embeddings", None)
    pids, prefix_len, pchar = probe_ids(tok, target=target, max_positions=maxpos)
    res["probe_char"] = pchar
    res["probe_raw_entry"] = "(direct-id)" not in pchar
    res["probe_prefix_tokens"] = int(prefix_len)
    res["probe_tokens"] = int(pids.shape[1])
    print(f"[{tag}] probe: char={pchar!r} raw_entry={res['probe_raw_entry']} prefix={prefix_len} total={pids.shape[1]}", flush=True)

    theta = build_theta(model, tok, NATURAL)
    torch.manual_seed(42)
    from get_kvcloak_config import get_kvcloak_config
    cfg = get_kvcloak_config(layers, H, D, b, theta, S_ratio, M_ratio, 2.0)

    with torch.no_grad():
        pc = model(input_ids=pids, use_cache=True).past_key_values
    plain_probe_full = pc[0][1][0].clone()  # full (prefix included); sliced later

    # premise: layer-0 V rows identical
    prem = all(torch.equal(plain_probe_full[h], plain_probe_full[h][:1].expand_as(plain_probe_full[h]))
               for h in range(H))
    res["premise_rows_identical"] = bool(prem)

    from kvcloak import KVCloak
    cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=need_ratio, add_a=True)
    torch.manual_seed(1234)
    t1 = time.time()
    prot_probe = cloak.obfuscate(pc)
    res["probe_obf_time_s"] = round(time.time() - t1, 3)

    NB = pids.shape[1] // b                    # unpadded block count
    nblk_pad = prot_probe[0][1].shape[2] // b  # obfuscate pads to block boundary
    first_blk = (prefix_len + b - 1) // b      # first block fully inside the run
    ncore = NB - first_blk
    plain_probe_v0 = plain_probe_full[:, first_blk * b:NB * b].clone()  # core-only rows
    res["probe_core_blocks"] = int(ncore)
    secrets = {}
    for h in range(H):
        blocks = prot_probe[0][1][0, h].float().view(nblk_pad, b, D)[first_blk:NB]
        cid, cents, intra, inter, lab = cluster_states(blocks)
        res[f"head{h}_states"] = cid
        res[f"head{h}_separation"] = round(inter / max(intra, 1e-12), 1) if inter < float("inf") else None
        # minimal prefix collecting all states (probe cost)
        first_all = None
        seen = set()
        for i in range(NB):
            seen.add(int(lab[i]))
            if len(seen) == cid:
                first_all = i + 1
                break
        res[f"head{h}_blocks_to_all_states"] = first_all
        if cid == b:
            sec = recover_secrets(cents, plain_probe_v0[h], b)
            errs = ground_truth_errors(sec, cfg[0][h][1], plain_probe_v0[h])
            res[f"head{h}_recovery"] = {k: (round(v, 6) if isinstance(v, float) else v)
                                        for k, v in errs.items()}
            secrets[h] = sec
        print(f"[{tag}] head {h}: {cid}/{b} states, sep={res[f'head{h}_separation']}, "
              f"first-all={first_all}", flush=True)

    DICT, norm_attr, norm_err = build_dictionary(model, H, D)
    res["norm_detected"] = norm_attr
    res["norm_capture_err"] = float(norm_err)
    core_pids = pids[:, first_blk * b:NB * b]
    res["dict_selfcheck"] = round(verify_dictionary(DICT, plain_probe_v0, core_pids), 4)
    Dn = torch.nn.functional.normalize(DICT, dim=-1)
    print(f"[{tag}] dict: norm={norm_attr} capterr={norm_err:.2e} selfcheck={res['dict_selfcheck']}", flush=True)

    # victims
    res["victims"] = {}
    for name, text in VICTIMS.items():
        vids = tok(text, return_tensors="pt").input_ids
        nb = vids.shape[1] // b
        if nb == 0:
            continue
        vids = vids[:, : nb * b]
        accs = {}
        raw_accs = {}
        rt = []
        for seed in (5001, 5002):
            with torch.no_grad():
                vc = model(input_ids=vids, use_cache=True).past_key_values
            torch.manual_seed(seed)
            prot = cloak.obfuscate(vc)
            blocks_prot = prot[0][1][0].float()
            cor = tot = 0
            for h in range(H):
                if h not in secrets:
                    continue
                sec = secrets[h]
                for k in range(nb):
                    C = blocks_prot[h, k * b:(k + 1) * b]
                    t = time.perf_counter()
                    pred = attack_block(C, sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
                    rt.append(time.perf_counter() - t)
                    true = vids[0, k * b:(k + 1) * b].tolist()
                    cor += sum((Counter(pred) & Counter(true)).values())
                    tot += b
            accs[f"seed{seed}"] = round(cor / tot, 4) if tot else None
            raw_accs[f"seed{seed}"] = [int(cor), int(tot)]
        res["victims"][name] = {"acc": accs, "raw": raw_accs, "tokens": int(vids.shape[1]),
                                "ms_per_block": round(1000 * sum(rt) / len(rt), 1)}
        print(f"[{tag}] victim {name}: {accs} ({vids.shape[1]} tok)", flush=True)

    # fresh-secrets control on one victim
    tv = tok(VICTIMS["prose_en"], return_tensors="pt").input_ids
    nb = tv.shape[1] // b
    tv = tv[:, : nb * b]
    torch.manual_seed(9991)
    from get_kvcloak_config import get_kvcloak_config as gk
    cfg2 = gk(layers, H, D, b, theta, S_ratio, M_ratio, 2.0)
    cloak2 = KVCloak(cfg2, dtype=torch.bfloat16, fused=False, need_ratio=need_ratio, add_a=True)
    with torch.no_grad():
        vc2 = model(input_ids=tv, use_cache=True).past_key_values
    torch.manual_seed(9992)
    prot2 = cloak2.obfuscate(vc2)
    cor = tot = 0
    h = 0
    if h in secrets:
        sec = secrets[h]
        bp = prot2[0][1][0, h].float()
        for k in range(nb):
            pred = attack_block(bp[k * b:(k + 1) * b], sec["s"], sec["a_hat"], sec["M"], Dn[:, h])
            true = tv[0, k * b:(k + 1) * b].tolist()
            cor += sum((Counter(pred) & Counter(true)).values())
            tot += b
    res["control_fresh_secrets"] = round(cor / tot, 4) if tot else None
    res["control_fresh_secrets_raw"] = [int(cor), int(tot)]
    res["control_fresh_secrets_counts"] = [int(cor), int(tot)]
    res["total_time_s"] = round(time.time() - t0, 1)
    print(f"[{tag}] control fresh-secrets acc: {res['control_fresh_secrets']} | total {res['total_time_s']}s", flush=True)

    with open(f"results/{tag}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    model_name, tag = sys.argv[1], sys.argv[2]
    kw = {}
    if len(sys.argv) > 3:
        kw["b"] = int(sys.argv[3])
    run(model_name, tag, **kw)
