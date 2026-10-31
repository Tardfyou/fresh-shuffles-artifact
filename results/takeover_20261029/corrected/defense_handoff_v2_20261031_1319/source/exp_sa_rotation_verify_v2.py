"""S/a rotation known-key functional verification v2 (rework per Codex sheet
presentation/sa-handoff-review-20261031/CLAUDE_REWORK_ZH.md).

Legal consumer path only: the test program generates and holds every
configuration and calls the official KV-Cloak protect/recover interfaces.
No attack code, no parameter identification, no dictionary reconstruction.

Fixes vs the rejected v1 delivery (all eight items of the rework sheet):
 1. M fixed: baseline config (seed 42); S_a_refresh regenerates S/a per
    request seed, then explicitly restores per-layer/per-head/per-K,V-branch
    M_angles/M_ratios/M_ratio/theta from the baseline and asserts equality;
    S/a/S_ratios asserted changed. Checks saved per request.
 2. attention_mask/position_ids passed to every forward: prefill, every
    continuation segment, and the full-forward reference.
 3. Timestep alignment: cached-segment logits[j] is compared against the
    full-forward logits at physical position cont_start+seg_off+j; whole
    [seg_len, vocab] tensors, both ends same shape, one-to-one positions.
 4. Three paths kept: full forward, plain cached, protected cached; primary
    functional error is protected_cached vs plain_cached; fresh-full vs
    plain-cached reported separately; plus a no-protection sanity control
    (two independent plain-cache copies must agree exactly).
 5. Real 2->3->1 consecutive continuation over six reserved real tokens;
    one prefill per condition; plain and protected branches each carry their
    own cache state across segments; right padding only inside history.
 6. Per-request S/a refresh (reproducible sha256-derived seed, no Python
    hash); config lifetime spans the three segments; mask variants of the
    same (model, sid) share the declared S/a config (declared in PROTOCOL).
    Paired shuffle: both cloak instances finish device-tensor init under a
    declared prep seed BEFORE the declared shuffle seed is set right before
    each obfuscate; actual permutations recorded via a declared torch.rand
    recorder (values unmodified) and fingerprint-compared across conditions.
 7. Public prefix truncated to exactly 16 tokens and asserted per model;
    tokenizer special-token behaviour recorded; actual input/mask/position
    fixtures saved once per (model, sid, mask) and hash-bound into every
    record, identical across dtype/config pairings by construction.
 8. Mutation checks over every layer, K and V, comparing uint8 byte views:
    separately "object handed to the operator was rewritten" and "retained
    plain baseline stayed unpolluted". True in-place behaviour preserved.

Matrix: 100 lifecycle IDs x 2 models x 2 native dtypes x 2 config
conditions x 5 mask shapes x 3 continuation segments = 12,000 keys.
Smoke = first 5 IDs of the same frozen list = 600 keys, separate directory.

Usage:
  prepare  : python exp_sa_rotation_verify_v2.py --prepare-outdir OUTDIR [--smoke]
  fixtures : python exp_sa_rotation_verify_v2.py --prepare-fixtures OUTDIR
  worker   : python exp_sa_rotation_verify_v2.py --worker OUTDIR MODEL_SHORT DTYPE CONDITION
  launch   : python exp_sa_rotation_verify_v2.py --launch OUTDIR [--smoke] [--workers N]
  gates    : python exp_sa_rotation_verify_v2.py --gates OUTDIR
  merge    : python exp_sa_rotation_verify_v2.py --merge OUTDIR
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CANONICAL_INPUTS = ("results/takeover_20261029/"
                    "defense_handoff_20261030_1600/inputs.jsonl")
SUBSETS = ("results/main_public/experiment_subsets.json")

MODELS = [("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
          ("unsloth/Llama-3.2-1B", "Llama-3.2-1B")]
DTYPES = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]
CONDITIONS = ["original", "S_a_refresh"]
MASKS = [("no_pad", None, 0), ("left8", "left", 8), ("right8", "right", 8),
         ("left16", "left", 16), ("right16", "right", 16)]
PREFIX_LEN = 16
MAX_BODY = 134           # declared body truncation, same as v1 protocol
MIN_BODY = 10            # below this the ID is error/not_run, never replaced
SEGMENTS = [("seg1", 1, 2, 0), ("seg2", 2, 3, 2), ("seg3", 3, 1, 5)]  # name, idx, len, offset
PREFIX_TEXT = ("The following is a public system prompt used for testing "
               "purposes. It provides general instructions.")
BASE_SEED = 42           # baseline ("original") config seed, per model
PREP_SEED = 20261031     # device-tensor init pairing seed (A-matrix rows)
SHUFFLE_SEED = 1234      # declared paired per-block shuffle seed
THREADS = 4
# fp32/bf16 functional pass thresholds, pre-declared, applied per segment key
THRESHOLDS = {"float32": 1e-3, "bfloat16": 0.6}


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def request_seed_for(sid, model_short):
    """Reproducible per-(model, sid) S/a seed. No Python hash (run-unstable)."""
    return int(hashlib.sha256(f"{sid}|{model_short}".encode()).hexdigest()[:16], 16)


def _parse_shard(spec):
    """Parse a 'i/n' shard spec into (i, n), or None when unsharded."""
    if not spec:
        return None
    i, n = (int(x) for x in spec.split("/"))
    assert 0 <= i < n, f"bad shard {spec}"
    return i, n


# ------------------------------------------------------------------ fixtures
def make_prefix_ids(tok):
    ids = tok(PREFIX_TEXT, return_tensors="pt").input_ids[0]
    assert len(ids) >= PREFIX_LEN, f"prefix shorter than {PREFIX_LEN}: {len(ids)}"
    return ids[:PREFIX_LEN].tolist(), ids.tolist()


def build_fixture(prefix_ids, body_ids, pad_side, pad_len, pad_token_id):
    """History = prefix + optional padding + body[:-6]; continuation = last 6
    real body tokens. Right padding only ever sits inside the history; the
    six continuation tokens are always real source tokens appended after it.
    Left padding: position ids = cumsum(mask)-1 (HF convention).
    Right padding: physical arange positions; continuation positions continue
    after the padded history (declared convention)."""
    cont = body_ids[-6:]
    hist = body_ids[:-6]
    prefix = torch.tensor(prefix_ids, dtype=torch.long)
    hist_t = torch.tensor(hist, dtype=torch.long)
    if pad_side is None:
        ids = torch.cat([prefix, hist_t])
        mask = torch.ones(len(ids), dtype=torch.long)
        pos = torch.arange(len(ids))
    elif pad_side == "left":
        pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ids = torch.cat([prefix, pad, hist_t])
        mask = torch.cat([torch.ones(len(prefix)), torch.zeros(pad_len),
                          torch.ones(len(hist_t))]).long()
        pos = torch.cumsum(mask, 0) - 1
    elif pad_side == "right":
        pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ids = torch.cat([prefix, hist_t, pad])
        mask = torch.cat([torch.ones(len(prefix) + len(hist_t)),
                          torch.zeros(pad_len)]).long()
        pos = torch.arange(len(ids))
    else:
        raise ValueError(pad_side)
    return ids, mask, pos, torch.tensor(cont, dtype=torch.long)


def fixture_digest(ids, mask, pos, cont):
    return sha256_json({"ids": ids.tolist(), "mask": mask.tolist(),
                        "pos": pos.tolist(), "cont": cont.tolist()})


# ------------------------------------------------------------- config (fix 1)
def build_base_config(layers, H, D_head, theta):
    torch.manual_seed(BASE_SEED)
    from get_kvcloak_config import get_kvcloak_config
    return get_kvcloak_config(layers, H, D_head, 16, theta, 1.0, 1.0, 2.0)


def build_refresh_config(base_cfg, layers, H, D_head, theta, seed):
    """Per-request S/a refresh with M held at the baseline values: regenerate
    under the request seed, then explicitly restore every M-related field and
    theta from the baseline config, layer by layer, head by head, K/V branch
    by K/V branch. Returns (config, m_fixed, s_a_changed) with per-field
    equality/change checks."""
    torch.manual_seed(seed)
    from get_kvcloak_config import get_kvcloak_config
    cfg = get_kvcloak_config(layers, H, D_head, 16, theta, 1.0, 1.0, 2.0)
    m_fixed = {"M_angles": True, "M_ratios": True, "M_ratio": True, "theta": True}
    # S and a must actually change. S_ratios cannot: with S_ratio=1.0 it is the
    # constant all-ones vector by construction (rand*(1-1)+1) and is unused
    # under need_ratio=False; recorded and verified constant instead.
    s_a_changed = {"S": True, "a": True}
    s_ratios_constant = True
    for li in range(layers):
        for hi in range(H):
            for ki in (0, 1):
                b, c = base_cfg[li][hi][ki], cfg[li][hi][ki]
                c["M_angles"] = b["M_angles"].clone()
                c["M_ratios"] = b["M_ratios"].clone()
                c["M_ratio"] = b["M_ratio"]
                c["theta"] = b["theta"]
                m_fixed["M_angles"] &= bool(torch.equal(c["M_angles"], b["M_angles"]))
                m_fixed["M_ratios"] &= bool(torch.equal(c["M_ratios"], b["M_ratios"]))
                m_fixed["M_ratio"] &= (c["M_ratio"] == b["M_ratio"])
                m_fixed["theta"] &= (c["theta"] == b["theta"])
                s_a_changed["S"] &= not torch.equal(c["S"], b["S"])
                s_a_changed["a"] &= not torch.equal(c["a"], b["a"])
                s_ratios_constant &= bool(torch.equal(c["S_ratios"], b["S_ratios"]))
    m_fixed["all"] = all(m_fixed.values())
    s_a_changed["all"] = all(s_a_changed.values())
    s_a_changed["S_ratios_constant"] = s_ratios_constant
    s_a_changed["note"] = ("S_ratios is the constant all-ones vector under "
                           "S_ratio=1.0 (need_ratio=False); verified constant, "
                           "not required to change")
    return cfg, m_fixed, s_a_changed


def config_fingerprint(cfg):
    """sha256 over every field of every layer/head/branch, binding the exact
    configuration object consumed by the official operator."""
    h = hashlib.sha256()
    for layer in cfg:
        for head in layer:
            for branch in head:
                for k in sorted(branch):
                    v = branch[k]
                    if isinstance(v, torch.Tensor):
                        h.update(k.encode())
                        h.update(v.detach().cpu().contiguous()
                                 .numpy().tobytes())
                    else:
                        h.update(f"{k}={v!r}".encode())
    return h.hexdigest()


# -------------------------------------------------- declared instrumentation
class PermRecorder:
    """Declared debug instrumentation (rework sheet section 1.6): records the
    torch.rand draws the official obfuscate consumes for its per-block row
    shuffles and computes the same argsort the implementation uses. Values
    are produced by the untouched original torch.rand; nothing is modified.
    Fingerprint lets the two conditions' actual shuffles be verified equal."""

    def __init__(self):
        self.draws = []
        self._orig = None

    def __enter__(self):
        self._orig = torch.rand
        rec = self

        def wrapped(*a, **k):
            v = rec._orig(*a, **k)
            rec.draws.append(v.detach().clone())
            return v
        torch.rand = wrapped
        return self

    def __exit__(self, *exc):
        torch.rand = self._orig
        return False

    def fingerprint(self, block_size):
        h = hashlib.sha256()
        shapes = []
        for v in self.draws:
            shapes.append(list(v.shape))
            h.update(torch.argsort(v, dim=1).to(torch.int64)
                     .numpy().tobytes())
        return h.hexdigest(), len(self.draws), shapes


# ------------------------------------------------------------ cache helpers
def _u8(t):
    t = t.detach().cpu().contiguous()
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def bytes_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(_u8(a), _u8(b))


def cache_bytes_report(before, after, layers):
    """Per-layer K/V byte-level comparison of two legacy-cache tuples."""
    out = []
    for li in range(layers):
        for kv, name in ((0, "K"), (1, "V")):
            a, b = before[li][kv], after[li][kv]
            entry = {"layer": li, "kv": name,
                     "shape_before": list(a.shape), "shape_after": list(b.shape),
                     "dtype_before": str(a.dtype), "dtype_after": str(b.dtype),
                     "shape_match": a.shape == b.shape, "dtype_match": a.dtype == b.dtype,
                     "bytes_equal": bytes_equal(a, b)}
            out.append(entry)
    return out


def fresh_dynamic(tuple_cache):
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(
        tuple((k, v) for k, v in tuple_cache))


def clone_legacy(tuple_cache):
    return tuple((k.clone(), v.clone()) for k, v in tuple_cache)


# ------------------------------------------------------------ measurement
def run_request(model, cloak, fixture, seg_outputs_writer):
    """One request-key: prefill once, protect once, recover, then consume the
    same 2/3/1 continuation through plain / plain-sanity / recovered caches.
    Returns (per-segment records, request-level record)."""
    ids, mask, pos, cont = fixture["ids"], fixture["mask"], fixture["pos"], fixture["cont"]
    cont_start = int(ids.shape[0])

    # ---- full-forward reference (mask/position passed; fix 2/3) ----------
    total_ids = torch.cat([ids, cont])
    total_mask = torch.cat([mask, torch.ones(len(cont), dtype=torch.long)])
    total_pos = torch.cat([pos, torch.arange(int(pos[-1]) + 1, int(pos[-1]) + 1 + len(cont))])
    assert total_ids.shape[0] == total_mask.shape[0] == total_pos.shape[0]
    with torch.no_grad():
        full = model(input_ids=total_ids.unsqueeze(0),
                     attention_mask=total_mask.unsqueeze(0),
                     position_ids=total_pos.unsqueeze(0),
                     use_cache=False)
    full_logits = full.logits[0]  # [hist+6, vocab]; slice per segment below

    # ---- prefill once (fix 2/5) ------------------------------------------
    with torch.no_grad():
        pre = model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0),
                    position_ids=pos.unsqueeze(0), use_cache=True)
    src = clone_legacy(pre.past_key_values.to_legacy_cache())
    n_layers = len(src)

    # snapshots for the two mutation checks (fix 8), taken before any operator
    plain_snapshot = clone_legacy(src)     # pollution check for the plain baseline
    op_input = clone_legacy(src)           # object handed to the operator
    op_snapshot = clone_legacy(op_input)   # its own pre-call state

    # ---- paired-shuffle protect (fix 6): device init already done by the
    # caller under PREP_SEED; declared shuffle seed set right before obfuscate
    torch.manual_seed(SHUFFLE_SEED)
    with PermRecorder() as rec:
        protected = cloak.obfuscate(fresh_dynamic(op_input))
    perm_hash, perm_calls, perm_shapes = rec.fingerprint(16)
    recovered_dyn = cloak.deobfuscate(protected)
    recovered = clone_legacy(recovered_dyn.to_legacy_cache())

    # ---- mutation checks (fix 8), before any consumption ------------------
    op_report = cache_bytes_report(op_snapshot, op_input, n_layers)
    plain_report = cache_bytes_report(plain_snapshot, src, n_layers)
    op_mutated = [f"{e['layer']}:{e['kv']}" for e in op_report if not e["bytes_equal"]]
    plain_polluted = [f"{e['layer']}:{e['kv']}" for e in plain_report if not e["bytes_equal"]]

    # ---- cache-level roundtrip (diagnostic; logits are the functional test)
    rt_max, len_mismatch = 0.0, []
    for li in range(n_layers):
        for kv, name in ((0, "K"), (1, "V")):
            r, s = recovered[li][kv], src[li][kv]
            if r.shape != s.shape:
                len_mismatch.append(f"{li}:{name}{list(r.shape)}!={list(s.shape)}")
                continue
            d = float((r.float() - s.float()).abs().max())
            rt_max = max(rt_max, d)
    cache_roundtrip = {"max_abs_diff": rt_max, "len_mismatch": len_mismatch,
                       "len_match_all": not len_mismatch}

    # ---- three consumer paths, sequential 2->3->1 (fix 4/5) ---------------
    plain_a = clone_legacy(src)
    plain_b = clone_legacy(src)
    rec_c = clone_legacy(recovered)
    seg_records = []
    cursor = cont_start
    for seg_name, seg_idx, seg_len, seg_off in SEGMENTS:
        seg = cont[seg_off:seg_off + seg_len]
        seg_pos = torch.arange(int(pos[-1]) + 1 + seg_off,
                               int(pos[-1]) + 1 + seg_off + seg_len)
        hist_mask = torch.cat([mask, torch.ones(seg_off + seg_len, dtype=torch.long)])
        kw = dict(input_ids=seg.unsqueeze(0), attention_mask=hist_mask.unsqueeze(0),
                  position_ids=seg_pos.unsqueeze(0), use_cache=True)
        with torch.no_grad():
            out_a = model(**kw, past_key_values=fresh_dynamic(plain_a))
            out_b = model(**kw, past_key_values=fresh_dynamic(plain_b))
            out_c = model(**kw, past_key_values=fresh_dynamic(rec_c))
        la, lb, lc = out_a.logits[0], out_b.logits[0], out_c.logits[0]
        f_log = full_logits[cont_start + seg_off: cont_start + seg_off + seg_len]

        def diff(x, y):
            d = (x.float() - y.float())
            return float(d.abs().max()), float(d.abs().mean())

        pp_max, pp_mean = diff(lc, la)
        pf_max, _ = diff(la, f_log)
        rf_max, _ = diff(lc, f_log)
        sb_max, _ = diff(lb, la)
        rec = {
            "seg": seg_name, "seg_index": seg_idx,
            "seg_len_declared": seg_len, "seg_len_actual": int(seg.shape[0]),
            "seg_abs_range": [cont_start + seg_off, cont_start + seg_off + seg_len],
            "cursor_before": cursor, "cont_start": cont_start,
            "sanity_max": sb_max,
            "prot_vs_plain_max": pp_max, "prot_vs_plain_mean": pp_mean,
            "prot_vs_plain_argmax_all": bool(torch.all(lc.argmax(-1) == la.argmax(-1))),
            "prot_vs_plain_argmax_frac": float((lc.argmax(-1) == la.argmax(-1)).float().mean()),
            "plain_vs_full_max": pf_max,
            "plain_vs_full_argmax_all": bool(torch.all(la.argmax(-1) == f_log.argmax(-1))),
            "prot_vs_full_max": rf_max,
            "prot_vs_full_argmax_all": bool(torch.all(lc.argmax(-1) == f_log.argmax(-1))),
            "shape_match": bool(la.shape == lc.shape == f_log.shape),
            "logits_dtype": str(lc.dtype),
            "finite": bool(torch.isfinite(lc).all() and torch.isfinite(la).all()),
            "seg_tokens": seg.tolist(),
        }
        seg_records.append(rec)
        plain_a = clone_legacy(out_a.past_key_values.to_legacy_cache())
        plain_b = clone_legacy(out_b.past_key_values.to_legacy_cache())
        rec_c = clone_legacy(out_c.past_key_values.to_legacy_cache())
        cursor += seg_len
    seg_outputs_writer(seg_records)

    request = {
        "hist_len": cont_start, "cont_len": int(cont.shape[0]),
        "perm_hash": perm_hash, "perm_calls": perm_calls,
        "perm_shapes": perm_shapes, "shuffle_seed": SHUFFLE_SEED,
        "prep_seed": PREP_SEED,
        "mutation": {
            "operator_input_mutated": op_mutated,
            "operator_input_mutated_n": len(op_mutated),
            "plain_baseline_polluted": plain_polluted,
            "plain_baseline_unpolluted": not plain_polluted,
            "n_entries": len(op_report),
            "detail_op": op_report, "detail_plain": plain_report,
        },
        "cache_roundtrip": cache_roundtrip,
        "input_sha256": fixture["digest"],
    }
    return seg_records, request


# ------------------------------------------------------------------ prepare
def load_inputs():
    inputs = [json.loads(l) for l in open(CANONICAL_INPUTS)]
    sids = [s["sid"] for s in inputs]
    assert len(sids) == 100 and len(set(sids)) == 100, "frozen list must be 100 unique IDs"
    subsets = json.load(open(SUBSETS))
    life = None
    for k, v in subsets.items():
        if "lifecycle" in k:
            life = set(v if isinstance(v, list) else v.get("ids", v))
    if life is not None:
        assert set(sids) == life, "inputs.jsonl does not match lifecycle subset"
    return inputs


def cmd_prepare_outdir(args):
    d = args.outdir
    if os.path.exists(d):
        existing = [f for f in os.listdir(d)]
        if any(f.startswith(("checks_", "workers")) for f in existing):
            raise SystemExit(f"refusing to overwrite results in {d}")
    os.makedirs(d, exist_ok=True)
    inputs = load_inputs()
    with open(os.path.join(d, "inputs.jsonl"), "w") as f:
        for s in inputs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    protocol = {
        "created": now(),
        "task_source": "presentation/sa-handoff-review-20261031/CLAUDE_REWORK_ZH.md",
        "supersedes": "defense_handoff_20261030_1600 (rejected; preserved untouched)",
        "scope": "known-key S/a rotation functional verification, legal consumer path",
        "mode": "smoke_first5" if args.smoke else "full_100",
        "input_spec": {"manifest": CANONICAL_INPUTS, "subset": "lifecycle",
                       "n_ids": 100, "ids_frozen": True, "resample": False,
                       "smoke_uses_first": 5},
        "models": [m[0] for m in MODELS],
        "matrix": {
            "dtypes": ["float32", "bfloat16"],
            "model_dtype_and_cache_dtype_matched": True,
            "config_conditions": ["original", "S_a_refresh"],
            "masks": ["no_pad", "left8", "right8", "left16", "right16"],
            "mask_semantics": {
                "pad_token": "tokenizer.pad_token_id (recorded per model)",
                "left": "prefix, pad*, body; mask 0 on pads; pos=cumsum(mask)-1",
                "right": "prefix, body, pad*; pos=arange; pads stay inside history; "
                         "continuation positions continue after padded history",
                "continuation": "6 reserved real body tokens consumed as 2,3,1; "
                                "always appended after the history, never padded",
            },
            "segments": [{"name": s[0], "index": s[1], "len": s[2], "offset": s[3]}
                         for s in SEGMENTS],
            "prefix": {"text": PREFIX_TEXT, "exact_tokens": PREFIX_LEN,
                       "rule": "tokenizer default call (add_special_tokens=True), "
                               "truncated to exactly 16, asserted per model; "
                               "full pre-truncation ids recorded per model"},
            "body": {"max_tokens": MAX_BODY, "min_tokens": MIN_BODY,
                     "shorter": "error/not_run retained, never replaced"},
            "planned_keys_per_id": 120, "planned_total": 12000,
            "keys_are_not_independent_requests": True,
        },
        "official_interface": {
            "modules": ["kvcloak.KVCloak", "get_kvcloak_config.get_kvcloak_config"],
            "constructor": "KVCloak(cfg, dtype=model dtype, fused=False, "
                           "need_ratio=False, add_a=True)",
            "note": "fused=False means M is applied inside the operator; this is "
                    "NOT the weight-fused experiment and must not be described as such",
            "theta": "build_theta(model fp32, run_suite.NATURAL, 4 threads), computed "
                     "once in prepare-fixtures, stored, hash-bound, loaded by workers",
        },
        "rng_control": {
            "base_config_seed": BASE_SEED,
            "request_seed": "int(sha256(sid|model_short)[:16],16); shared by all "
                            "dtype and mask variants of the same (model, sid) "
                            "(declared); no Python hash",
            "m_fix": "S_a_refresh regenerates under request_seed then restores "
                     "M_angles/M_ratios/M_ratio/theta from baseline per "
                     "layer/head/K,V branch and asserts equality; S and a "
                     "asserted changed; S_ratios is the constant all-ones "
                     "vector under S_ratio=1.0 (need_ratio=False), verified "
                     "constant and recorded; checks saved per request",
            "prep_seed": PREP_SEED,
            "prep_rule": "both cloak instances finish device-tensor initialization "
                         "(A-matrix row draws) under prep_seed BEFORE the shuffle "
                         "seed is set",
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_rule": "seed set immediately before each obfuscate; per-block "
                            "row permutations recorded via declared torch.rand "
                            "recorder (values unmodified) and fingerprint-compared "
                            "across the two conditions",
        },
        "comparison": {
            "primary": "protected_cached vs plain_cached, same model+input+mask+"
                       "position+dtype, whole [seg_len, vocab] logits",
            "secondary": "plain_cached vs full_forward (timestep-aligned: cached "
                         "segment position j vs full logits[cont_start+seg_off+j])",
            "sanity": "two independent plain-cache copies must agree exactly "
                      "(sanity_max == 0) to exclude indexing/state pollution",
            "diffs_computed_in": "float32",
            "config_lifetime": "one config per request-key; its three segments "
                               "share one protect/consume lifecycle",
        },
        "thresholds": {
            "functional_pass_per_segment": {
                "float32": {"prot_vs_plain_max<=": THRESHOLDS["float32"]},
                "bfloat16": {"prot_vs_plain_max<=": THRESHOLDS["bfloat16"]},
            },
            "plus": ["prot_vs_plain_argmax_all", "shape_match", "finite"],
            "negative_results": "valid functional failures are retained, never "
                                "retried away; thresholds never tuned per sample",
        },
        "workers_default": {"processes": 4, "threads": THREADS,
                            "note": "default 4 processes x 4 threads; the launcher "
                                    "may shard each (model, dtype, condition) job "
                                    "over further ID ranges (--workers > 8) to use "
                                    "more cores; actual process/thread counts and "
                                    "shard layout recorded in launcher_run.json "
                                    "and environment.json; deviations from the "
                                    "default documented in HANDOFF_FOR_CODEX.md"},
        "deliverables": ["PROTOCOL.json (this file, written before measurement)",
                         "inputs.jsonl", "environment*.json", "fixtures/",
                         "workers/*/checks.jsonl+requests.jsonl+log.txt",
                         "accounting_validation.json", "summary.json",
                         "source/exp_sa_rotation_verify_v2.py"],
    }
    with open(os.path.join(d, "PROTOCOL.json"), "w") as f:
        json.dump(protocol, f, indent=1, ensure_ascii=False)
    os.makedirs(os.path.join(d, "source"), exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(d, "source"))
    print(f"prepared {d} (PROTOCOL first, no measurement yet)")


# ----------------------------------------------------------------- fixtures
def cmd_prepare_fixtures(args):
    from attack_lib import load_model, build_theta
    from run_suite import NATURAL
    d = args.outdir
    fix_dir = os.path.join(d, "fixtures")
    os.makedirs(fix_dir, exist_ok=True)
    inputs = [json.loads(l) for l in open(os.path.join(d, "inputs.jsonl"))]
    n_ids = 5 if json.load(open(os.path.join(d, "PROTOCOL.json")))["mode"].startswith("smoke") else len(inputs)
    inputs = inputs[:n_ids]
    env_all = {"created": now(), "prepared_by": "prepare-fixtures", "models": {}}
    index = []
    for full_name, short in MODELS:
        torch.set_num_threads(THREADS)
        tok, model = load_model(full_name, threads=THREADS)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(full_name)
        env_all["models"][short] = {
            "name": full_name, "revision": cfg._commit_hash,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "threads": torch.get_num_threads(),
            "tokenizer_class": tok.__class__.__name__,
            "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
            "special_tokens_map": {k: v for k, v in (tok.special_tokens_map or {}).items()
                                   if isinstance(v, str)},
        }
        theta = build_theta(model, tok, NATURAL)
        theta_path = os.path.join(fix_dir, f"theta_{short}.json")
        with open(theta_path, "w") as f:
            json.dump(theta, f)
        prefix16, prefix_full = make_prefix_ids(tok)
        assert len(prefix16) == PREFIX_LEN
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for s in inputs:
            body = tok(s["text"], return_tensors="pt").input_ids[0][:MAX_BODY].tolist()
            eligible = len(body) >= MIN_BODY
            for mask_name, pad_side, pad_len in MASKS:
                if eligible:
                    ids, mask, pos, cont = build_fixture(prefix16, body, pad_side, pad_len, pad_id)
                    digest = fixture_digest(ids, mask, pos, cont)
                    rec = {"sid": s["sid"], "model": short, "mask": mask_name,
                           "eligible": True,
                           "prefix_ids": prefix16, "prefix_full_ids": prefix_full,
                           "body_tokens": len(body),
                           "ids": ids.tolist(), "mask": mask.tolist(),
                           "pos": pos.tolist(), "cont": cont.tolist(),
                           "hist_len": int(ids.shape[0]), "cont_len": int(cont.shape[0]),
                           "pad_token_id": pad_id, "input_sha256": digest}
                else:
                    rec = {"sid": s["sid"], "model": short, "mask": mask_name,
                           "eligible": False, "reason": f"body {len(body)} < {MIN_BODY}",
                           "prefix_ids": prefix16, "prefix_full_ids": prefix_full,
                           "body_tokens": len(body), "pad_token_id": pad_id}
                fp = os.path.join(fix_dir, f"base_{short}_{mask_name}_{s['sid']}.json")
                with open(fp, "w") as f:
                    json.dump(rec, f)
                index.append({"file": os.path.basename(fp), "sid": s["sid"],
                              "model": short, "mask": mask_name,
                              "eligible": rec["eligible"],
                              "input_sha256": rec.get("input_sha256")})
        del model
    with open(os.path.join(fix_dir, "index.json"), "w") as f:
        json.dump({"n_fixtures": len(index), "entries": index}, f)
    with open(os.path.join(d, "environment_prepare.json"), "w") as f:
        json.dump(env_all, f, indent=1)
    print(f"fixtures: {len(index)} base fixtures, {len(MODELS)} thetas, env bound")


# ------------------------------------------------------------------- worker
def cmd_worker(args):
    from attack_lib import load_model, model_dims
    d = args.outdir
    model_short, dtype_name, condition = args.model, args.dtype, args.condition
    dtype = dict(DTYPES)[dtype_name]
    full_name = dict((s, f) for f, s in MODELS)[model_short]
    shard = _parse_shard(getattr(args, "shard", None))  # (i, n) or None
    shard_tag = f"_s{shard[0]}of{shard[1]}" if shard else ""
    wdir = os.path.join(d, "workers", f"{model_short}_{dtype_name}_{condition}{shard_tag}")
    os.makedirs(wdir, exist_ok=True)
    checks_path = os.path.join(wdir, "checks.jsonl")
    requests_path = os.path.join(wdir, "requests.jsonl")
    log = open(os.path.join(wdir, "log.txt"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(f"[{now()}] {msg}\n")
        log.flush()

    proto = json.load(open(os.path.join(d, "PROTOCOL.json")))
    smoke = proto["mode"].startswith("smoke")
    inputs = [json.loads(l) for l in open(os.path.join(d, "inputs.jsonl"))]
    inputs = inputs[:5] if smoke else inputs
    if shard:
        i, n = shard
        inputs = inputs[i::n]
        say(f"shard {i}/{n}: {len(inputs)} ids")

    # resume: skip request-keys whose three segment keys are already present
    done_keys = set()
    attempt = 0
    if os.path.exists(checks_path):
        for line in open(checks_path):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                attempt += 1
                continue
            try:
                done_keys.add(json.loads(line)["key"])
            except Exception:
                pass
    attempt += 1

    torch.set_num_threads(THREADS)
    tok, model = load_model(full_name, threads=THREADS)
    model = model.to(dtype)
    layers, H, D_head = model_dims(model)
    theta = json.load(open(os.path.join(d, "fixtures", f"theta_{model_short}.json")))

    env = {"worker": f"{model_short}_{dtype_name}_{condition}", "pid": os.getpid(),
           "started": now(), "model": full_name,
           "model_revision": __import__("transformers").AutoConfig.from_pretrained(
               full_name)._commit_hash,
           "torch": torch.__version__,
           "transformers": __import__("transformers").__version__,
           "threads": torch.get_num_threads(), "dtype": dtype_name,
           "condition": condition, "attempt": attempt,
           "resumed_keys": len(done_keys)}
    with open(os.path.join(wdir, "environment.json"), "w") as f:
        json.dump(env, f, indent=1)

    base_cfg = build_base_config(layers, H, D_head, theta)
    base_fp = config_fingerprint(base_cfg)
    say(f"start attempt={attempt} resumed={len(done_keys)} base_fp={base_fp[:16]}")

    fix_dir = os.path.join(d, "fixtures")
    out_c = open(checks_path, "a")
    out_r = open(requests_path, "a")
    out_c.write(f"#run attempt={attempt} started={now()} pid={os.getpid()}\n")

    counts = {"measured": 0, "error": 0, "not_run": 0, "skipped_resume": 0}
    t0 = time.time()
    for s in inputs:
        sid = s["sid"]
        for mask_name, pad_side, pad_len in MASKS:
            key0 = f"{sid}|{model_short}|{dtype_name}|{condition}|{mask_name}"
            seg_keys = [f"{key0}|{sg[0]}" for sg in SEGMENTS]
            if all(k in done_keys for k in seg_keys):
                counts["skipped_resume"] += 3
                continue

            def emit(rec):
                out_c.write(json.dumps(rec, ensure_ascii=False) + "\n")

            fix = json.load(open(os.path.join(fix_dir, f"base_{model_short}_{mask_name}_{sid}.json")))
            if not fix.get("eligible"):
                reason = fix.get("reason", "ineligible")
                for sg in SEGMENTS:
                    emit({"key": f"{key0}|{sg[0]}", "sid": sid, "model": model_short,
                          "dtype": dtype_name, "condition": condition, "mask": mask_name,
                          "seg": sg[0], "seg_index": sg[1], "seg_len_declared": sg[2],
                          "status": "not_run", "reason": reason, "attempt": attempt})
                    counts["not_run"] += 1
                out_r.write(json.dumps({"key": key0, "status": "not_run",
                                        "reason": reason, "attempt": attempt}) + "\n")
                continue
            fixture = {"ids": torch.tensor(fix["ids"]), "mask": torch.tensor(fix["mask"]),
                       "pos": torch.tensor(fix["pos"]), "cont": torch.tensor(fix["cont"]),
                       "digest": fix["input_sha256"]}
            assert fixture["ids"].shape[0] == fix["hist_len"]

            try:
                if condition == "original":
                    cfg, m_fixed, s_changed = base_cfg, {"all": True}, {"all": True}
                    cfg_fp = base_fp
                else:
                    seed = request_seed_for(sid, model_short)
                    cfg, m_fixed, s_changed = build_refresh_config(
                        base_cfg, layers, H, D_head, theta, seed)
                    cfg_fp = config_fingerprint(cfg)
                    if not (m_fixed["all"] and s_changed["all"]):
                        raise AssertionError(f"config control failed: {m_fixed} {s_changed}")

                from kvcloak import KVCloak
                cloak = KVCloak(cfg, dtype=dtype, fused=False, need_ratio=False, add_a=True)
                # paired init (fix 6): finish device-tensor init under prep seed
                torch.manual_seed(PREP_SEED)
                cloak._prepare_device_tensors(torch.device("cpu"))

                holder = {}

                def writer(segs):
                    holder["segs"] = segs

                seg_recs, req = run_request(model, cloak, fixture, writer)
                thr = THRESHOLDS[dtype_name]
                for r in seg_recs:
                    r.update({"key": f"{key0}|{r['seg']}", "sid": sid,
                              "model": model_short, "dtype": dtype_name,
                              "condition": condition, "mask": mask_name,
                              "status": "measured", "attempt": attempt,
                              "input_sha256": fixture["digest"],
                              "functional_pass": bool(
                                  r["prot_vs_plain_max"] <= thr
                                  and r["prot_vs_plain_argmax_all"]
                                  and r["shape_match"] and r["finite"])})
                    emit(r)
                    counts["measured"] += 1
                req.update({"key": key0, "sid": sid, "model": model_short,
                            "dtype": dtype_name, "condition": condition,
                            "mask": mask_name, "status": "measured",
                            "attempt": attempt,
                            "request_seed": None if condition == "original"
                            else request_seed_for(sid, model_short),
                            "config_fingerprint": cfg_fp,
                            "base_config_fingerprint": base_fp,
                            "m_fixed": m_fixed, "s_a_changed": s_changed})
                out_r.write(json.dumps(req, ensure_ascii=False) + "\n")
            except Exception as e:
                tb = traceback.format_exc(limit=6)
                say(f"ERROR {key0}: {type(e).__name__}: {e}")
                log.write(tb + "\n")
                for sg in SEGMENTS:
                    emit({"key": f"{key0}|{sg[0]}", "sid": sid, "model": model_short,
                          "dtype": dtype_name, "condition": condition, "mask": mask_name,
                          "seg": sg[0], "seg_index": sg[1], "seg_len_declared": sg[2],
                          "status": "error", "error_type": type(e).__name__,
                          "error": str(e)[:400], "attempt": attempt})
                    counts["error"] += 1
                out_r.write(json.dumps({"key": key0, "status": "error",
                                        "error_type": type(e).__name__,
                                        "error": str(e)[:400], "attempt": attempt},
                                       ensure_ascii=False) + "\n")
            if (counts["measured"] + counts["error"] + counts["not_run"]) % 45 == 0:
                out_c.flush()
                out_r.flush()
                say(f"progress m={counts['measured']} e={counts['error']} "
                    f"nr={counts['not_run']} skip={counts['skipped_resume']} "
                    f"elapsed={time.time()-t0:.0f}s")
    out_c.close()
    out_r.close()
    env["finished"] = now()
    env["counts"] = counts
    with open(os.path.join(wdir, "environment.json"), "w") as f:
        json.dump(env, f, indent=1)
    say(f"done {counts} in {time.time()-t0:.0f}s")


# -------------------------------------------------------------------- merge
def expected_keys(d):
    proto = json.load(open(os.path.join(d, "PROTOCOL.json")))
    inputs = [json.loads(l) for l in open(os.path.join(d, "inputs.jsonl"))]
    n = 5 if proto["mode"].startswith("smoke") else len(inputs)
    keys = set()
    for s in inputs[:n]:
        for _, ms in MODELS:
            for dn, _ in DTYPES:
                for cond in CONDITIONS:
                    for mn, _, _ in MASKS:
                        for sg, _, _, _ in SEGMENTS:
                            keys.add(f"{s['sid']}|{ms}|{dn}|{cond}|{mn}|{sg}")
    return keys


def load_shards(d):
    seg_recs, req_recs = {}, {}
    dup = 0
    wroot = os.path.join(d, "workers")
    for w in sorted(os.listdir(wroot)) if os.path.isdir(wroot) else []:
        cpath = os.path.join(wroot, w, "checks.jsonl")
        rpath = os.path.join(wroot, w, "requests.jsonl")
        if os.path.exists(cpath):
            for line in open(cpath):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                r = json.loads(line)
                k = r["key"]
                if k in seg_recs:
                    dup += 1
                    if r.get("attempt", 0) >= seg_recs[k].get("attempt", 0):
                        seg_recs[k] = r
                else:
                    seg_recs[k] = r
        if os.path.exists(rpath):
            for line in open(rpath):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                r = json.loads(line)
                k = r["key"]
                if k in req_recs:
                    if r.get("attempt", 0) >= req_recs[k].get("attempt", 0):
                        req_recs[k] = r
                else:
                    req_recs[k] = r
    return seg_recs, req_recs, dup


def cmd_merge(args):
    d = args.outdir
    exp = expected_keys(d)
    seg, req, dup = load_shards(d)
    got = set(seg)
    by_status = {}
    for r in seg.values():
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    measured = [r for r in seg.values() if r["status"] == "measured"]
    fp = sum(1 for r in measured if r.get("functional_pass"))
    ff = sum(1 for r in measured if not r.get("functional_pass"))
    # pairing / control validation across conditions (uses request records)
    pair_ok, pair_bad, mf_bad, inbad = 0, [], [], []
    req_by = {r["key"]: r for r in req.values()}
    for k, r in req.items():
        if r.get("status") != "measured":
            continue
        if not r.get("m_fixed", {}).get("all", True):
            mf_bad.append(k)
        other = "S_a_refresh" if k.split("|")[3] == "original" else "original"
        ok = k.replace(f"|{k.split('|')[3]}|", f"|{other}|")
        o = req_by.get(ok)
        if o and o.get("status") == "measured":
            if r.get("perm_hash") and r["perm_hash"] == o.get("perm_hash"):
                pair_ok += 1
            else:
                pair_bad.append([k, ok])
        base = r.get("base_config_fingerprint")
        if base and r.get("config_fingerprint") == base and \
                k.split("|")[3] == "S_a_refresh":
            inbad.append(k)  # refresh must not equal baseline config
    # input identity across dtype/config pairings
    inputs_bad = []
    per_input = {}
    for k, r in req.items():
        if r.get("status") != "measured":
            continue
        parts = k.split("|")
        ik = (parts[0], parts[1], parts[4])
        per_input.setdefault(ik, set()).add(r.get("input_sha256"))
    for ik, hs in per_input.items():
        if len(hs) != 1:
            inputs_bad.append([ik, sorted(hs)])
    accounting = {
        "generated": now(), "planned": len(exp), "records": len(seg),
        "expected_set_equal": got == exp, "missing": sorted(exp - got)[:20],
        "n_missing": len(exp - got), "unexpected": sorted(got - exp)[:20],
        "n_unexpected": len(got - exp), "duplicate_key_lines": dup,
        "by_status": by_status, "functional_pass": fp, "functional_fail": ff,
        "measured": by_status.get("measured", 0),
        "error": by_status.get("error", 0), "not_run": by_status.get("not_run", 0),
        "requests_measured": sum(1 for r in req.values() if r.get("status") == "measured"),
        "requests_total": len(req),
        "perm_pairs_equal": pair_ok, "perm_pairs_unequal": pair_bad[:10],
        "m_fixed_violations": mf_bad[:10], "refresh_equals_baseline": inbad[:10],
        "input_identity_violations": inputs_bad[:10],
        "sanity_nonzero": sum(1 for r in measured if r.get("sanity_max", 0.0) != 0.0),
        "plain_baseline_pollution": sum(1 for r in req.values()
                                        if r.get("status") == "measured"
                                        and r.get("mutation", {}).get("plain_baseline_polluted")),
    }
    with open(os.path.join(d, "accounting_validation.json"), "w") as f:
        json.dump(accounting, f, indent=1)

    # summary aggregates by cell
    summary = {"generated": now(), "cells": {}}
    for r in measured:
        ck = f"{r['model']}|{r['dtype']}|{r['condition']}|{r['mask']}"
        c = summary["cells"].setdefault(ck, {"n": 0, "pass": 0,
                                             "max": 0.0, "sanity_bad": 0})
        c["n"] += 1
        c["pass"] += 1 if r.get("functional_pass") else 0
        c["max"] = max(c["max"], r.get("prot_vs_plain_max", 0.0))
        c["sanity_bad"] += 1 if r.get("sanity_max", 0.0) != 0.0 else 0
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(accounting, indent=1)[:3000])


def cmd_gates(args):
    """Smoke gates: control variables, input alignment, timestep alignment,
    state trajectory, plain-cache sanity, record completeness."""
    d = args.outdir
    proto = json.load(open(os.path.join(d, "PROTOCOL.json")))
    seg, req, dup = load_shards(d)
    exp = expected_keys(d)
    measured = [r for r in seg.values() if r["status"] == "measured"]
    fp32 = [r for r in measured if r["dtype"] == "float32"]
    gates = {}

    gates["record_completeness"] = {
        "pass": set(seg) == exp and dup == 0,
        "records": len(seg), "planned": len(exp),
        "set_equal": set(seg) == exp, "duplicates": dup,
        "by_status": {s: sum(1 for r in seg.values() if r["status"] == s)
                      for s in {r["status"] for r in seg.values()}},
    }
    mreq = [r for r in req.values() if r.get("status") == "measured"]
    gates["control_variables"] = {
        "pass": all(r.get("m_fixed", {}).get("all") for r in mreq)
        and all(r.get("s_a_changed", {}).get("all") for r in mreq
                if r["condition"] == "S_a_refresh")
        and all(r.get("config_fingerprint") == r.get("base_config_fingerprint")
                for r in mreq if r["condition"] == "original"),
        "requests": len(mreq),
        "m_fixed_violations": [r["key"] for r in mreq if not r.get("m_fixed", {}).get("all")][:5],
        "s_a_change_violations": [r["key"] for r in mreq
                                  if r["condition"] == "S_a_refresh"
                                  and not r.get("s_a_changed", {}).get("all")][:5],
        "base_fingerprint_unique": len({r.get("base_config_fingerprint") for r in mreq
                                        if r.get("base_config_fingerprint")}),
    }
    fix_index = json.load(open(os.path.join(d, "fixtures", "index.json")))
    fx = json.load(open(os.path.join(d, "fixtures", fix_index["entries"][0]["file"]))) \
        if fix_index["entries"] else {}
    gates["input_alignment"] = {
        "pass": fix_index["n_fixtures"] == len(fix_index["entries"])
        and all(len(json.load(open(os.path.join(d, "fixtures", e["file"])))
                    .get("prefix_ids", [])) == PREFIX_LEN
                for e in fix_index["entries"][:50])
        and all(r.get("seg_len_actual") == r.get("seg_len_declared") for r in measured)
        and all(r.get("cont_len") == 6 for r in mreq),
        "n_fixtures": fix_index["entries"] and len(fix_index["entries"]),
        "prefix_len_sample": len(fx.get("prefix_ids", [])),
        "seg_len_mismatches": sum(1 for r in measured
                                  if r.get("seg_len_actual") != r.get("seg_len_declared")),
        "input_identity": len({(r["key"].split("|")[0], r["key"].split("|")[1],
                                r["key"].split("|")[4], r.get("input_sha256"))
                               for r in mreq}),
    }
    gates["timestep_alignment"] = {
        "pass": all(r.get("shape_match") for r in measured)
        and all(r.get("plain_vs_full_max", 1.0) <= 1e-3 for r in fp32),
        "shape_mismatches": sum(1 for r in measured if not r.get("shape_match")),
        "fp32_plain_vs_full_max": max((r.get("plain_vs_full_max", 0.0) for r in fp32),
                                      default=0.0),
        "fp32_threshold": 1e-3,
    }
    gates["state_trajectory"] = {
        "pass": all(r.get("sanity_max", 0.0) == 0.0 for r in measured)
        and sorted({r["seg"] for r in measured}) == [s[0] for s in SEGMENTS],
        "sanity_nonzero": sum(1 for r in measured if r.get("sanity_max", 0.0) != 0.0),
        "sanity_max": max((r.get("sanity_max", 0.0) for r in measured), default=0.0),
        "segments_present": sorted({r["seg"] for r in measured}),
    }
    gates["plain_cache_sanity"] = {
        "pass": all(not r.get("mutation", {}).get("plain_baseline_polluted")
                    for r in mreq)
        and all(r.get("sanity_max", 0.0) == 0.0 for r in measured),
        "polluted_requests": sum(1 for r in mreq
                                 if r.get("mutation", {}).get("plain_baseline_polluted")),
        "mutation_entries_expected": "num_layers x 2 (K,V): Qwen 24x2=48, Llama 16x2=32",
        "mutation_entries_ok": all(
            r.get("mutation", {}).get("n_entries") in (24 * 2, 16 * 2)
            for r in mreq),
    }
    pair_bad = []
    req_by = {r["key"]: r for r in mreq}
    for k, r in req_by.items():
        parts = k.split("|")
        other = "S_a_refresh" if parts[3] == "original" else "original"
        o = req_by.get(k.replace(f"|{parts[3]}|", f"|{other}|"))
        if o and r.get("perm_hash") != o.get("perm_hash"):
            pair_bad.append(k)
    gates["paired_shuffle"] = {
        "pass": not pair_bad,
        "pairs_checked": sum(1 for k in req_by
                             if req_by.get(k.replace(
                                 f"|{k.split('|')[3]}|",
                                 f"|{'S_a_refresh' if k.split('|')[3] == 'original' else 'original'}|"))),
        "unequal": pair_bad[:5],
    }
    gates["_all_pass"] = all(v["pass"] for k, v in gates.items() if k != "_all_pass")
    with open(os.path.join(d, "smoke_gates.json"), "w") as f:
        json.dump(gates, f, indent=1)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v)
                      for k, v in gates.items()}, indent=1))


# ------------------------------------------------------------------ launcher
def cmd_launch(args):
    d = args.outdir
    if not os.path.exists(os.path.join(d, "PROTOCOL.json")):
        raise SystemExit("run --prepare-outdir first")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS=str(THREADS))
    cmd_prepare_fixtures(args)
    jobs = [(m, dn, c) for (_, m) in MODELS for (dn, _) in DTYPES for c in CONDITIONS]
    n_jobs = len(jobs)
    shards_per_job = 1
    if args.workers and args.workers > n_jobs:
        shards_per_job = min(args.workers // n_jobs, 25)
    procs = []
    os.makedirs(os.path.join(d, "launcher_logs"), exist_ok=True)
    t0 = now()
    for (m, dn, c) in jobs:
        for si in range(shards_per_job):
            tag = f"{m}_{dn}_{c}" + (f"_s{si}of{shards_per_job}" if shards_per_job > 1 else "")
            lg = open(os.path.join(d, "launcher_logs", f"{tag}.log"), "a")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--outdir", d, "--worker", m, dn, c]
            if shards_per_job > 1:
                cmd += ["--shard", f"{si}/{shards_per_job}"]
            p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env,
                                 cwd=REPO)
            procs.append((p, tag))
            print(f"launched worker {tag} pid={p.pid}", flush=True)
    rc = {}
    for p, name in procs:
        code = p.wait()
        rc[name] = code
        print(f"worker {name} exit={code}", flush=True)
    with open(os.path.join(d, "launcher_run.json"), "w") as f:
        json.dump({"started": t0, "finished": now(), "workers": len(procs),
                   "shards_per_job": shards_per_job,
                   "threads_per_worker": THREADS, "exit_codes": rc}, f, indent=1)
    bad = {k: v for k, v in rc.items() if v != 0}
    if bad:
        print(f"NONZERO EXITS: {bad}; run --worker ... to resume, then --merge")
        sys.exit(2)
    args.merge = True
    cmd_merge(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prepare-outdir", action="store_true")
    ap.add_argument("--prepare-fixtures", action="store_true")
    ap.add_argument("--worker", nargs=3, metavar=("MODEL", "DTYPE", "CONDITION"))
    ap.add_argument("--shard", default=None, help="i/n ID-range shard for this worker")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.prepare_outdir:
        cmd_prepare_outdir(args)
    elif args.prepare_fixtures:
        cmd_prepare_fixtures(args)
    elif args.worker:
        args.model, args.dtype, args.condition = args.worker
        cmd_worker(args)
    elif args.gates:
        cmd_gates(args)
    elif args.merge:
        cmd_merge(args)
    elif args.launch:
        cmd_launch(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
