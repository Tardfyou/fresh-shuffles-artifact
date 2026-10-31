"""VeilKV deployment-semantics measurement (comment task 7).

Legal known-key consumer only: the test program holds every configuration
and calls the delivered defense codec's public interfaces
(protect_cache / recover_cache / veilkv_runtime.recover_for_model /
corrected.freshness.KeyService). No attacker-visibility claims are made.

Conditions (see PROTOCOL.json task7):
  multi_turn     100 IDs x 2 models x 2 dtypes x 3 segments: one TEE-profile
                 protect at prefill (epoch 3, reproducible principal/nonce),
                 authenticated recovery, 2/3/1 continuation through plain and
                 protected branches, whole-tensor same-timestep logits.
  shared_prefix  100 IDs x 2 models (fp32): same policy-public 16-token
                 prefix, two principals (ID i and ID i+1 tails, equal length);
                 artifact reuse, public-object identity, per-principal
                 recovery equality, cross-principal and prefix-swap rejection.
  reencode       100 IDs x 2 models (fp32): recover -> re-protect under a
                 fresh nonce and epoch 4 (npb=0, declared policy) -> recover,
                 consume one more segment; old-epoch rejection, duplicate
                 nonce rejection.
  ledger         scripted scenarios on a dedicated sqlite freshness DB:
                 restart persistence, public-object consistency, concurrent
                 claims; event order and exit codes recorded.
  device         actual hardware probe; no GPU -> not_available.
  scheduler      repo/release searched for a real serving adapter; none ->
                 not_implemented (toy PageManager not substituted).

Inputs: frozen lifecycle texts (defense_handoff_20261030_1600/inputs.jsonl)
and the accepted v2 fixtures (prefix16/body134/cont6, no_pad) are reused
without resampling. Freshness DB is per-run (path recorded); nonce derivation
is reproducible from (sid, model, dtype, condition, run_token) with run_token
recorded in launcher_run.json.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "source", "defense_runtime"))
sys.path.insert(0, os.path.join(HERE, "source"))

from corrected.freshness import KeyService, FreshnessError  # noqa: E402
import exp_veilkv_plus as codec  # noqa: E402
from veilkv_runtime import recover_for_model  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

MODELS = [("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("unsloth/Llama-3.2-1B", "Llama-3.2-1B")]
DTYPES = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]
SEGMENTS = [("seg1", 1, 2, 0), ("seg2", 2, 3, 2), ("seg3", 3, 1, 5)]
V2_FIXTURES = ("results/takeover_20261029/"
               "corrected/defense_handoff_v2_20261031_1319/fixtures")
INPUTS = ("results/takeover_20261029/"
          "defense_handoff_20261030_1600/inputs.jsonl")
THREADS = 4
NPB = 1  # one policy-public 16-token prefix block


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def nonce_for(run_token, *parts):
    return hashlib.sha256("|".join([run_token] + [str(p) for p in parts])
                          .encode()).digest()[:16]


def load_model_pair(full_name, dtype):
    tok = AutoTokenizer.from_pretrained(full_name)
    model = AutoModelForCausalLM.from_pretrained(full_name, torch_dtype=torch.float32)
    model = model.to(dtype)
    model.eval()
    return tok, model


def clone_legacy(tc):
    return tuple((k.clone(), v.clone()) for k, v in tc)


def fresh_dynamic(tc):
    return DynamicCache.from_legacy_cache(tuple((k, v) for k, v in tc))


def bytes_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    u = torch.uint8
    return torch.equal(x.view(u) if x.dtype != u else x, y.view(u) if y.dtype != u else y)


def mk_ctx(model, tok, ids, profile, sdt, principal, nonce, epoch, npb, true_len):
    """Build the protection context the same way the retained harness did; the
    delivered codec exposes only protect/recover/_digests, so the context is
    assembled here (fields verified against protect_cache/recover_cache)."""
    d = codec._digests(model, tok)
    cfg = model.config
    return dict(model_dig=d["model_dig"], tok_dig=d["tok_dig"], rope=d["rope"],
                tokens=ids.unsqueeze(0), n_prefix_blocks=npb, profile=profile,
                storage_dtype=sdt, principal=principal, nonce=nonce,
                epoch_id=epoch, variant={}, bsz=1,
                H=cfg.num_key_value_heads,
                D=getattr(cfg, "head_dim", None)
                or (cfg.hidden_size // cfg.num_attention_heads),
                true_len=true_len)


def cache_mismatch(tc1, tc2):
    bad = []
    for li in range(min(len(tc1), len(tc2))):
        for kv, name in ((0, "K"), (1, "V")):
            if not bytes_equal(tc1[li][kv], tc2[li][kv]):
                bad.append(f"{li}:{name}")
    if len(tc1) != len(tc2):
        bad.append(f"layers {len(tc1)}!={len(tc2)}")
    return bad


# ------------------------------------------------------------- multi_turn
def run_multi_turn(model, tok, service, run_token, sid, model_short, dtype_name, dtype, fix):
    ids = torch.tensor(fix["ids"])
    mask = torch.tensor(fix["mask"])
    pos = torch.tensor(fix["pos"])
    cont = torch.tensor(fix["cont"])
    L = int(ids.shape[0])
    principal = f"principal-{sid}".encode()
    nonce = nonce_for(run_token, sid, model_short, dtype_name, "multi_turn")

    total_ids = torch.cat([ids, cont])
    total_mask = torch.cat([mask, torch.ones(len(cont), dtype=torch.long)])
    total_pos = torch.cat([pos, torch.arange(int(pos[-1]) + 1,
                                             int(pos[-1]) + 1 + len(cont))])
    with torch.no_grad():
        full = model(input_ids=total_ids.unsqueeze(0),
                     attention_mask=total_mask.unsqueeze(0),
                     position_ids=total_pos.unsqueeze(0), use_cache=False)
    full_logits = full.logits[0]

    with torch.no_grad():
        pre = model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0),
                    position_ids=pos.unsqueeze(0), use_cache=True)
    src = clone_legacy(pre.past_key_values.to_legacy_cache())
    plain_snapshot = clone_legacy(src)

    ctx = mk_ctx(model, tok, ids, "tee", dtype, principal, nonce, codec.EPOCH, NPB, L)
    ctx["key_service"] = service
    prot = codec.protect_cache(src, ctx)
    rec = tuple(recover_for_model(prot, ctx, model_dtype=dtype))

    plain_a, plain_b, rec_c = clone_legacy(src), clone_legacy(src), clone_legacy(rec)
    segs = []
    for seg_name, seg_idx, seg_len, seg_off in SEGMENTS:
        seg = cont[seg_off:seg_off + seg_len]
        seg_pos = torch.arange(int(pos[-1]) + 1 + seg_off,
                               int(pos[-1]) + 1 + seg_off + seg_len)
        hist_mask = torch.cat([mask, torch.ones(seg_off + seg_len, dtype=torch.long)])
        kw = dict(input_ids=seg.unsqueeze(0), attention_mask=hist_mask.unsqueeze(0),
                  position_ids=seg_pos.unsqueeze(0), use_cache=True)
        with torch.no_grad():
            oa = model(**kw, past_key_values=fresh_dynamic(plain_a))
            ob = model(**kw, past_key_values=fresh_dynamic(plain_b))
            oc = model(**kw, past_key_values=fresh_dynamic(rec_c))
        la, lb, lc = oa.logits[0], ob.logits[0], oc.logits[0]
        f_log = full_logits[L + seg_off: L + seg_off + seg_len]

        def d(x, y):
            t = x.float() - y.float()
            return float(t.abs().max()), float(t.abs().mean())

        pp_max, pp_mean = d(lc, la)
        segs.append({
            "seg": seg_name, "seg_index": seg_idx,
            "seg_len_declared": seg_len, "seg_len_actual": int(seg.shape[0]),
            "seg_abs_range": [L + seg_off, L + seg_off + seg_len],
            "sanity_max": d(lb, la)[0],
            "prot_vs_plain_max": pp_max, "prot_vs_plain_mean": pp_mean,
            "prot_vs_plain_bitexact": bool(la.shape == lc.shape and torch.equal(la, lc)),
            "prot_vs_plain_argmax_all": bool(torch.all(lc.argmax(-1) == la.argmax(-1))),
            "plain_vs_full_max": d(la, f_log)[0],
            "shape_match": bool(la.shape == lc.shape == f_log.shape),
            "logits_dtype": str(lc.dtype), "finite": bool(torch.isfinite(lc).all()),
            "seg_tokens": seg.tolist(),
        })
        plain_a = clone_legacy(oa.past_key_values.to_legacy_cache())
        plain_b = clone_legacy(ob.past_key_values.to_legacy_cache())
        rec_c = clone_legacy(oc.past_key_values.to_legacy_cache())

    request = {
        "hist_len": L, "cont_len": int(cont.shape[0]), "epoch": codec.EPOCH,
        "principal": principal.decode(), "nonce_hex": nonce.hex(),
        "n_prefix_blocks": NPB, "profile": "tee", "storage_dtype": dtype_name,
        "prefix_artifact_id": prot.blocks[(0, 0)][0]["artifact_id"][:16].hex(),
        "plain_baseline_polluted": cache_mismatch(plain_snapshot, src),
        "recovered_equals_plain_bitexact": not cache_mismatch(rec, src),
        "cache_versions": {"seg3_plain_len": plain_a[0][1].shape[2],
                           "seg3_protected_len": rec_c[0][1].shape[2]},
    }
    return segs, request


# ---------------------------------------------------------- shared_prefix
def run_shared_prefix(model, tok, service, run_token, sid_a, sid_b, model_short, tails):
    prefix = tails["prefix"]
    La = len(prefix) + len(tails[sid_a])
    ids_a = torch.tensor(prefix + tails[sid_a])
    ids_b = torch.tensor(prefix + tails[sid_b])
    with torch.no_grad():
        ca = model(input_ids=ids_a.unsqueeze(0), use_cache=True).past_key_values.to_legacy_cache()
        cb = model(input_ids=ids_b.unsqueeze(0), use_cache=True).past_key_values.to_legacy_cache()
    ca, cb = clone_legacy(ca), clone_legacy(cb)
    na = nonce_for(run_token, sid_a, model_short, "shared_A")
    nb = nonce_for(run_token, sid_b, model_short, "shared_B")
    pa, pb = f"principal-{sid_a}".encode(), f"principal-{sid_b}".encode()
    ctx_a = mk_ctx(model, tok, ids_a, "tee", torch.float32, pa, na, codec.EPOCH, NPB, La)
    ctx_b = mk_ctx(model, tok, ids_b, "tee", torch.float32, pb, nb, codec.EPOCH, NPB, La)
    ctx_a["key_service"] = ctx_b["key_service"] = service
    prot_a = codec.protect_cache(ca, ctx_a)
    art_a = [prot_a.blocks[(0, 0)][j]["artifact_id"] for j in range(NPB)]
    # production reuse path: B's first tail block chains from A's prefix tag
    chain_seed = {(li, kvf): prot_a.blocks[(li, kvf)][NPB - 1]["tag"]
                  for li in range(len(ca)) for kvf in (0, 1)}
    prot_b = codec.protect_cache(cb, ctx_b, chain_seed=chain_seed)
    art_b = [prot_b.blocks[(0, 0)][j]["artifact_id"] for j in range(NPB)]

    rec_a = codec.recover_cache(prot_a, ctx_a)
    rec_b = codec.recover_cache(prot_b, ctx_b, chain_seed=chain_seed)
    eq_a = cache_mismatch(rec_a, ca)
    eq_b = cache_mismatch(rec_b, cb)

    cross_rej, cross_err = False, None
    try:
        codec.recover_cache(prot_b, ctx_a)
    except codec.TamperError as e:
        cross_rej, cross_err = True, str(e)[:80]
    other_prefix = list(prefix)
    other_prefix[-1] = (other_prefix[-1] + 1) % 30000
    ids_c = torch.tensor(other_prefix + tails[sid_b])
    ctx_swap = mk_ctx(model, tok, ids_c, "tee", torch.float32, pb, nb, codec.EPOCH, NPB, La)
    swap_rej, swap_err = False, None
    try:
        codec.recover_cache(prot_b, ctx_swap)
    except codec.TamperError as e:
        swap_rej, swap_err = True, str(e)[:80]
    return {
        "sid_a": sid_a, "sid_b": sid_b, "tail_len": len(tails[sid_a]), "hist_len": La,
        "artifact_ids_equal": art_a == art_b, "artifact_a": art_a[0][:16].hex(),
        "public_object_reuse_ok": True,  # protect(B) reused prefix artifacts without conflict
        "recover_a_equals_plain": not eq_a, "recover_b_equals_plain": not eq_b,
        "cross_principal_rejected": cross_rej, "cross_principal_error": cross_err,
        "prefix_swap_rejected": swap_rej, "prefix_swap_error": swap_err,
        "epoch": codec.EPOCH, "profile": "tee", "storage_dtype": "float32",
        "nonces": [na.hex(), nb.hex()],
    }


# -------------------------------------------------------------- reencode
def run_reencode(model, tok, service, run_token, sid, model_short, fix):
    ids = torch.tensor(fix["ids"])
    mask = torch.tensor(fix["mask"])
    pos = torch.tensor(fix["pos"])
    cont = torch.tensor(fix["cont"])
    L = int(ids.shape[0])
    principal = f"principal-{sid}".encode()
    n1 = nonce_for(run_token, sid, model_short, "reencode_e3")
    n2 = nonce_for(run_token, sid, model_short, "reencode_e4")
    ctx1 = mk_ctx(model, tok, ids, "tee", torch.float32, principal, n1, 3, NPB, L)
    ctx1["key_service"] = service
    with torch.no_grad():
        pre = model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0),
                    position_ids=pos.unsqueeze(0), use_cache=True)
    src = clone_legacy(pre.past_key_values.to_legacy_cache())
    prot1 = codec.protect_cache(src, ctx1)
    rec1 = codec.recover_cache(prot1, ctx1)
    seg = cont[:2]
    seg_pos = torch.arange(int(pos[-1]) + 1, int(pos[-1]) + 3)
    hm = torch.cat([mask, torch.ones(2, dtype=torch.long)])
    kw = dict(input_ids=seg.unsqueeze(0), attention_mask=hm.unsqueeze(0),
              position_ids=seg_pos.unsqueeze(0), use_cache=True)
    with torch.no_grad():
        oa = model(**kw, past_key_values=fresh_dynamic(clone_legacy(src)))
        oc = model(**kw, past_key_values=fresh_dynamic(clone_legacy(rec1)))
    logits_bitexact = bool(torch.equal(oa.logits[0], oc.logits[0]))
    consumed_plain = clone_legacy(oa.past_key_values.to_legacy_cache())
    consumed_rec = clone_legacy(oc.past_key_values.to_legacy_cache())
    L2 = consumed_rec[0][1].shape[2]
    ctx2 = mk_ctx(model, tok, torch.cat([ids, seg]), "tee", torch.float32,
                  principal, n2, 4, 0, L2)
    ctx2["key_service"] = service
    prot2 = codec.protect_cache(consumed_rec, ctx2)
    rec2 = codec.recover_cache(prot2, ctx2)
    reenc_eq = cache_mismatch(rec2, consumed_plain)
    old_rej, old_err = False, None
    try:
        codec.recover_cache(prot1, ctx2)
    except codec.TamperError as e:
        old_rej, old_err = True, str(e)[:80]
    dup_rej, dup_err = False, None
    try:
        codec.protect_cache(src, ctx1)
    except FreshnessError as e:
        dup_rej, dup_err = True, str(e)[:80]
    return {
        "hist_len": L, "seg1_bitexact_logits": logits_bitexact,
        "reprotected_under": {"epoch": 4, "npb": 0, "nonce_hex": n2.hex()},
        "reencoded_equals_plain_bitexact": not reenc_eq,
        "shape_after_reencode": list(rec2[0][1].shape) == list(consumed_plain[0][1].shape),
        "old_epoch_rejected": old_rej, "old_epoch_error": old_err,
        "duplicate_nonce_rejected": dup_rej, "duplicate_nonce_error": dup_err,
        "source_interface_supports_reprotect": True,
    }


# ---------------------------------------------------------------- ledger
def run_ledger(service_path):
    events = []

    def ev(name, ok, detail):
        events.append({"event": name, "ok": bool(ok), "detail": str(detail)[:120]})

    s = KeyService(service_path)
    sc1 = b"scope-nonce-1"
    try:
        s.claim(sc1)
        ev("claim_first", True, "registered")
    except FreshnessError as e:
        ev("claim_first", False, e)
    try:
        s.claim(sc1)
        ev("claim_duplicate", False, "UNEXPECTED-ACCEPT")
    except FreshnessError as e:
        ev("claim_duplicate", True, f"rejected: {e}")
    s2 = KeyService(service_path)  # restart
    try:
        s2.claim(sc1)
        ev("claim_after_restart", False, "UNEXPECTED-ACCEPT")
    except FreshnessError as e:
        ev("claim_after_restart", True, f"persistent reject: {e}")
    sc_art = b"scope-art-1"
    s2.public(sc_art, b"payload-v1")
    ev("public_store", True, "stored")
    s2.public(sc_art, b"payload-v1")
    ev("public_idempotent", True, "same content accepted")
    try:
        s2.public(sc_art, b"payload-v2")
        ev("public_conflict", False, "UNEXPECTED-ACCEPT")
    except FreshnessError as e:
        ev("public_conflict", True, f"rejected: {e}")
    s3 = KeyService(service_path)
    s3.public(sc_art, b"payload-v1")
    ev("public_after_restart", True, "consistent")
    import threading
    results = []

    def claim(sc):
        try:
            s3.claim(sc)
            results.append("ok")
        except FreshnessError:
            results.append("reject")

    t1 = threading.Thread(target=claim, args=(b"scope-conc-A",))
    t2 = threading.Thread(target=claim, args=(b"scope-conc-B",))
    t1.start(); t2.start(); t1.join(timeout=30); t2.join(timeout=30)
    ev("concurrent_distinct", sorted(results) == ["ok", "ok"], results)
    results = []
    t1 = threading.Thread(target=claim, args=(b"scope-conc-same",))
    t2 = threading.Thread(target=claim, args=(b"scope-conc-same",))
    t1.start(); t2.start(); t1.join(timeout=30); t2.join(timeout=30)
    ev("concurrent_same_nonce_exactly_one", sorted(results) == ["ok", "reject"], results)
    return {"events": events,
            "all_expected": all(e["ok"] for e in events),
            "note": "software ledger; not hardware rollback resistance"}


def run_device():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        gpu = out.returncode == 0 and bool(out.stdout.strip())
        detail = (out.stdout.strip()[:200] if gpu else out.stderr.strip()[:200])
    except Exception as e:
        gpu, detail = False, f"nvidia-smi unavailable: {e}"
    return {"gpu_available": gpu, "cuda_available": bool(torch.cuda.is_available()),
            "status": "device_transfer_testable" if gpu else "not_available",
            "detail": detail, "note": "no simulated GPU results are produced"}


def run_scheduler():
    roots = ["experiments/kvcloak",
             "results/takeover_20261029/release/defense-stage-v2pyknb7"]
    hits = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if any(part in dirpath for part in (".venv", "site-packages", "__pycache__")):
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                try:
                    txt = open(os.path.join(dirpath, f), errors="ignore").read()
                except Exception:
                    continue
                if re.search(r"class\s+\w*(Serving|Scheduler)\w*|def\s+serve\b", txt):
                    hits.append(os.path.join(dirpath, f))
    return {"status": "not_implemented" if not hits else "candidate_found",
            "searched": roots, "serving_class_hits": hits,
            "note": "no real serving adapter found; toy PageManager not substituted"}


# ------------------------------------------------------------- fixtures
def load_v2_fixture(model_short, sid):
    return json.load(open(os.path.join(V2_FIXTURES, f"base_{model_short}_no_pad_{sid}.json")))


def prepare_shared_tails(outdir):
    inputs = [json.loads(l) for l in open(INPUTS)]
    sids = [s["sid"] for s in inputs]
    text = {s["sid"]: s["text"] for s in inputs}
    table = {}
    for full, short in MODELS:
        tok = AutoTokenizer.from_pretrained(full)
        prefix = json.load(open(os.path.join(
            V2_FIXTURES, f"base_{short}_no_pad_{sids[0]}.json")))["prefix_ids"]
        entries = []
        for i, sid in enumerate(sids):
            other = sids[(i + 1) % len(sids)]
            ba = tok(text[sid], return_tensors="pt").input_ids[0][:134].tolist()
            bb = tok(text[other], return_tensors="pt").input_ids[0][:134].tolist()
            n = min(len(ba), len(bb))
            if n < 16:
                entries.append({"sid_a": sid, "sid_b": other, "eligible": False,
                                "reason": f"tail {n} < 16"})
                continue
            entries.append({"sid_a": sid, "sid_b": other, "eligible": True,
                            "tail_a": ba[:n], "tail_b": bb[:n], "tail_len": n})
        table[short] = {"prefix": prefix, "entries": entries}
    path = os.path.join(outdir, "shared_tails.json")
    with open(path, "w") as f:
        json.dump(table, f)
    return path


# ------------------------------------------------------------- worker
def _parse_shard(spec):
    if not spec:
        return None
    i, n = (int(x) for x in spec.split("/"))
    assert 0 <= i < n
    return i, n


def cmd_worker(args):
    d = args.outdir
    cond = args.condition
    inputs = [json.loads(l) for l in open(INPUTS)]
    sids = [s["sid"] for s in inputs]
    if args.sanity:
        sids = sids[:5]
    shard = _parse_shard(args.shard)
    run_token = json.load(open(os.path.join(d, "launcher_run.json")))["run_token"]

    wdir = os.path.join(d, "workers", cond + (f"_s{shard[0]}of{shard[1]}" if shard else ""))
    os.makedirs(wdir, exist_ok=True)
    log = open(os.path.join(wdir, "log.txt"), "a")

    def say(m):
        print(m, flush=True)
        log.write(f"[{now()}] {m}\n")
        log.flush()

    torch.set_num_threads(THREADS)
    counts = {"measured": 0, "error": 0, "not_run": 0}
    out = open(os.path.join(wdir, "checks.jsonl"), "a")
    env_path = os.path.join(wdir, "environment.json")

    def write_env(model_short=None, extra=None):
        e = {"condition": cond, "pid": os.getpid(), "started": now(),
             "torch": torch.__version__,
             "transformers": __import__("transformers").__version__,
             "threads": torch.get_num_threads(), "run_token": run_token,
             "shard": args.shard}
        if model_short:
            full = dict((s, f) for f, s in MODELS)[model_short]
            e.update({"model": full, "model_revision":
                      AutoConfig.from_pretrained(full)._commit_hash})
        if extra:
            e.update(extra)
        with open(env_path, "w") as f:
            json.dump(e, f, indent=1)

    def emit(rec):
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        if cond == "ledger":
            write_env(extra={"db": os.path.join(d, "ledger_test.sqlite")})
            r = run_ledger(os.path.join(d, "ledger_test.sqlite"))
            emit({"key": "ledger|scenarios", "condition": "ledger",
                  "status": "measured", **r})
            counts["measured"] += 1
            say(f"ledger: all_expected={r['all_expected']}")
        elif cond == "device":
            write_env()
            r = run_device()
            st = "measured" if r["status"] == "device_transfer_testable" else "not_run"
            emit({"key": "device|probe", "condition": "device", "status": st, **r})
            counts["measured" if st == "measured" else "not_run"] += 1
        elif cond == "scheduler":
            write_env()
            r = run_scheduler()
            st = "measured" if r["status"] == "candidate_found" else "not_run"
            emit({"key": "scheduler|search", "condition": "scheduler", "status": st, **r})
            counts["measured" if st == "measured" else "not_run"] += 1
        elif cond == "multi_turn":
            jobs = [(m, dn) for (_, m) in MODELS for (dn, _) in DTYPES]
            mine = [j for idx, j in enumerate(jobs)
                    if not shard or idx % shard[1] == shard[0]]
            svc = KeyService(os.path.join(d, "freshness.sqlite"))
            for model_short, dtype_name in mine:
                full = dict((s, f) for f, s in MODELS)[model_short]
                dtype = dict(DTYPES)[dtype_name]
                write_env(model_short, {"dtype": dtype_name})
                tok, model = load_model_pair(full, dtype)
                say(f"loaded {model_short} {dtype_name}")
                for sid in sids:
                    fix = load_v2_fixture(model_short, sid)
                    key = f"{sid}|{model_short}|{dtype_name}"
                    try:
                        if not fix.get("eligible"):
                            for sn, _, _, _ in SEGMENTS:
                                emit({"key": f"{key}|{sn}", "condition": "multi_turn",
                                      "status": "not_run", "reason": fix.get("reason")})
                                counts["not_run"] += 1
                            continue
                        segs, req = run_multi_turn(model, tok, svc, run_token, sid,
                                                   model_short, dtype_name, dtype, fix)
                        for s in segs:
                            s.update({"key": f"{key}|{s['seg']}", "condition": "multi_turn",
                                      "sid": sid, "model": model_short, "dtype": dtype_name,
                                      "status": "measured",
                                      "functional_pass": bool(
                                          s["prot_vs_plain_bitexact"]
                                          and s["sanity_max"] == 0.0
                                          and s["shape_match"] and s["finite"])})
                            emit(s)
                            counts["measured"] += 1
                        emit({"key": key, "condition": "multi_turn_req", "sid": sid,
                              "model": model_short, "dtype": dtype_name,
                              "status": "measured", **req})
                    except Exception as e:
                        log.write(traceback.format_exc(limit=6) + "\n")
                        say(f"ERROR {key}: {type(e).__name__}: {str(e)[:120]}")
                        for sn, _, _, _ in SEGMENTS:
                            emit({"key": f"{key}|{sn}", "condition": "multi_turn",
                                  "sid": sid, "model": model_short, "dtype": dtype_name,
                                  "status": "error", "error_type": type(e).__name__,
                                  "error": str(e)[:300]})
                            counts["error"] += 1
                    if counts["measured"] and counts["measured"] % 30 == 0:
                        out.flush()
                        say(f"progress {counts}")
                del model
        elif cond == "shared_prefix":
            tails = json.load(open(os.path.join(d, "shared_tails.json")))
            order = [m for _, m in MODELS]
            mine = [m for idx, m in enumerate(order) if not shard or idx % shard[1] == shard[0]]
            svc = KeyService(os.path.join(d, "freshness.sqlite"))
            for model_short in mine:
                full = dict((s, f) for f, s in MODELS)[model_short]
                write_env(model_short, {"dtype": "float32"})
                tok, model = load_model_pair(full, torch.float32)
                tbl = tails[model_short]
                for e in tbl["entries"]:
                    key = f"{e['sid_a']}|{model_short}"
                    if args.sanity and e["sid_a"] not in sids:
                        continue
                    try:
                        if not e.get("eligible"):
                            emit({"key": key, "condition": "shared_prefix",
                                  "status": "not_run", "reason": e.get("reason")})
                            counts["not_run"] += 1
                            continue
                        r = run_shared_prefix(model, tok, svc, run_token, e["sid_a"],
                                              e["sid_b"], model_short,
                                              {"prefix": tbl["prefix"],
                                               e["sid_a"]: e["tail_a"], e["sid_b"]: e["tail_b"]})
                        checks = ["artifact_ids_equal", "recover_a_equals_plain",
                                  "recover_b_equals_plain", "cross_principal_rejected",
                                  "prefix_swap_rejected"]
                        r.update({"key": key, "condition": "shared_prefix",
                                  "model": model_short, "status": "measured",
                                  "functional_pass": all(r[c] for c in checks)})
                        emit(r)
                        counts["measured"] += 1
                    except Exception as ex:
                        log.write(traceback.format_exc(limit=6) + "\n")
                        emit({"key": key, "condition": "shared_prefix",
                              "model": model_short, "status": "error",
                              "error_type": type(ex).__name__, "error": str(ex)[:300]})
                        counts["error"] += 1
                del model
        elif cond == "reencode":
            order = [m for _, m in MODELS]
            mine = [m for idx, m in enumerate(order) if not shard or idx % shard[1] == shard[0]]
            svc = KeyService(os.path.join(d, "freshness.sqlite"))
            for model_short in mine:
                full = dict((s, f) for f, s in MODELS)[model_short]
                write_env(model_short, {"dtype": "float32"})
                tok, model = load_model_pair(full, torch.float32)
                for sid in sids:
                    key = f"{sid}|{model_short}"
                    try:
                        fix = load_v2_fixture(model_short, sid)
                        if not fix.get("eligible"):
                            emit({"key": key, "condition": "reencode", "status": "not_run",
                                  "reason": fix.get("reason")})
                            counts["not_run"] += 1
                            continue
                        r = run_reencode(model, tok, svc, run_token, sid, model_short, fix)
                        checks = ["seg1_bitexact_logits", "reencoded_equals_plain_bitexact",
                                  "shape_after_reencode", "old_epoch_rejected",
                                  "duplicate_nonce_rejected"]
                        r.update({"key": key, "condition": "reencode", "sid": sid,
                                  "model": model_short, "status": "measured",
                                  "functional_pass": all(r[c] for c in checks)})
                        emit(r)
                        counts["measured"] += 1
                    except Exception as ex:
                        log.write(traceback.format_exc(limit=6) + "\n")
                        emit({"key": key, "condition": "reencode", "sid": sid,
                              "model": model_short, "status": "error",
                              "error_type": type(ex).__name__, "error": str(ex)[:300]})
                        counts["error"] += 1
                del model
        else:
            raise ValueError(cond)
    finally:
        out.close()
        say(f"done {counts}")
        if os.path.exists(env_path):
            e = json.load(open(env_path))
            e["finished"] = now()
            e["counts"] = counts
            with open(env_path, "w") as f:
                json.dump(e, f, indent=1)


# ------------------------------------------------------------- launcher
def cmd_launch(args):
    d = args.outdir
    lr_path = os.path.join(d, "launcher_run.json")
    run_token = args.run_token_override or dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    lr = {"run_token": run_token, "started": now(),
          "threads_per_worker": THREADS, "sanity": bool(args.sanity)}
    with open(lr_path, "w") as f:
        json.dump(lr, f, indent=1)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS=str(THREADS))
    os.makedirs(os.path.join(d, "launcher_logs"), exist_ok=True)
    if not os.path.exists(os.path.join(d, "shared_tails.json")):
        prepare_shared_tails(d)
    jobs = []
    for si in range(4):
        jobs.append(("multi_turn", f"{si}/4"))
    jobs += [("shared_prefix", "0/2"), ("shared_prefix", "1/2"),
             ("reencode", "0/2"), ("reencode", "1/2"),
             ("ledger", None), ("device", None), ("scheduler", None)]
    procs = []
    for cond, shard in jobs:
        tag = cond + (f"_s{shard.replace('/', 'of')}" if shard else "")
        lg = open(os.path.join(d, "launcher_logs", f"{tag}.log"), "a")
        cmd = [sys.executable, os.path.abspath(__file__), "--outdir", d,
               "--worker", "--condition", cond]
        if shard:
            cmd += ["--shard", shard]
        if args.sanity:
            cmd += ["--sanity"]
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=d)
        procs.append((p, tag))
        print(f"launched {tag} pid={p.pid}", flush=True)
    rc = {}
    for p, tag in procs:
        rc[tag] = p.wait()
        print(f"{tag} exit={rc[tag]}", flush=True)
    lr.update({"finished": now(), "exit_codes": rc,
               "nonzero": {k: v for k, v in rc.items() if v}})
    with open(lr_path, "w") as f:
        json.dump(lr, f, indent=1)
    if any(v for v in rc.values()):
        sys.exit(2)


def cmd_merge(args):
    d = args.outdir
    n_ids = 5 if args.sanity else 100
    all_sids = [json.loads(l)["sid"] for l in open(INPUTS)]
    sids = all_sids[:n_ids]
    exp = set()
    for sid in sids:
        for _, ms in MODELS:
            for dn, _ in DTYPES:
                for sn, _, _, _ in SEGMENTS:
                    exp.add(f"{sid}|{ms}|{dn}|{sn}")
            exp.add(f"{sid}|{ms}")
    recs, reqs = {}, {}
    wroot = os.path.join(d, "workers")
    for w in sorted(os.listdir(wroot)) if os.path.isdir(wroot) else []:
        cp = os.path.join(wroot, w, "checks.jsonl")
        if not os.path.exists(cp):
            continue
        for line in open(cp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] == "multi_turn_req":
                reqs[r["key"]] = r
            else:
                recs.setdefault(r["condition"], {})[r["key"]] = r
    mt = recs.get("multi_turn", {})
    mt_exp = {k for k in exp if k.count("|") == 3}
    m = [r for r in mt.values() if r["status"] == "measured"]
    acc = {"generated": now(), "sanity": args.sanity,
           "multi_turn": {
               "expected_total": len(mt_exp), "records": len(mt),
               "set_complete": set(mt) == mt_exp,
               "measured": len(m),
               "error": sum(1 for r in mt.values() if r["status"] == "error"),
               "not_run": sum(1 for r in mt.values() if r["status"] == "not_run"),
               "functional_pass": sum(1 for r in m if r.get("functional_pass")),
               "bitexact": sum(1 for r in m if r.get("prot_vs_plain_bitexact")),
               "sanity_nonzero": sum(1 for r in m if r.get("sanity_max", 0.0) != 0.0),
               "plain_vs_full_max": max((r.get("plain_vs_full_max", 0.0) for r in m),
                                        default=0.0)},
           "multi_turn_requests": {
               "expected_total": n_ids * 2 * 2, "records": len(reqs),
               "plain_pollution": sum(1 for r in reqs.values()
                                      if r.get("plain_baseline_polluted")),
               "recovered_bitexact": sum(1 for r in reqs.values()
                                         if r.get("recovered_equals_plain_bitexact"))}}
    for cond, want in (("shared_prefix", n_ids * 2), ("reencode", n_ids * 2)):
        rr = recs.get(cond, {})
        mm = [r for r in rr.values() if r["status"] == "measured"]
        acc[cond] = {"expected_total": want, "records": len(rr), "measured": len(mm),
                     "error": sum(1 for r in rr.values() if r["status"] == "error"),
                     "not_run": sum(1 for r in rr.values() if r["status"] == "not_run"),
                     "functional_pass": sum(1 for r in mm if r.get("functional_pass"))}
    for cond in ("ledger", "device", "scheduler"):
        rr = recs.get(cond, {})
        acc[cond] = list(rr.values())
    with open(os.path.join(d, "accounting_validation.json"), "w") as f:
        json.dump(acc, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in acc.items()
                      if not isinstance(v, list)}, indent=1, default=str)[:2500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--condition")
    ap.add_argument("--shard")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--run-token-override")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--prepare-shared-tails", action="store_true")
    args = ap.parse_args()
    if args.prepare_shared_tails:
        prepare_shared_tails(args.outdir)
    elif args.launch:
        cmd_launch(args)
    elif args.merge:
        cmd_merge(args)
    elif args.worker:
        cmd_worker(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
