"""Descriptive re-statistics over saved v2 records (no model runs).

Groups functional pass/fail by block-alignment class: for each mask variant
the real-history length modulo 16 decides whether KV-Cloak's per-block row
shuffle can mix padding rows with real-token rows inside one 16-row block.
Smoke (first 5 IDs) was degenerate (all real lengths ≡ 0 mod 16); the full
run exercises non-aligned cases, and this script reports the observed
association. Association only — not causal attribution by itself.
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def load_all():
    seg, req = {}, {}
    wroot = os.path.join(D, "workers")
    for w in sorted(os.listdir(wroot)):
        cp, rp = (os.path.join(wroot, w, x) for x in ("checks.jsonl", "requests.jsonl"))
        if os.path.exists(cp):
            for line in open(cp):
                if line.startswith("#") or not line.strip():
                    continue
                r = json.loads(line)
                if r["key"] not in seg or r.get("attempt", 0) >= seg[r["key"]].get("attempt", 0):
                    seg[r["key"]] = r
        if os.path.exists(rp):
            for line in open(rp):
                if line.startswith("#") or not line.strip():
                    continue
                r = json.loads(line)
                if r["key"] not in req or r.get("attempt", 0) >= req[r["key"]].get("attempt", 0):
                    req[r["key"]] = r
    return seg, req


def real_len_mod(model, sid, mask):
    f = json.load(open(os.path.join(D, "fixtures", f"base_{model}_{mask}_{sid}.json")))
    return sum(f["mask"]) % 16, f.get("eligible", False)


def main():
    seg, req = load_all()
    # mixed-block prediction per (mask, r): pads share a final block with real
    # tokens iff r != 0 and pad_len > 16 - r  (pad_len 8/16; r in 1..15)
    def mixed(mask, r):
        pad = {"no_pad": 0, "left8": 8, "right8": 8, "left16": 16, "right16": 16}[mask]
        if mask == "no_pad":
            return False
        if mask.startswith("left"):
            # left pads sit after the 16-token prefix: block 1 is mixed iff
            # pad_len < 16 (left8 always mixes 8 pads + 8 real; left16 never)
            return pad < 16
        # right pads sit immediately after the real history: whenever r != 0
        # the pad rows start inside the block that still holds real rows, so
        # mask-padding rows share a 16-row block with real-token rows
        return r != 0

    rows = []
    for k, r in seg.items():
        if r["status"] != "measured":
            continue
        sid, model, dtype, cond, mask, sname = k.split("|")
        r_mod, _ = real_len_mod(model, sid, mask)
        rows.append({"model": model, "dtype": dtype, "cond": cond, "mask": mask,
                     "seg": sname, "mod": r_mod,
                     "mixed": mixed(mask, r_mod),
                     "pass": bool(r.get("functional_pass")),
                     "pp": r.get("prot_vs_plain_max")})
    out = {"n_seg_records": len(rows), "groups": []}
    for dtype in ("float32", "bfloat16"):
        for mask in ("no_pad", "left8", "right8", "left16", "right16"):
            for mx in (False, True):
                sel = [x for x in rows if x["dtype"] == dtype and x["mask"] == mask
                       and x["mixed"] == mx]
                if not sel:
                    continue
                npass = sum(x["pass"] for x in sel)
                out["groups"].append({
                    "dtype": dtype, "mask": mask, "mixed_block_class": mx,
                    "n": len(sel), "pass": npass, "fail": len(sel) - npass,
                    "pp_median": sorted(x["pp"] for x in sel)[len(sel) // 2],
                    "pp_max": max(x["pp"] for x in sel),
                    "mods_seen": sorted({x["mod"] for x in sel}),
                })
    # the pre-registered check: fp32 failures concentrate in mixed-block class
    fp32 = [x for x in rows if x["dtype"] == "float32"]
    fp32_fail_mixed = sum(1 for x in fp32 if not x["pass"] and x["mixed"])
    fp32_fail_clean = sum(1 for x in fp32 if not x["pass"] and not x["mixed"])
    out["fp32_failure_concentration"] = {
        "fail_and_mixed": fp32_fail_mixed, "fail_and_not_mixed": fp32_fail_clean,
        "consistent_with_block_mixing_mechanism": fp32_fail_clean == 0,
    }
    path = os.path.join(D, "analysis_block_alignment.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["fp32_failure_concentration"]))
    for g in out["groups"]:
        if g["dtype"] == "float32":
            print(f"{g['mask']:8s} mixed={str(g['mixed_block_class']):5s} "
                  f"n={g['n']:4d} pass={g['pass']:4d} med={g['pp_median']:.2e} max={g['pp_max']:.2e}")
    print("written:", path)


if __name__ == "__main__":
    main()
