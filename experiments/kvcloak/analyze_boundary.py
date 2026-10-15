"""Deterministic derivation of boundary-pair direction analysis from SAVED files.
Inputs: metrics_boundary.jsonl + samples_boundary.jsonl (no model, no rerun).
Output: boundary_direction_analysis.json"""
import json
import statistics
from collections import defaultdict

BASE = "/mnt/hdd/research/kvcloak_results/order_v2"
metrics = {}
for line in open(f"{BASE}/metrics_boundary.jsonl"):
    d = json.loads(line)
    metrics[d["id"]] = d
samples = {}
for line in open(f"{BASE}/samples_boundary.jsonl"):
    d = json.loads(line)
    samples[d["id"]] = d

pairs = defaultdict(dict)
for sid, s in samples.items():
    if s.get("category") == "boundary_pair":
        pairs[s["pair_id"]][s["direction"]] = sid

per_pair, pa_a, pa_b, ro = [], [], [], []
for pid, dirs in sorted(pairs.items()):
    if "a" not in dirs or "b" not in dirs:
        continue
    ma, mb = metrics[dirs["a"]], metrics[dirs["b"]]
    pa_a.append(ma["end2end"]["position_accuracy"])
    pa_b.append(mb["end2end"]["position_accuracy"])
    ro.append(ma["random_order"]["position_accuracy"])
    per_pair.append(dict(
        pair_id=pid,
        id_a=dirs["a"], id_b=dirs["b"],
        pa_a=ma["end2end"]["position_accuracy"],
        pa_b=mb["end2end"]["position_accuracy"],
        exact_a=bool(ma["end2end"].get("exact_match")),
        exact_b=bool(mb["end2end"].get("exact_match")),
        both_exact=bool(ma["end2end"].get("exact_match")
                        and mb["end2end"].get("exact_match"))))

n = len(per_pair)
out = dict(
    source_files=["metrics_boundary.jsonl", "samples_boundary.jsonl"],
    derivation="deterministic join on pair_id/direction; no rerun",
    n_pairs=n,
    direction_bias=dict(
        pa_dir_a=round(statistics.mean(pa_a), 4),
        pa_dir_b=round(statistics.mean(pa_b), 4),
        diff=round(abs(statistics.mean(pa_a) - statistics.mean(pa_b)), 4),
        random_order=round(statistics.mean(ro), 4)),
    exact_match=dict(
        any_exact=sum(1 for p in per_pair if p["exact_a"] or p["exact_b"]),
        both_directions_exact=sum(1 for p in per_pair if p["both_exact"])),
    per_pair=per_pair)
with open(f"{BASE}/boundary_direction_analysis.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "per_pair"}, indent=1))
print(f"saved -> {BASE}/boundary_direction_analysis.json ({n} pairs)")
