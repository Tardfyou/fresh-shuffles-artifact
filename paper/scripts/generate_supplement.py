#!/usr/bin/env python3
"""Deterministic appendix tables from frozen JSON; no model or network use."""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
read = lambda name: json.loads((root / "evidence" / name).read_text())
labels = {
    "llama32_1b": "Llama-3.2-1B", "olmo2_1b": "OLMo-2-1B",
    "phi3_mini": "Phi-3-mini", "qwen25_05b": "Qwen2.5-0.5B",
    "qwen25_05b_b8": "Qwen2.5-0.5B", "qwen25_05b_b32": "Qwen2.5-0.5B",
    "qwen25_15b": "Qwen2.5-1.5B", "qwen25_3b": "Qwen2.5-3B",
    "qwen25_7b": "Qwen2.5-7B", "qwen3_06b": "Qwen3-0.6B",
    "qwen3_17b": "Qwen3-1.7B", "tinyllama": "TinyLlama-1.1B",
}
def table(spec, header, rows, group_models=False):
    body=[]
    previous=None
    for row in rows:
        if group_models and previous is not None and row[0] != previous:
            body.append(r"\hline")
        body.append(" & ".join(map(str,row)) + r" \\")
        previous=row[0]
    styled_header=" & ".join(r"\textbf{" + cell.strip() + "}" for cell in header.split("&"))
    return "\n".join([
        r"\begin{tabular}{" + spec + "}", r"\toprule",
        r"\rowcolor{TableHead}", styled_header + r" \\", r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", "",
    ])
rows = []
for r in read("STATS.json")["per_config"]:
    if r["correct"] is None:
        continue
    d = read(r["tag"] + ".json")
    ds = [d[f"head{h}_recovery"] for h in range(d["kv_heads"])]
    assert all(d[f"head{h}_states"] == d["b"] for h in range(d["kv_heads"]))
    rows.append([
        labels[r["tag"]], d["b"], d["probe_prefix_tokens"], d["probe_core_blocks"],
        f'{min(d[f"head{h}_separation"] for h in range(d["kv_heads"])):.1f}',
        f'{min(x["S_min_signed_cos"] for x in ds):.6f}',
        f'{max(x["w_relerr"] for x in ds):.6f}',
    ])
outputs = {"app_diagnostics.tex": table(
    r"lrrrr>{\columncolor{TableBlue}}rr", r"Model & $b$ & Prefix & Core blocks & Min. separation & Min. $S$ cosine & Max. $w$ rel. err.", rows)}
rows = []
for tag, d in read("independent_eval.json").items():
    name = {"qwen05": "Qwen2.5-0.5B", "llama1b": "Llama-3.2-1B", "phi3": "Phi-3-mini"}[tag]
    for e in d["epochs"]:
        rows.append([
            name, e["epoch"]+1, f'{e["cal_heads"]}/{e["cal_total"]}',
            e["probe_tokens"], f'{100*e["cond_recovery_mean"]:.2f}',
            f'{100*e["cond_recovery_min"]:.2f}', f'{100*e["ctrl_acc"]:.2f}',
        ])
outputs["app_epochs.tex"] = table(
    r"lrrr>{\columncolor{LeakHigh}}r>{\columncolor{LeakHigh}}r>{\columncolor{LeakLow}}r",
    r"Model & Epoch & Heads & Probe & Session mean (\%) & Min. session (\%) & Control (\%)",
    rows, group_models=True)
privacy = read("privacy_v3.json")["per_entity"]
rows = []
for tag, label in [("ssn","SSN-like"),("credit","Card-like"),("password","Password"),
                   ("mrn","MRN"),("name","Name"),("address","Address")]:
    rs = [r for r in privacy if r["type"] == tag]
    assert len(rs) == 5
    unique = sum(r["strictly_higher"] == 0 and r["tie_group_size"] == 1 for r in rs)
    expected = [100*sum(min(1,max(0,(k-r["strictly_higher"])/r["tie_group_size"])) for r in rs)/len(rs)
                for k in (1,5,10)]
    rows.append([label, f"{unique}/5", max(r["tie_group_size"] for r in rs),
                 *[r"\RateCell{" + f"{v:.2f}" + "}" for v in expected]])
outputs["app_privacy.tex"] = table(
    "lrrrrr", r"Type & Unique & Max. $m$ & Top-1 & Top-5 & Top-10", rows)
check = argparse.ArgumentParser()
check.add_argument("--check", action="store_true")
args = check.parse_args()
for name, text in outputs.items():
    path = root / "tables" / name
    if args.check:
        assert path.read_text() == text, f"Generated table differs: {name}"
    else:
        path.write_text(text)
print(json.dumps({"tables": list(outputs), "mode": "checked" if args.check else "generated"}))
