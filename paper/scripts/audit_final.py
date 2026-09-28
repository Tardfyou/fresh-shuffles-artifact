#!/usr/bin/env python3
"""Validate final prose against frozen counts, citations, PDF fonts, and links."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import pdfplumber

root = Path(__file__).resolve().parents[1]
evidence = root / "evidence"
read = lambda name: json.loads((evidence / name).read_text())
stats = read("STATS.json")
configs = [row for row in stats["per_config"] if row["correct"] is not None]
correct = total = heads = 0
perfect = []
diagnostics = []
text_counts = {}
for row in configs:
    data = read(row["tag"] + ".json")
    pairs = [v for target in data["victims"].values() for v in target["raw"].values()]
    for name, target in data["victims"].items():
        entry = text_counts.setdefault(name, [0, 0])
        for hits, opportunities in target["raw"].values():
            entry[0] += hits
            entry[1] += opportunities
    c, n = map(sum, zip(*pairs))
    assert [c, n] == [row["correct"], row["total"]]
    assert data["control_fresh_secrets_raw"] == row["control_raw"]
    assert data["probe_raw_entry"]
    for h in range(data["kv_heads"]):
        assert data[f"head{h}_states"] == data["b"]
        diagnostics.append(data[f"head{h}_recovery"])
    if c == n:
        perfect.append(row["tag"])
    correct += c
    total += n
    heads += data["kv_heads"]
assert [correct, total] == stats["global"] == [263029, 263904]
assert len(configs) == 12 and len(perfect) == 6
assert all(d["S_bijection"] for d in diagnostics)
fusion_module = root.parent / "experiments/kvcloak/fusion.py"
fusion_snapshot = root.parent / "sources/defense__core__fusion.py.txt"
assert fusion_module.is_file()
assert fusion_module.read_bytes() == fusion_snapshot.read_bytes()
privacy = read("privacy_v3.json")["per_entity"]
unique = sum(r["strictly_higher"] == 0 and r["tie_group_size"] == 1 for r in privacy)
expected = {str(k): sum(min(1, max(0, (k-r["strictly_higher"])/r["tie_group_size"]))
                       for r in privacy) / len(privacy) for k in (1, 5, 10)}
assert len(privacy) == 30 and unique == 21
for k, value in expected.items():
    assert abs(value - read("privacy_v3_scoring_audit.json")["all"]["topk_expected"][k]) < 1e-12
sections = "\n".join(p.read_text() for p in sorted((root / "sections").glob("*.tex")))
text_labels = {
    "prose_en": "English prose", "python_code": "Python code",
    "chinese": "Chinese prose", "dialogue": "Dialogue", "math": "Mathematical prose"
}
for name, (hits, opportunities) in text_counts.items():
    assert f"{text_labels[name]} & {hits:,} / {opportunities:,} & {opportunities-hits}" in sections
assert sum(n-c for c,n in text_counts.values()) == total-correct == 875
assert "100\\% on six configurations" in sections
assert "b\\ge2" in sections
assert "drops multiplicity" in sections
assert "were not recorded in per-run manifests" in sections
bib_keys = set(re.findall(r"@\w+\{([^,]+),", (root / "references.bib").read_text()))
cites = {key.strip() for match in re.findall(r"\\cite\{([^}]+)\}", sections) for key in match.split(",")}
assert cites <= bib_keys
assert not re.search(r"(?i)\b(?:TODO|TBD|FIXME|PLACEHOLDER)\b|\?\?", sections)
with pdfplumber.open(root / "build/main.pdf") as pdf:
    fonts = Counter(ch["fontname"].split("+")[-1] for ch in pdf.pages[0].chars)
    assert fonts["NimbusRomNo9L-Regu"] > 1000
    assert fonts["NimbusRomNo9L-Medi"] > 100
    assert not any("LMRoman" in name or "LatinModern" in name for name in fonts)
    urls = [link.get("uri", "") for page in pdf.pages for link in page.hyperlinks]
    assert "https://anonymous.4open.science/r/fresh-shuffles-artifact-ED8D/" in urls
    assert len(pdf.pages) <= 18
    assert not pdf.metadata.get("Author")
    page_count = len(pdf.pages)
log_path = root / "build/main.log"
if log_path.exists():
    log = log_path.read_text(errors="replace")
    assert not re.search(r"(?:Citation|Reference).*undefined|Overfull \\hbox|^!", log, flags=re.M)
out = {
    "scope": "Arithmetic, source/prose reconciliation, references, and compiled PDF; no model reruns",
    "head_token_counts": [correct, total],
    "configurations": len(configs),
    "per_fixed_text_counts": text_counts,
    "misses_total": total-correct,
    "calibrated_configuration_heads": heads,
    "perfect_configurations": perfect,
    "minimum_S_cosine": min(d["S_min_signed_cos"] for d in diagnostics),
    "maximum_w_relative_error": max(d["w_relerr"] for d in diagnostics),
    "privacy_unique_first": [unique, len(privacy)],
    "privacy_expected_topk": expected,
    "citation_keys": sorted(cites),
    "all_citations_resolve": True,
    "fusion_dependency_matches_snapshot": True,
    "pdf_pages": page_count,
    "page1_fonts": dict(fonts),
    "anonymous_artifact_hyperlink_present": True,
    "pdf_sha256": hashlib.sha256((root / "build/main.pdf").read_bytes()).hexdigest(),
    "limitations_retained": [
        "Head-token observations are correlated; no population prevalence estimate",
        "Privacy study is single-model/single-key with incomplete retained traces",
        "S/a refresh lacks validated full functional correctness and general security",
        "Model and dataset revisions are retrospective, not per-run provenance",
        "Exploratory ordering implementation loses multiplicity"
    ],
    "status": "pass"
}
(root / "review/final_audit.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
