#!/usr/bin/env python3
"""Check the frozen manuscript package without running models or using network."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "build" / "main.pdf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = {}
failures = []

manifest = json.loads((ROOT / "evidence" / "manifest.json").read_text())
bad_hashes = []
for name, record in manifest.items():
    copy = ROOT / "evidence" / name
    source = ROOT.parent / record["source"]
    expected = record["sha256"]
    if not copy.exists() or sha256(copy) != expected:
        bad_hashes.append(f"copy:{name}")
    if not source.exists() or sha256(source) != expected:
        bad_hashes.append(f"source:{name}")
checks["evidence_manifest_entries"] = len(manifest)
checks["evidence_hashes_ok"] = not bad_hashes
if bad_hashes:
    failures.append({"evidence_hashes": bad_hashes})

if not PDF.exists():
    failures.append("missing_pdf")
else:
    with pdfplumber.open(PDF) as pdf:
        checks["pages_total"] = len(pdf.pages)
        checks["letter_page_points"] = [float(pdf.pages[0].width), float(pdf.pages[0].height)]
        checks["page_limit_ok"] = len(pdf.pages) <= 18
        checks["us_letter_ok"] = abs(pdf.pages[0].width - 612) < 1 and abs(pdf.pages[0].height - 792) < 1
        texts = [page.extract_text() or "" for page in pdf.pages]
        # Read each column separately: full-page extraction can merge an
        # appendix heading with a bibliography line in the other column.
        column_texts = [
            [
                page.crop((0, 0, page.width / 2, page.height)).extract_text() or "",
                page.crop((page.width / 2, 0, page.width, page.height)).extract_text() or "",
            ]
            for page in pdf.pages
        ]
        appendix_pages = [
            i + 1
            for i, columns in enumerate(column_texts)
            if any(line.strip() == "Appendix A." for text in columns for line in text.splitlines())
        ]
        conclusion_pages = [i + 1 for i, text in enumerate(texts) if re.search(r"\b8\.\s+Conclusion\b", text)]
        checks["conclusion_pages"] = conclusion_pages
        checks["appendix_start_page"] = appendix_pages[0] if appendix_pages else None
        checks["appendix_heading_present"] = bool(appendix_pages)
        reference_pages = [
            i + 1 for i, columns in enumerate(column_texts)
            if any(line.strip() == "References" for text in columns for line in text.splitlines())
        ]
        checks["references_start_page"] = reference_pages[0] if reference_pages else None
        checks["references_and_appendix_pages"] = (
            len(pdf.pages) - reference_pages[0] + 1 if reference_pages else None
        )
        checks["backmatter_target_ok"] = (
            checks["references_and_appendix_pages"] is not None
            and 4 <= checks["references_and_appendix_pages"] <= 5
        )
        if not checks["backmatter_target_ok"]:
            failures.append("backmatter_not_four_to_five_pages")
        checks["main_text_within_13_pages"] = bool(conclusion_pages) and max(conclusion_pages) <= 13
        if not checks["page_limit_ok"] or not checks["us_letter_ok"] or not checks["main_text_within_13_pages"] or not checks["appendix_heading_present"]:
            failures.append("pdf_format_or_page_limit")

tex = "\n".join(path.read_text(errors="replace") for path in sorted(ROOT.rglob("*.tex")))
placeholders = re.findall(r"TODO|TBD|FIXME|PLACEHOLDER|\[\[", tex, flags=re.I)
checks["placeholder_count"] = len(placeholders)
main_source = (ROOT / "main.tex").read_text(errors="replace")
checks["anonymous_source_ok"] = "\\author{}" in main_source and not re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+", tex, flags=re.I)
generated_rows = [line.strip() for line in (ROOT / "tables" / "breadth_rows.tex").read_text().splitlines() if line.strip()]
evaluation_source = (ROOT / "sections" / "evaluation.tex").read_text()
checks["generated_table_rows_match_source"] = bool(generated_rows) and all(row in evaluation_source for row in generated_rows)
if placeholders:
    failures.append("placeholders")
if not checks["anonymous_source_ok"]:
    failures.append("identity_in_tex")
if not checks["generated_table_rows_match_source"]:
    failures.append("generated_table_rows_drift")

checks["paper_sha256"] = sha256(PDF) if PDF.exists() else None
checks["status"] = "pass" if not failures else "fail"
checks["failures"] = failures
(ROOT / "review" / "submission_check.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
raise SystemExit(0 if not failures else 1)
