# Manuscript build

This directory is the current IEEE S&P 2027 cycle-2 manuscript package.

1. Regenerate figures, the main table, ordering diagnostics, and checked derived numbers:

   `../.paper-venv/bin/python scripts/generate_assets.py`

   `../.paper-venv/bin/python scripts/generate_order_assets.py`

   `../.paper-venv/bin/python scripts/generate_supplement.py --check`

2. Build from this directory:

   `tectonic -o build main.tex`

3. Inspect `build/main.pdf`, `review/derived_numbers.json`, and `review/submission_check.json`.

The generators read frozen JSON/JSONL records only; they do not run a model or access the network. `evidence/manifest.json` maps each frozen copy to its source result and SHA-256. Historical exploratory reports outside this directory are not authoritative manuscript evidence.

The VeilKV-Plus v4 defense-candidate evidence is stored in `evidence/veilkv_plus_v4.json`, with the run-1 archive and v4 bugfix log retained under `../experiments/kvcloak/`. It records chain-aware prefix-to-tail reuse, persistent nonce freshness, integrity/replay checks, and TEE-profile functionality. Its performance and dictionary records are inherited from run 1. It is reported under a simulated TEE key-release boundary, not as a hardware attestation proof. The subsequent arithmetic-profile optimization and confirmed spectral-identification negative result are frozen separately in `evidence/arithmetic_optimized_*` and `evidence/arithmetic_spectrum_blocks.jsonl`; the manifest points to the retained source records.

The 2026-10-15 ordering delivery is frozen in `evidence/order_v2/`, including all physical result rows, the retrospective protocol, source snapshots, and a retrospective offline-tokenizer audit. Run `scripts/audit_order_v2.py` to reconcile its 600 main and 300 boundary metric IDs. `review/order_v2_audit.json` records the descriptive results and limits: 581 distinct main spans, 109 nominal-length overshoots, missing raw predictions, whole-span rather than per-block conservation, and direction pairs that do not hold blockwise bags fixed. The ordering records do not replace the original 263,029/263,904 main denominator. The delivered direction-analysis file labels grouped position accuracy; it is not a factual-direction classifier. The performance comparison instantiates KV-Cloak with `fused=False`.
