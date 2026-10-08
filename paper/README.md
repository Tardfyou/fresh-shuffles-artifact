# Manuscript build

This directory is the current IEEE S&P 2027 cycle-2 manuscript package.

1. Regenerate figures, the main table, and checked derived numbers:

   `../.venv-paper/bin/python scripts/generate_assets.py`

2. Build from this directory:

   `tectonic -o build main.tex`

3. Inspect `build/main.pdf`, `review/derived_numbers.json`, and `review/submission_check.json`.

The asset generator reads only `evidence/*.json`; it does not run a model or access the network. `evidence/manifest.json` maps each frozen copy to its source result and SHA-256. Historical exploratory reports outside this directory are not authoritative manuscript evidence.

The VeilKV v2 defense-prototype evidence is stored in `evidence/veilkv2_05b.json` and its implementation in `../experiments/kvcloak/exp_veilkv2.py`. It records FP32 correctness, the stored-artifact prefix boundary, five-repetition timing, residual transfer measurements, and the open BF16 K-storage issue. It is reported as bounded prototype evidence, not as a security proof.
