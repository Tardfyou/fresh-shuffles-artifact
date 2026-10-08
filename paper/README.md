# Manuscript build

This directory is the current IEEE S&P 2027 cycle-2 manuscript package.

1. Regenerate figures, the main table, and checked derived numbers:

   `../.venv-paper/bin/python scripts/generate_assets.py`

2. Build from this directory:

   `tectonic -o build main.tex`

3. Inspect `build/main.pdf`, `review/derived_numbers.json`, and `review/submission_check.json`.

The asset generator reads only `evidence/*.json`; it does not run a model or access the network. `evidence/manifest.json` maps each frozen copy to its source result and SHA-256. Historical exploratory reports outside this directory are not authoritative manuscript evidence.

The VeilKV-Plus v4 defense-candidate evidence is stored in `evidence/veilkv_plus_v4.json`, with the run-1 archive and v4 bugfix log retained under `../experiments/kvcloak/`. It records chain-aware prefix-to-tail reuse, persistent nonce freshness, integrity/replay checks, TEE-profile functionality and performance, and the arithmetic-profile negative results. It is reported under a simulated TEE key-release boundary, not as a hardware attestation proof.
