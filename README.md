# Fresh Shuffles, Reused Secrets

Research artifact for **Fresh Shuffles, Reused Secrets: Breaking KV-Cloak with Repeated-Token Prompts**.

Anonymous review entry: https://anonymous.4open.science/r/fresh-shuffles-artifact-ED8D/

- [Paper (PDF)](paper/build/main.pdf)
- [Manuscript source](paper/main.tex)
- [Frozen evidence and hash manifest](paper/evidence/manifest.json)
- [Validation record](paper/review/submission_check.json)
- [Experiment source and saved results](experiments/kvcloak/)

## Reproduce the manuscript assets

Use Python 3.12 and run from the repository root:

```sh
python3 -m venv .venv-paper
.venv-paper/bin/pip install -r requirements-paper.txt
.venv-paper/bin/python paper/scripts/generate_assets.py
.venv-paper/bin/python paper/scripts/check_submission.py
```

These commands regenerate plots and table rows from saved evidence and validate the supplied PDF and evidence hashes. They do not load a model or contact a model API.

To rebuild the PDF, install Tectonic and run:

```sh
tectonic -o paper/build paper/main.tex
.venv-paper/bin/python paper/scripts/check_submission.py
```

## Evidence scope

The main result is 263,029 / 263,904 head-token multiset matches across ten models and twelve configurations. This is a descriptive count with correlated observations, not a count of independent requests. The privacy study is limited to one model, one secret configuration, and 30 synthetic closed-set cases. Per-request refresh is a mitigation candidate whose complete functional correctness and general security are not established.

The original JSON and experiment scripts are retained for provenance. Historical aggregate fields and script comments may predate the frozen corrections. For privacy statistics, use the integer `strictly_higher` and `tie_group_size` fields and the corrected derivation in `paper/scripts/generate_assets.py`; the legacy `unique_first` and top-k aggregate fields in `privacy_v3.json` are superseded. Earlier defense results do not establish the security of later defense versions. Consult [frozen claim boundaries](paper/evidence/FROZEN_CLAIMS.md) and the manuscript before interpreting exploratory results.

The supplied requirements describe manuscript regeneration only. The historical model runs lack a complete contemporaneous dependency lock; no claim of independently reproduced or bit-identical model runs is made by the asset checks.

## Third-party material

KV-Cloak source snapshots derive from `SiO-2/kvcloak` commit `6b40f36edb2f337557543e7e60b10022308883d4`; its Apache-2.0 license is retained in [THIRD_PARTY_KVCLOAK_LICENSE](THIRD_PARTY_KVCLOAK_LICENSE). IEEEtran files retain their original notices. No additional blanket license is asserted for third-party material or the manuscript.
