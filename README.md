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
.venv-paper/bin/python paper/scripts/generate_order_assets.py
.venv-paper/bin/python paper/scripts/generate_supplement.py
.venv-paper/bin/python paper/scripts/generate_supplement.py --check
.venv-paper/bin/python paper/scripts/check_submission.py
.venv-paper/bin/python paper/scripts/audit_final.py
```

These commands regenerate plots and table rows from saved evidence and validate the supplied PDF and evidence hashes. They do not load a model or contact a model API.

The current PDF has 18 pages: 13 pages of main content and five pages of references and appendices. Its 25 cited works include the model, architectural, and software sources used to explain the artifact. Supplemental tables expose the retained per-configuration diagnostics, all 15 model-key epochs, and per-type privacy tie statistics.

To rebuild the PDF, install Tectonic and run:

```sh
tectonic -o paper/build paper/main.tex
.venv-paper/bin/python paper/scripts/check_submission.py
```

## Evidence scope

The main result is 263,029 / 263,904 head-token multiset matches across ten models and twelve configurations. This is a descriptive count with correlated observations, not a count of independent requests. The privacy study is limited to one model, one secret configuration, and 30 synthetic closed-set cases. Per-request refresh is a mitigation candidate whose complete functional correctness and general security are not established.

The original JSON and experiment scripts are retained for provenance. Historical aggregate fields and script comments may predate the frozen corrections. For privacy statistics, use the integer `strictly_higher` and `tie_group_size` fields and the corrected derivation in `paper/scripts/generate_assets.py`; the legacy `unique_first` and top-k aggregate fields in `privacy_v3.json` are superseded. Earlier defense results do not establish the security of later defense versions. Consult [frozen claim boundaries](paper/evidence/FROZEN_CLAIMS.md) and the manuscript before interpreting exploratory results.

The final manuscript corrects the number of perfect main-suite configurations to six. The current ordering follow-up preserves token multiplicity and supersedes the exploratory estimates. Its 600 main records yield 22.3% mean position accuracy and 18 exact spans; duplicated inputs, nominal-length overshoots, missing raw predictions and uncontrolled blockwise boundary bags limit interpretation. See paper/review/ORDER_V2_ACCEPTANCE_20261015.md. Cached model and dataset revisions are retrospective records, not contemporaneous per-run provenance. See [final validation](paper/review/final_audit.json).

The supplied requirements describe manuscript regeneration only. The historical model runs lack a complete contemporaneous dependency lock; no claim of independently reproduced or bit-identical model runs is made by the asset checks.

The official-entry record is a historical file-format experiment: its defender thresholds used the first eight target conversations, and its harness did not enforce successful subprocess completion before reading a reused output directory. The summary alone does not certify a fresh end-to-end run. The S/a comparison retains rounded percentages, not raw counts. These qualifications are explicit in the manuscript. The importable fusion helper is packaged byte-for-byte from its retained upstream snapshot.

VeilKV-Plus v4 is included as a bounded defense-candidate measurement. Its evidence is `paper/evidence/veilkv_plus_v4.json` and its implementation is `experiments/kvcloak/exp_veilkv_plus_v2.py`. The result records chain-aware prefix-to-tail reuse, persistent nonce freshness, integrity/replay detection, TEE-profile functionality and performance, and the arithmetic-profile negative results. The attestation path remains a stub, so this is not a hardware-secure-system proof.

## Third-party material

KV-Cloak source snapshots derive from `SiO-2/kvcloak` commit `6b40f36edb2f337557543e7e60b10022308883d4`; its Apache-2.0 license is retained in [THIRD_PARTY_KVCLOAK_LICENSE](THIRD_PARTY_KVCLOAK_LICENSE). IEEEtran files retain their original notices. No additional blanket license is asserted for third-party material or the manuscript.

## Ordering delivery and figure typography

The 2026-10-15 ordering delivery is frozen in paper/evidence/order_v2/. Run `paper/scripts/audit_order_v2.py` and `paper/scripts/generate_order_assets.py` in the manuscript environment; they use saved files only and do not run a language model. This package preserves the delivered retrospective protocol and all result rows, with the acceptance report taking precedence over overbroad summary claims. The 100 direction-reversal pairs share global bags but differ in their block bags, so they do not establish zero factual-direction leakage.

Statistical-figure regeneration requires a locally licensed Times New Roman installation (for example the redistributable Microsoft Core Fonts package). Raw font files are not distributed. The supplied vector PDF figures embed subsets and can be compiled into the manuscript without installing the plotting font. The private/local visual-audit baseline is reduced here to numeric/table source files; no conversation history, credentials or experiment environments are packaged.
