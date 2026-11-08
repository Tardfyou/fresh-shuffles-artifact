# Repeat to Reveal: code and experimental data

Anonymous review entry: https://anonymous.4open.science/r/fresh-shuffles-artifact-ED8D/

This repository contains experiment code and saved experimental data. Manuscript
sources, PDFs, figures, presentation files, build outputs and runtime logs are
excluded. Model weights and credentials are not included.

- `experiments/kvcloak/`: retained experiment sources and saved results.
- `sources/`: retained third-party source snapshots.

The corrected fixed-cohort ordering study is now included under
`results/takeover_20261029/corrected/`, with its fixed manifest under
`results/main_public/`. See `ORDERING_STUDY.md` for the two completed GPT
comparisons, four local ordering conditions, semantic scores and offline
verification. Older records in `experiments/kvcloak/results/` keep their original
scopes and must not be mixed with these corrected cohorts. The sections below describe the additional retained studies.

Machine-local author paths are replaced with relative paths in distribution
copies. Original source worktrees and raw records remain unchanged.
The upstream KV-Cloak license is retained in `THIRD_PARTY_KVCLOAK_LICENSE`.


The defensive supplement is now included: see `DEFENSE_SUPPLEMENT.md`,
`defense_runtime/`, and the accepted byte-cache and row-secret continuation
records under `results/takeover_20261029/corrected/`. It retains both successful
and failed functional checks. Independent accounting and reporting corrections
are in `VALIDATION_STATUS.json`. Use `requirements-defense.txt` for this part.
The new supplement exports defensive components and known-key tests; it does
not replace the historical mixed experiment runners.

## Saved-record follow-up (2026-11-01)

Historical main counters are supplied for all twelve configurations. The separate
saved-record follow-up includes every sample/seed counter and decode timing for
two configurations, all their failed-block diagnostics, final-worker phase
records and fixed-candidate observations. It corrects the decoding denominator
from head-token opportunities to target-token positions across seed runs.
`EXPORT_STATUS.json` specifies the exported scope and omitted all-correct-block
observations; aggregate audit fields are retained for those blocks.

Run `python tools/inspect_saved_review.py` from this directory for an independent
offline count, unit and diagnostic-subset check. No model or API is called.
`AUDIT_STATUS.json` distinguishes accepted saved evidence from unmeasured
properties. The defensive supplement also includes known-key shared-prefix,
segmented-continuation and single-reprotection measurements.

## Expanded defensive validation (2026-11-02)

The accepted CPU study reuses the existing OR602 cohort (including lifecycle100):
48,160 condition requests, 96,320 protected/plain consumer comparisons, five mask
shapes and two native dtypes on two models. It includes repeated protection,
identity/epoch transition, non-prefill boundary protection and stored-prefix reuse.
Single-field rejection tests isolate identity and epoch. Every valid old outcome
is retained separately. See DEFENSE_SUPPLEMENT.md and EXPANDED_DEFENSE_VALIDATION.json.

The dependency lock was installed from public indexes in a new environment.
Unpacked data/runtime checks passed; plotting sources use saved timing arrays.
The 512-token cache timing fixture is an existing WildChat prompt, and all 800
observations are supplied. No manuscript, slides, logs, databases or model weights
are included. GPU/production serving/hardware guarantees are not inferred from
these known-key CPU interface checks.

## Current manuscript scope (2026-11-08)

The manuscript reports nine models and eleven configurations, retaining all
3,000 fixed prompts, all configured heads and both target-permutation seeds.
The author selected a narrower extended-model scope, omitting Phi-3-mini from
all manuscript results. This is a scope change, not rejection of invalid data.
Original full-scope records remain available with all positive and negative
outcomes. Historical twelve-configuration totals are not the current paper total.

`results/reporting_scope_20261108/summary.json` is the current reporting summary:
91,528,679/91,611,072 hits (99.910062%), with a prompt-stratified bootstrap 95%
interval of [99.908954%, 99.911160%]. Recompute it without inference or APIs:

```
python tools/summarize_reporting_scope.py --root .
```

Saved diagnostic rows locate one out-of-bag prediction at the selected
marker-cancellation row in 33,619/37,454 failing Llama head-blocks, each with
one missed token, and in all six Qwen failures. These are new offline counts
of existing observations, not a new recovery experiment. The selected row is
not a saved ground-truth marker label. See REPORTING_SCOPE.md for the interpretation.

## Historical full-scope follow-up observations (2026-11-02)

`results/final_record_observations_20261102/` contains complete-session timings,
legacy attempt-label accounting, fixed-candidate scores, and the Phi seed-counter
follow-up. Its three configurations cover 471,739/516,666 main-table misses.
`results/new_evidence_20261102/` adds fully logged probe-session observations,
supplementary rotation-angle diagnostics, first/all-head identification counts,
306 blocks with nine complete prediction arrays each, reproducible rounded-gap
histograms, and six fixed scaling settings. All positive and negative outcomes
remain, including the two-hit Qwen regression and per-prompt regressions at
settings with positive aggregate gains. No new recovery implementation is added.

Inspect these records without model loading or API calls:

```
python tools/inspect_final_observations.py --root .
python tools/inspect_new_evidence_records.py --root .
```

The a/S predicate and supplementary M diagnostic have separate scopes. Numerical
path stability is measured on the selected Qwen/Llama blocks. The scaling grid
contains four Qwen and two Llama settings; it is not a general scaling guarantee.
GPU, real TEE and production-service experiments are outside this study.

## Visual summaries

The final figures are generated from `results/visual_summary_20261102/data.json`.
This includes all 602 cosine scores per condition, all 100 synthetic privacy
cases, all 800 operation timings, every scaling setting and both seed scores in
the head-budget comparison. These are frozen selections of accepted records,
not additional experiments. Exact main and per-key counts remain in their data.

```
python tools/plot_paper_gallery.py --data results/visual_summary_20261102/data.json --out /tmp/paper-panels
```

Tables and the key matrix use the author-selected teal/coral endpoints. Other
panels directly follow USENIX Security Approximate Caches Figures 11/22 (frequency
histograms and lavender/grey paired bars) and NDSS PROMPTPEEK Figures 10/11/13
(light-blue boxes, red medians, a pale background with white grids, and horizontal
stacks). These panels use their source figures' palettes.
Numpy/matplotlib/scipy and installed Arial are required for plotting.

## Accepted fixed-cohort records

The corrected result directory also includes the frozen probe/key-budget records,
paired-calibrator diagnostics, 300-prompt fused study, 100-prompt scaling/lifecycle/
row-rotation records, all 1,500 cross-key prompt/configuration records, 100 synthetic
privacy cases with candidates and matched controls, integrity checks, the two
100-candidate arithmetic-profile studies, and the completed seven-stage prototype
battery. `FIXED_COHORT_EXPORT.json` lists the exported files. Numeric values,
experimental text and predictions are preserved; only local-path metadata is
normalized. No new measurement or recovery implementation is added.

```
python tools/inspect_fixed_cohorts.py --root .
python tools/inspect_visual_summary.py --root .
```

## Supplementary technical details

See [SUPPLEMENTARY_DETAILS.md](SUPPLEMENTARY_DETAILS.md) for implementation,
measurement-accounting and historical version details. The retrospective model
revision inventory and twelve calibration rows are retained in
`results/supplementary_reporting_values.json`; all underlying results remain.
