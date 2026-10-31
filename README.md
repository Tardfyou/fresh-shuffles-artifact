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
scopes and must not be mixed with these corrected cohorts. This update covers
the ordering study; other later corrected studies are maintained separately.

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
