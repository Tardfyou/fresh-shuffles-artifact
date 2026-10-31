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
