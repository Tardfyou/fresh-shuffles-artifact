# Repeat to Reveal: code and experimental data

Anonymous review: https://anonymous.4open.science/r/fresh-shuffles-artifact-ED8D/

This artifact contains the fixed inputs, experiment sources, saved predictions
and counts, and offline analysis for Repeat to Reveal and VeilKV.

## Start here

- [Experiment inventory](docs/EXPERIMENTS.md): studies, cohorts and evidence.
- [Dataset and result catalog](metadata/studies.json): one index of all studies.
- [Current reporting scope](docs/REPORTING_SCOPE.md): GPU primary/CPU auxiliary, nine models, eleven
  configurations, and the original 3,000-prompt manifest.
- [Defensive implementation and validation](docs/DEFENSE_SUPPLEMENT.md).
- [Ordering study](docs/ORDERING_STUDY.md).
- [GPU results and handoff layout](docs/GPU_RESULTS.md).
- [Current602-prompt native results](docs/GPU602_NATIVE.md).
- [Current602-prompt live integration](docs/GPU602_LIVE.md).
- [Technical details](docs/SUPPLEMENTARY_DETAILS.md).

## Directory layout

| Directory | Contents |
|---|---|
| `docs/` | Setup, study descriptions and measurement definitions |
| `metadata/` | Study catalog, saved validation reports and GPU run specifications |
| `requirements/` | Dependency files and the defensive environment lock |
| `data_sources/` | Dataset source and license information |
| `experiments/kvcloak/` | Retained CPU experiment sources and historical records |
| `defense_runtime/` | Runtime associated with the accepted CPU defense records |
| `sources/` | Third-party source snapshots |
| `tools/` | Offline checks and analysis |
| `results/` | Retained CPU measurements at their recorded paths |
| `results/gpu/` | Accepted GPU runs, profile summaries and their index |

Accepted GPU results are the primary measurements; CPU results remain auxiliary.
The main A100 suite, the accepted602-prompt native rerun and the602-prompt live vLLM attack/defense integration are indexed in
`results/gpu/index.json`. Compressed record parts retain every observation.
The separately issued `veilkv-gpu-80g-experiments-20261109.zip` defines the
single-80GiB execution plan; its job specifications are mirrored in
`metadata/gpu/protocol/`. Each returned run is checked before its portable
records enter `results/gpu/`.

## Inspect saved results

Run from the repository root. These commands use saved records and make no
model or external API calls:

```bash
python tools/check_artifact_catalog.py
python tools/summarize_reporting_scope.py --root . --out /tmp/veilkv-reporting-scope
python tools/inspect_saved_review.py --root .
python tools/inspect_final_observations.py --root .
python tools/inspect_new_evidence_records.py --root .
python tools/inspect_fixed_cohorts.py --root .
python tools/inspect_visual_summary.py --root .
python tools/inspect_marker_numerics.py --root .
```

The primary manifest is
`results/main_public/samples_main_public.jsonl`. Focused WildChat studies reuse
its saved IDs; private-value cases are synthetic. All positive and negative outcomes in current accepted cohorts remain.
Superseded smaller GPU cohorts have been removed; selected original engine
diagnostics are retained. The smaller live integration cohort is superseded by the complete602 rerun. Historical twelve-configuration totals and the
current eleven-configuration reporting summary have separate scopes.

The accepted CPU large-model extension is in
`results/cpu_large_models_20261110/`: Qwen3-14B and Qwen3-32B each reuse all3,000
fixed IDs, eight KV heads and both seeds. Inspect it with
`python tools/inspect_cpu_large_models.py --root .`.

## Dependencies and source versions

CPU defensive checks use `requirements/requirements-defense.txt`; ordering
analysis uses `requirements/requirements-ordering.txt`. The GPU bundle has its
own pinned native and vLLM environments. Installation and entry points are
listed in [the documentation index](docs/README.md).

Accepted CPU byte-cache results use codec format v1. The GPU execution bundle
uses format v2, which adds model/tokenizer/RoPE binding for private-only caches.
Accepted v2 measurements are cataloged with that version.

Manuscript files, slides, credentials, model weights, runtime logs and machine
local paths are excluded from this artifact. Dataset terms remain applicable;
the upstream KV-Cloak license is in `THIRD_PARTY_KVCLOAK_LICENSE`.
