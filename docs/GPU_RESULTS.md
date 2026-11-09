# GPU results and handoff

The single-80GiB GPU execution bundle is
`veilkv-gpu-80g-experiments-20261109.zip`, science version
`gpu80g-20261109-v2-fp32-protection`. The expected jobs, pinned model revisions,
precision profiles and codec version are mirrored in `metadata/gpu/protocol/`.
There are currently no accepted GPU measurements in this repository.

## Returned run

The execution bundle exports `gpu-results-private-handoff.zip`. Its original
records, logs, invocation metadata and interrupted attempts go to the private
intake workspace. The received archive remains unchanged there. Each distinct
execution gets a run ID, for example `a100-80g-20261109-01`.

After review, the portable release uses this layout:

```text
results/gpu/
  index.json
  accepted/<run_id>/
    protocol/                 # Frozen jobs, model revisions and source version
    environment/              # Portable hardware/software and invocation records
    source/                   # Scientific code used by this run
    raw/<job_id>/             # All measured outputs, including normal failures
    validation/               # ID accounting and independent prediction checks
  summaries/<run_id>/
    by_profile.json           # Recomputed counts and prompt-level intervals
    comparisons.json          # Explicitly matched comparisons
```

`index.json` lists accepted runs and their raw/summary locations. The accepted
raw records are kept once; analysis and figures refer to those locations.

## Acceptance

A run must supply its exact source version, declared model/tokenizer revisions,
actual device and dtype, calibration profile, cache scope, block size, fixed
sample IDs and completed observation keys. The execution bundle's source lock,
CUDA startup suite, ID accounting, prediction recounts and plain controls must
pass. Incomplete transport or runtime observations resume on the same IDs;
normal incorrect predictions and functional failures remain measured outcomes.

Raw environment paths are normalized in distribution copies. Runtime logs,
local databases and original transport fragments stay in private intake.
Reported environments, seeds, numeric results, token arrays and predictions
remain intact.

## Precision and comparison groups

New primary GPU measurements use FP32 protection, calibration and matching.
Qwen3-14B/32B full-model BF16 weights are paired with FP32 protection. The 32B
FP32 first-layer profile and full-model BF16 profile retain distinct scopes.
Historical BF16-protection bridges, S/a consumer conditions and numerical
controls retain their declared precision. Each summary groups by model/revision,
model dtype, protection dtype, calibrator, block size, forward scope and codec
version.

Native HF block-size compatibility and unprotected vLLM baselines have separate
study entries. CPU format-v1 observations remain separate from GPU format-v2
observations. MMLU and SQuAD retain their frozen complete evaluation cohorts.

## Where to find the current plan

- `metadata/gpu/protocol/jobs.json`: 93 formal execution units, including shards.
- `metadata/gpu/protocol/smoke_jobs.json`: 60 required CUDA startup jobs.
- `metadata/gpu/protocol/models.json`: pinned model repositories and revisions.
- `metadata/gpu/protocol/NUMERICAL_PRECISION.json`: arithmetic profiles.
- `metadata/gpu/protocol/DEFENSE_BINDING_VERSION.json`: codec-format change.

Run `python tools/check_artifact_catalog.py` to check the index and directory
references. New measurement counts enter the GPU index only after acceptance.
