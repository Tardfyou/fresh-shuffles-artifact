# Documentation

All file paths and shell commands in these notes are relative to the repository
root unless a command explicitly changes directory.

| Document | Purpose |
|---|---|
| [GPU_RESULTS.md](GPU_RESULTS.md) | Current primary GPU results, sources and offline checks |
| [GPU602_NATIVE.md](GPU602_NATIVE.md) | Current602-prompt native studies |
| [GPU602_LIVE.md](GPU602_LIVE.md) | Current602-prompt live attack and defense |
| [LATENCY_LENGTHS_1K.md](LATENCY_LENGTHS_1K.md) | Current five-length timing study |
| [REPORTING_SCOPE.md](REPORTING_SCOPE.md) | Current model and statistical reporting scope |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Auxiliary CPU experiment inventory and original cohort sizes |
| [DEFENSE_SUPPLEMENT.md](DEFENSE_SUPPLEMENT.md) | CPU defensive runtime, dependencies and checks |
| [ORDERING_STUDY.md](ORDERING_STUDY.md) | Local and GPT ordering records and metrics |
| [SUPPLEMENTARY_DETAILS.md](SUPPLEMENTARY_DETAILS.md) | Implementation and measurement definitions |
| [EXTENDED_EXPERIMENT_DETAILS.md](EXTENDED_EXPERIMENT_DETAILS.md) | Extended protocols and historical diagnostics |
| [ORBIT_VERIFICATION.md](ORBIT_VERIFICATION.md) | Operator-level repetition controls |
| [PERMLLM_ORBIT_VERIFICATION.md](PERMLLM_ORBIT_VERIFICATION.md) | Permutation and native three-party signature checks |

The machine-readable study index is `metadata/studies.json`. Saved CPU
validation receipts are in `metadata/audit/`; their original dated scopes are
retained. Dependency files are grouped in `requirements/`.

For setup, select the dependencies associated with the relevant study. Run
CPU runtime checks with:

```bash
python -m pip install -r requirements/requirements-defense.txt
python tools/check_defense_runtime.py --runtime defense_runtime --report /tmp/veilkv-runtime-check.json
```

The general plot gallery reads the historical CPU summary at
`results/visual_summary_20261102/data.json`; it does not reproduce the current
GPU figures. For the current length-stratified timing plot, use:

```bash
python tools/plot_latency_length_groups.py \
  --results results/gpu/accepted/latency-lengths-1k-20261115 \
  --output /tmp/veilkv-panels/cache_latency_distribution.pdf \
  --receipt /tmp/veilkv-panels/cache_latency_distribution.json
```

The original CPU source entry points and corrected cohorts are documented in
`ORDERING_STUDY.md` and `DEFENSE_SUPPLEMENT.md`. The fixed input manifest defines
the cohort; retain its IDs, token arrays and normal incorrect outcomes when
resuming a run.

Keep the frozen source baselines and their embedded CPU reference archives:
source-lock verification depends on those exact files. They are reference
inputs, not additional current measurement cohorts.
