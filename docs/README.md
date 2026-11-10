# Documentation

All file paths and shell commands in these notes are relative to the repository
root unless a command explicitly changes directory.

| Document | Purpose |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Experiment inventory and cohort sizes |
| [REPORTING_SCOPE.md](REPORTING_SCOPE.md) | Current model and statistical reporting scope |
| [DEFENSE_SUPPLEMENT.md](DEFENSE_SUPPLEMENT.md) | CPU defensive runtime, dependencies and checks |
| [ORDERING_STUDY.md](ORDERING_STUDY.md) | Local and GPT ordering records and metrics |
| [SUPPLEMENTARY_DETAILS.md](SUPPLEMENTARY_DETAILS.md) | Implementation and measurement definitions |
| [GPU_RESULTS.md](GPU_RESULTS.md) | GPU handoff, acceptance and result organization |

The machine-readable study index is `metadata/studies.json`. Saved CPU
validation receipts are in `metadata/audit/`; their original dated scopes are
retained. Dependency files are grouped in `requirements/`.

For setup, select the dependencies associated with the relevant study. Run
CPU runtime checks with:

```bash
python -m pip install -r requirements/requirements-defense.txt
python tools/check_defense_runtime.py --runtime defense_runtime --report /tmp/veilkv-runtime-check.json
```

Plot sources read `results/visual_summary_20261102/data.json`. For example:

```bash
python tools/plot_paper_gallery.py --data results/visual_summary_20261102/data.json --out /tmp/veilkv-panels
```

The original CPU source entry points and corrected cohorts are documented in
`ORDERING_STUDY.md` and `DEFENSE_SUPPLEMENT.md`. The fixed input manifest defines
the cohort; retain its IDs, token arrays and normal incorrect outcomes when
resuming a run.

- [Extended experiment details](EXTENDED_EXPERIMENT_DETAILS.md): category counts, historical probe/error/timing diagnostics, and native/live-engine controls retained outside the compact paper.
