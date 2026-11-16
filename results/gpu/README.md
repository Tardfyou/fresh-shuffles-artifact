# GPU measurements

Status: accepted GPU measurements are primary; retained CPU records are auxiliary.

- `accepted/<run_id>/`: portable raw records and per-invocation metadata.
- `summaries/<run_id>/`: derived statistics linked to that run.
- `index.json`: authoritative accepted-run catalog and current job overrides.

The five accepted runs cover the original main suite, selected unprotected
engine diagnostics, the complete602-prompt native rerun, the complete602-prompt
live integration and the five-length latency study. The current live and latency
run IDs are `primary_live_run` and `primary_latency_run` in `index.json`.
Superseded smaller GPU cohorts and the original fixed512-token timing results
have been removed. Startup records and retained diagnostics remain separate
from formal measurements.

See [the GPU result guide](../../docs/GPU_RESULTS.md) for the intake and
acceptance format. Existing CPU measurements remain at their recorded paths.
