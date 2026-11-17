# Server operation scope

Reuse the existing native environment and model cache, inspect resources, and
run, supervise, resume and export the serial queue completely. The five length
groups, sampling with replacement,1,000 calls per operation and original timing
boundaries are fixed. Do not change scientific code, input schedules, random
seeds, precision, thresholds, timing boundaries or results.

Do not delete errors, repeat completed measurements because their values are
unfavorable, or use old100-call records to complete the new counts. Preserve the
execution state on failure and return it with `run.py export --partial`.
Use `--retry-blocked` only after diagnosing and fixing a clear environment or
resource problem. Do not automatically start experiments outside this bundle.

Finish with `run.py verify` and `run.py export`. Return the complete export,
including source_snapshot and SCIENTIFIC_BASELINE.
