# VeilKV defensive implementation and evidence supplement

The current manuscript uses accepted GPU measurements as primary; see
[GPU results](GPU_RESULTS.md). The following retained sections document the
auxiliary CPU studies and their original outcomes.

This code/data supplement contains the defensive cache codec, authenticated
runtime dtype adapter, arithmetic operations, freshness ledger, official known-key
cache transform, and accepted known-key validation records. Manuscripts, PDFs,
slides, review handoffs, API credentials, model weights, logs and databases are
excluded. The codec and three public-model utility function bodies are extracted
unchanged from the retained sources. The mixed experiment runner is not included.

## Evidence and outcomes

The two byte-cache conditions reuse 100 lifecycle prompts and retain the
unadapted native-bf16 errors and unrun later segments alongside all successful
FP32 and repaired native-bf16 transitions. The S/a study retains all 12,000
segment checks: 4,597 functional passes and 7,403 failures. These are conditions
on 100 source prompts, not 12,000 independent users.

There are 2,000 independent paired request conditions and 6,000 paired segments.
Original and refreshed configurations pass 2,306/6,000 and 2,291/6,000 segments;
63 pairs disagree (39 original-only passes; 24 refresh-only passes). Equal rates
or unconditional functional equivalence are not established. The stored
metadata/audit/VALIDATION_STATUS.json contains independent recomputed counts and scope.

The row-secret study uses one prefill protection/recovery followed by three
continuation segments. A separate known-key byte-profile study includes 1,200
continuation segments, 200 equal-length shared-prefix cases and 200 single
re-protection cases, plus nine software-ledger scenarios. Its validation record
states the exact functional scope. Neither study establishes real serving
scheduler integration, hardware attestation, device-transfer correctness or
security against within-request observations. All valid negative outcomes remain.

## Offline inspection

Use Python 3.12 with the listed dependencies and cached public model tokenizers
and configurations. From the repository root set PYTHONPATH to defense_runtime and
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1, then run:

```
python tools/audit_sa_handoff_v2.py --input results/takeover_20261029/corrected/defense_handoff_v2_20261031_1319 --manifest results/main_public/samples_main_public.jsonl --subsets results/main_public/experiment_subsets.json --report /tmp/sa-validation.json
```

The auditor loads tokenizers/configurations and regenerates known test
parameters; it does not load model weights, run inference or contact APIs.
Measurements can be inspected without repeating the experiment. Their actual
measurement source copies remain next to the original data, with known machine
paths replaced only in distribution copies. The source test's historical input
path is supplied with the same 100 fixed texts; rejected v1 measurements are not
included or pooled. Discarded harness attempts were permanently removed at the owner's request.
All accepted positive and negative functional outcomes remain available.

Worker files originally contain #run headers. In these copies only, those
headers are moved into separate run_events.json sidecars, leaving strict JSONL
with every experimental record unchanged. metadata/audit/NORMALIZATION.json records the
conversion. The raw private files remain untouched. Data licenses and the
upstream KV-Cloak license are retained.

## Expanded OR602 software validation

The expanded known-key CPU study reuses all 602 existing ordering-cohort IDs,
including all prior lifecycle100 IDs. Two models, two dtypes, five mask shapes
and four conditions yield 48,160 request keys and 96,320 protected/plain consumer
comparisons. Input arrays are frozen; all 1,000 overlapping older fixtures agree.
The source and data include every condition outcome. Software ledger tests use
100 rounds with eight processes (3,200 competing operations), an additional
fresh-process restart check and 1,000 sequential resource calls. SQLite handles
are closed explicitly after transaction completion; crypto/key scopes are unchanged.

The original FP32 scheduling workers were intentionally retired after preserving
their complete records. A frozen completed-key list and separate resumed worker
records retain all outcomes without duplicate keys. Resource scheduling and exit
metadata are included; nonzero retired-worker exits are not silently erased.

Use requirements/requirements-defensive-20261102.lock for the separately installed public
dependency environment. From the artifact root, inspect the expanded records:

```
python tools/audit_defense602.py --root results/takeover_20261029/corrected/defense_complete602_20261102 --complete
python tools/check_defense_runtime.py --runtime defense_runtime --report /tmp/defense-runtime-units.json
```

The first command reads records and sources only. The second runs 16 small
finite-tensor codec units, not model inference or extra WildChat observations.
Full model measurement sources are supplied separately and require cached public
checkpoints. GPU, production serving adapters, hardware trust/rollback and
within-request security are not established by these CPU functional checks.

To repeat model measurements, create a new sibling directory under
results/takeover_20261029/corrected/. Copy defense_runtime into its source/
directory and copy the original study's inputs.jsonl and PROTOCOL.json there.
Use that new directory with tools/defense_complete602.py prepare --root NEW_ROOT,
then launch --smoke --shards 1 --root NEW_ROOT and, only after the smoke gate,
launch --shards 8 --threads 4 --root NEW_ROOT. Keep the published measurement
directory intact. Shared lifecycle inputs and prior fixtures are already supplied
at their referenced sibling paths. All model/tokenizer loads are offline.

Latency panels can be regenerated from the saved eight-condition timing record:

```
python tools/plot_defense_cache_perf.py --result results/takeover_20261029/corrected/defense_complete602_20261102/matched_cache_perf/result.json --out /tmp/defense-latency
```

Plotting uses the locked Matplotlib dependency and an installed Arial font for
the approved appearance. Font files are not distributed. The command writes
plots outside this code/data snapshot and never runs models or requests an API.
