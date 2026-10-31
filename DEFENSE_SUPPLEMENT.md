# VeilKV defensive implementation and evidence supplement

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
VALIDATION_STATUS.json contains independent recomputed counts and scope.

Only one prefill protection/recovery is followed by three continuation segments.
The test does not establish repeated encryption, shared-prefix scheduler,
hardware-attestation or device-transfer correctness. The current observations
are retained as functional results, including all valid negative outcomes.

## Offline inspection

Use Python 3.12 with the listed dependencies and cached public model tokenizers
and configurations. From this directory set PYTHONPATH to defense_runtime and
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
included or pooled. Earlier harness failures remain in the private provenance
archive and are not counted as functional outcomes of this corrected condition.

Worker files originally contain #run headers. In these copies only, those
headers are moved into separate run_events.json sidecars, leaving strict JSONL
with every experimental record unchanged. NORMALIZATION.json records the
conversion. The raw private files remain untouched. Data licenses and the
upstream KV-Cloak license are retained.
