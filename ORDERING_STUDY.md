# Fixed-cohort ordering results

The final API comparison contains GPT-6.1-Sol and GPT-6-Luna, each with all
602 fixed WildChat sample IDs. The original prompt and every returned response
are retained. Sol uses medium reasoning and Luna high reasoning; both have a
4,096-token completion budget and temperature zero. The accepted scope is in
`results/takeover_20261029/corrected/api_protocol.json`.

| Model / condition | Mean position accuracy (%) | Exact spans / 602 | Mean embedding cosine |
|---|---:|---:|---:|
| Local E2E beam | 27.776 | 1 | 0.945 |
| GPT-6.1-Sol | 8.241 | 4 | 0.902 |
| GPT-6-Luna | 3.361 | 0 | 0.771 |

The local beam receives block-level bags and enforces token multiplicities;
the GPT models receive one global bag and return free text. These settings
vary both the supplied ordering information and the decoding constraints.
Exact spans, positional agreement and descriptive text similarity are separate
metrics. The table reports prompt means, not ratios of pooled token counts.

The local study retains E2E, oracle-bag, no-context and random-order outputs
for the same 602 IDs (574 distinct complete token spans). Candidate compatibility
is a separate 553-pair diagnostic; it is not semantic direction classification.

Cosine values were measured using all-MiniLM-L6-v2 with a 256-encoder-token
limit. The verifier below checks saved values and aggregates; it does not run
a new encoder or contact a model API.

## Saved files

All paths below are relative to `results/takeover_20261029/corrected/`.

- `api_ready/`: identical fixed API inputs, complete GPT responses and any saved transport attempts.
- `api_*_scores.jsonl`, `api_summary.json`, `api_gpt_acceptance.json`: every prediction, denominator and aggregate.
- `bags/`, `order/`, `ordering_final_summary.json`: saved block bags, four local prediction conditions and counters.
- `order_semantics_*.json*`: retained local per-prompt cosine scores and distributions.
- `direction/`, `direction_final_summary.json`: separate candidate-compatibility records.

The 3,000-prompt manifest and fixed subsets are under `results/main_public/`.
Raw texts, IDs, predictions and measured outcomes are preserved. Source-local
metadata paths are normalized only in these distribution copies.

## Offline verification

Use Python 3.12 with `requirements-ordering.txt` and the cached public Qwen2.5-0.5B
tokenizer (no model weights are loaded by this verification). From the root:

```sh
python tools/verify_gpt_ordering.py \
  --results results/takeover_20261029/corrected \
  --manifest results/main_public/samples_main_public.jsonl \
  --report /tmp/gpt-ordering-verification.json
```

This verifies all 1,204 raw GPT outputs, tokenized predictions, counters, exact
spans, supplied-bag conservation, settings and saved cosine aggregates.
`result_checks.py` provides the saved-record local ordering and candidate
compatibility checks. No new reconstruction experiment is needed for inspection.

Only the two complete GPT cohorts enter this API comparison. Other earlier
API collections are excluded. Manuscript files and private correspondence are
not distributed.
