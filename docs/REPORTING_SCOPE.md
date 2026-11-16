# GPU-primary reporting

The accepted A100 measurements are now primary. The nine-model/eleven-configuration FP32-protection suite records91,611,072/91,611,072 hits; Qwen3-14B and32B extensions each record12,480,000/12,480,000. The32B FP32 profile is native first-layer execution; both large models also have full-model BF16/FP32-protection profiles.

GPU protection precision and calibrator differ from the historical CPU suite.
BF16-protection/legacy GPU bridges remain separate. The current602-prompt live
vLLM run recovers359,936/359,936 token occurrences, passes6,020/6,020 functional
cases and rejects19,264/19,264 tamper cases before model execution. Its separate
KV-Cloak ordered-restore adapter preserves outputs on562/602 targets; this is
distinct from attack recovery and VeilKV correctness. Current performance uses
five length groups and1,000 calls per method, operation and group, retaining
180,000 formal timings. See [GPU_RESULTS.md](GPU_RESULTS.md),
[GPU602_LIVE.md](GPU602_LIVE.md), [LATENCY_LENGTHS_1K.md](LATENCY_LENGTHS_1K.md)
and `results/gpu/index.json`.

## Auxiliary CPU reporting history


The 2026-11-08 manuscript uses nine models and eleven model/block-size
configurations on the same 3,000 frozen WildChat prompts. The author requested
removal of Phi-3-mini from the extended-model scope. This changes the model
scope after the original experiments; it is not a data-quality exclusion.
Every retained model keeps every prompt, head and permutation seed.

Original main tables and follow-up records remain under their original result
paths, including valid Phi positive and negative measurements. They document
the full historical collection. Use results/reporting_scope_20261108/summary.json
for the current manuscript total and selected diagnostic/epoch aggregates.
The inventory and visual-summary files describe the current selection.
No model or recovery execution was performed for this change.

## Residual-error analysis

The selected auxiliary CPU total is 91,528,679/91,611,072, leaving 82,393 misses.
Llama-3.2-1B contributes 37,460; Qwen2.5-0.5B contributes six. Existing
instrumented records reuse all 3,000 IDs per model, and all 12,000 seed counters
agree with those two historical model rows. They cover 45.47231% of the current
main misses. They are follow-up observations, not historical block predictions.

The existing results/saved_record_review_20261101/{tag}/failed_block_diagnostics.jsonl.gz
files retain every failing block from these two follow-ups, its row predictions, gold token types, selected
marker-cancellation row and recorded scores. Exact one-error blocks with one
out-of-bag prediction at that row number 33,619/37,454 for Llama and 6/6 for
Qwen. Llama has 37,448 one-error blocks and six two-error blocks in total.
The pattern is consistent with residual perturbations after estimated-marker subtraction from bf16-protected values. The selected row is in recovered S coordinates. It is not a stored historical marker label. The later seeded-layout audit in
results/marker_numerical_diagnostics_20261109/ reconstructs the marker positions
and confirms hypothesis selection in every saved failing Qwen/Llama block.
Known-key arithmetic controls demonstrate BF16 marker-scale precision loss;
parameter-estimation effects are not individually decomposed. The 306-block nine-path numerical comparison
retains identical predictions and does not support a blanket scheduling-noise
explanation. Remaining model rows have no corresponding full block diagnostics.

Recompute all reporting counts, bootstrap and row-localization counts:

    python tools/summarize_reporting_scope.py --root .

The bootstrap uses 2,000 prompt-level resamples within the original five
600-prompt length strata, seed 20261029. Configurations, heads and permutation
seeds remain paired. The interval conditions on the recorded secret keys.
