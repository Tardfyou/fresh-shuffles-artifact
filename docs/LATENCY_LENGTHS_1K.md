# Length-stratified latency measurements

`latency-lengths-1k-20261115` replaces the old single512-token GPU timings.
All20 formal jobs and20 startup jobs are complete:180000 and360 observations.
All78 returned source/input files match both the returned baseline and the
issued baseline byte for byte. The exact server source snapshot was supplied.

The fixed main3000 real prompts are tokenized with each pinned model tokenizer
and grouped into32–63,64–127,128–255,256–511 and512–767 tokens. Input texts
are complete, with their tokenizer prefixes; no truncation. A fixed seeded
uniform draw with replacement selects1000 calls per group. Repeated draws are
retained:each cell covers475–497 distinct inputs, not1000 distinct prompts.
All methods and same-model block profiles share their actual input schedules.

The four profiles preserve the earlier models/precisions: Llama3.1-8B/FP32 at
KV-Cloak block16/32/64 and Qwen2.5-0.5B/bf16 at block16. Methods are unfused
KV-Cloak, fused KV-Cloak, VeilKV with a32-token public prefix, and authenticated
encryption with no public prefix. Each protection/recovery series has1000
calls; separate whole-request prefill uses the same schedule. Each series has
20 additional warm-ups. Host threads16, one A100-SXM4-80GB GPU, serial queue.

Wall time synchronizes CUDA at both boundaries. Byte profiles include CPU
cryptography and GPU–CPU copies; KV-Cloak tensors stay on GPU. Cache generation,
configuration preparation, fresh input cloning, nonce creation, protection of
recovery inputs and fused VM preparation remain outside operation timing.
Untimed cache generation uses canonical32-token public-prefix evaluation then
private text to avoid prefix-payload conflicts from full-prefill length rounding.
All methods receive the same cache. Whole-request native prefill is separate.
The first32tokens are public in the sharing profile; a32-token request is all
public. The public fraction therefore varies with input length, and the no-sharing
baseline is all private. This experiment measures protect/recover calls; it is
not a measurement of production cache-hit throughput or fused-model prefill.

Every raw timing, draw index, source ID, actual length, byte count and invocation
environment is retained. Seconds/1000tokens use each call's actual length;
ms/GiB uses its actual stored KV bytes. The saved summary reports raw-call
p50/p95/p99 and median normalized times separately for all20 profiles/groups.
The appendix figure uses Llama8B/FP32/block16 and retains every outlier.

On that profile, VeilKV protection/recovery medians range103.17–554.69ms and
37.05–494.02ms across the five groups. Unfused KV-Cloak medians range32.44–33.05
and50.38–52.06ms. These are ranges of group medians, not a pooled statistic.
Llama block32/64 and the smaller-model profile are separate in LATENCY_SUMMARY.
No timing observation is discarded for being slow, and no result claims a
component-level causal breakdown of CPU crypto, freshness storage and copying.

```
python tools/inspect_latency_lengths.py --root .
```

Raw data: `results/gpu/accepted/latency-lengths-1k-20261115/`.
Acceptance, source comparison, input bindings, all groups and percentiles:
`results/gpu/summaries/latency-lengths-1k-20261115/`.
Issued and actual-verified source: `sources/latency-lengths-1k/`.
The original execution ZIP can be reused; its previous100-call GPU result
cohort is removed under author-authorized cleanup. Historical CPU experiments
retain their separate scope. Full measurement failures and extreme observations
remain in the returned private handoff and portable records where present.
