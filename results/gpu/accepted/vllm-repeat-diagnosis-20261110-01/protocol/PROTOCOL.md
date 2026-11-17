# Prespecified diagnostic protocol: inconsistent unprotected vLLM repeats

Goal: diagnose why repeated generations disagree in the original
vllm_baseline_b16(93/100 agreement) and b32(92/100 agreement) runs, using saved
evidence. Do not modify the main or live-vLLM bundles. This diagnostic directory
operates independently and retains all results and evidence.

## Verified starting facts

- Original implementation: runner/extended.py:194-226. The same engine generates
  twice consecutively for one input, with temperature0, max_tokens16,
  max_num_seqs1, eager execution, prefix caching enabled, BF16, TP1, the UniProc
  backend and VLLM_USE_V1=0. Block16 uses72 blocks/max_len1024; block32 uses40 blocks.
- Each block size has100 records from one successful invocation
  (b16=dd781495..., b32=c6ed3139...). An initial CUDA invalid-argument error
  occurred before any record was written because concurrent runs interfered;
  it does not explain the measured divergences.
- There are seven b16 and eight b32 divergent inputs, with five shared IDs:
  0068/0119/0180/0307/0346. Their execution positions range15-73; first
  divergences occur at positions0/1/5/6/8/9/10. Shared divergent inputs often
  first differ at the same position:0068@1,0119@9,0307@8 and0346@6.
- Reuse the100 original `input_ids` directly, without retokenization.
  Assert that b16 and b32 have identical input sets.

## Planned variants

Each variant uses a separate engine process. All other settings match the
original run exactly. This table records the intended protocol; the later
source-based reclassification of B is documented in the run README.

| Variant | Prefix caching | Action before generation | Comparison | Question |
|---|---|---|---|---|
| A | False | None | Twice in the same engine | Does disabling caching remove disagreement? |
| B | True | Reset before each generation | Cold-cold | Does a fresh prefill reproduce the effect? |
| C | True | Reset before each pair | Cold-warm | Does the original comparison reproduce? |
| C0 | True | Reset before each pair | Cold-warm | C without logprobs; instrumentation control |
| D | True | Warm once, then no reset | Warm-warm | Is a full-cache-hit path consistent? |
| E(FP32) | True | Reset before each pair | Cold-warm | Only15 divergent cases, dtype=float32, reported separately |

- Use `scheduler.reset_prefix_cache()`, the vLLM0.8.5 method. After every reset,
  record its return value and `get_num_free_gpu_blocks()`, which must equal the
  complete capacity. Otherwise mark reset_failed and stop that variant.
- Record free blocks before and after every generation, and all
  RequestOutput.metrics timing fields.

## Observations and numerical evidence

- Use `SamplingParams(temperature=0,max_tokens=16,logprobs=20)`, except C0.
- For each pair, retain both token-ID sequences, equality and the first
  differing position. At the differing step and preceding equal step, retain
  top-20(token,logprob), top1-top2 gaps and the other run's corresponding
  top-k. Compare distributions only under identical generation histories.
- C0 and C must produce exactly the same token outputs. If they differ,
  treat the instrumentation as affecting execution and use an offline copy.

## Inputs and order

- Stage1:15 divergent inputs plus five agreeing inputs in their original order.
  Variant order: C0,C,B,D,A,E.
- Stage2 after diagnosis: repeat the key controls on all100 original inputs
  at both block sizes, for at least three complete rounds.

## Prespecified interpretation

- A and B disagree: kernel/scheduling nondeterminism independent of caching.
- A agrees and B disagrees: cold-prefill nondeterminism; identical full inputs
  can yield different logits.
- A and B agree, C disagrees: numerical differences between cache reuse and recomputation.
- C agrees but the original run disagrees: test cross-input cache state,
  including shared prefixes/eviction, by reproducing the original continuous
  per-input execution in stage2.
- D disagrees: the warm path itself is nondeterministic.
- E removes disagreement: BF16 near-ties contribute. Report this evidence
  separately; it does not replace the main interpretation.

Do not attribute the result to BF16 in advance. Use the control matrix and
numerical records. Retain every ordinary disagreement; do not change seeds,
remove inputs or relax metrics.

## Handoff

Retain analysis scripts and conclusions in `analysis/`, all per-input records
in `records/`, and actual source in `scripts/`. HANDOFF.md answers: the cause,
supporting controls, whether the original baseline needs correction, and the
effect on correctness comparisons for protected integration.
