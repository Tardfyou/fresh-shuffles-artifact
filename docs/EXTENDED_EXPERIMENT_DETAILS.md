# Extended experiment details

These retained details supplement the compact paper. No experiment or outcome
is removed. GPU measurements are primary; historical CPU records remain auxiliary.

## Qwen0.5B category counts

The same3,000 prompts, both KV heads and two permutation seeds are regrouped
by heuristic prompt category. Sampling is by length, not category.

| Category | Matches | Opportunities | Misses |
|---|---:|---:|---:|
| General |2,035,072|2,035,072|0|
| Code |491,008|491,008|0|
| Creative writing |375,872|375,872|0|
| Question answering |152,640|152,640|0|
| Mathematics |67,264|67,264|0|
| Total |3,121,856|3,121,856|0|

These are correlated opportunities from the existing main row, not an extra
cohort. Counts are reproducible from GPU records and the fixed manifest.

## Auxiliary CPU probe accounting

Auxiliary CPU records logged a low-margin change policy without executing it and lack submitted arrays for later attempts. The separate fully logged follow-up saves three sessions per model: 30 session--head observations, or 20 after identical-input deduplication. Of 14 event rows referencing those sessions, seven inputs equal primary and seven differ. Thresholds clear for 0/2 Qwen and 2/5 Llama triggered heads.

The $a/S$ criterion requires cosines at least 0.999 and bijective column matching. It does not test $M$. All 30 observations pass it. Five content blocks per session yield $2,396/2,400$ hits. Individual-head ranges are 100% and 98.75--100% for Qwen/Llama. Supplementary $M$ reconstructions report maximum relative errors 1.022% and 1.080%.

| Model | a/S diagnostic passes | Session mean content recovery | Maximum M angle error |
|---|---:|---:|---:|
| Qwen0.5B |6/6|100.000%|0.030rad|
| Llama1B |24/24|99.688–99.844%|0.037rad|

The diagnostic does not include M error. The session-level a/S and content
metrics remain distinct from full parameter recovery. Sources are retained in
`results/new_evidence_20261102/` and the numerical diagnostic directories.

## Historical error diagnostics and timing denominators

The auxiliary CPU Qwen/Llama follow-up covers 37,466/82,393 misses (45.47%), with matching seed counters. Source-and-seed layout reconstruction confirms marker selection in all saved failing blocks. Absent-token median gaps are 0.0005/0.1611. Llama gaps at least 0.0101 account for at least 66.96% of misses. Historical block predictions were not retained. The 306-block nine-path check retains 2,754 prediction arrays and 2,448 unchanged comparisons.

GPU known-key controls reuse those 306 blocks plus 140 Llama failures across all seven failing heads. All 446 forward outputs match the official code bitwise. Fixed candidates from recorded truths/predictions yield 28 bf16 positional misrankings, versus zero with FP32 or no marker. Sampled Llama marked-row vector error has median 77.62%, versus 1.025% without the marker. Auxiliary CPU controls retain 40 misrankings. Complete coordinate traces are online.

Five-repeat GPU probe-session medians are 0.853/1.847 seconds for Qwen/Llama, including probe construction, model prefill, protection and CPU calibration. Loading, threshold fitting and dictionary preparation are separate. Auxiliary CPU Qwen/Llama decoding takes 43.4/120.0 seconds per 1,000 target tokens (1,560,928/1,538,464 positions across seeds), including diagnostics and host contention. Five-repeat Qwen/Llama probe-session medians are 13.266/32.105 seconds for construction, prefill, protection and calibration. Loading/configuration/dictionary preparation is separate. Epoch duration is unmeasured. Full phase details are online.

## Native controls and live engine details

The single-GPU integration uses vLLM 0.8.5.post1, its V0 UniProc executor, FlashAttention, eager execution and bf16 model caches. Each of four model/block-size cells has 100 functional inputs, 20 shared-prefix pairs, 10 isolation pairs and 40 tamper cases. All 520 functional cases preserve logits and six generated tokens exactly; all 160 tamper cases stop before model execution. Referenced GPU pages are zeroed between calls, then authenticated and restored before consumption. A 32-token public prefix shares physical pages; private identities are scoped before scheduling. Forty functional cases also transition epochs after two tokens.

The separate Qwen0.5B/$b=16$ attack uses the same 100 frozen inputs and obtains 41,344/41,344 head--token hits from actual protected pages. Official KV-Cloak obfuscation uses FP32. Its defender-side ordered restoration additionally holds the permutation and actual-matrix inverse; these are not supplied to the attacker. This adapter preserves six-token outputs on 90/100 inputs; its logits differ from the unprotected path. VeilKV's byte-exact results and the attack's content-recovery results are reported separately. Native Llama-3.2-1B block-size controls at 16/32/64 preserve all 600 greedy decisions per size. The matching MMLU and SQuAD runs retain 14,042 and 10,570 items; scores are unchanged from plain inference (33.451% macro accuracy and 45.714% exact match). Real TEE deployment remains untested.

With bf16 prefix caching and temperature zero, repeated Qwen0.5B outputs agree on 93/100 inputs at block size 16 and 92/100 at size 32. Three independent reruns reproduce both complete output transcripts at each size. Disabling prefix caching yields 100/100 agreement at both sizes in all three runs. At all 15 first-divergence events, competing selected tokens lie in the near-tied leading score group (top-two log-probability gaps 0--0.125). An FP32 control also yields agreement on all ten diagnostic inputs at both sizes. These controls identify prefix-cache-associated bf16 numerical sensitivity: small score changes can alter greedy choices near ties. The live integration retains its paired plain controls and exact byte/logit comparisons.

The repeat-diagnosis B label was corrected after source inspection: only
one reset occurred before each pair, so B repeats C rather than measuring
cold–cold. The paper uses the valid A/C/E evidence. Full corrected diagnostic
scope is in `GPU_RESULTS.md` and the repeat run's README. No further experiment
is required by these retained claims.
