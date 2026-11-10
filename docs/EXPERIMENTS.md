# Experiment inventory

The current manuscript uses accepted GPU measurements as primary; see
[GPU results](GPU_RESULTS.md). The following retained sections document the
auxiliary CPU studies and their original outcomes.

The primary input manifest has 3,000 distinct WildChat prompt texts. Focused
studies reuse saved IDs, including the 602-ID ordering cohort. Privacy-value
identification uses a separate 100-case synthetic candidate distribution.
Repeated heads, seeds, sessions and analyses do not add independent users.

| Study | Recorded scale | Evidence target |
|---|---|---|
| Main recovery | 3,000 prompts, current 11 configurations; historical 12 retained; two seeds | Content-multiset recovery across the current nine-model scope |
| Probe budget | 100 independent configurations at eight lengths | State coverage and calibration budget |
| Fused path | 300 prompts, two KV heads | Fused/public-model separation |
| Scaling | Same 100 prompts at four Qwen and two Llama ratios | Six empirical unchanged/fitted comparisons, retaining regressions |
| Lifecycle | 100 prompts, three paired conditions | Same-epoch, stale-secret and recalibrated behavior |
| Cross-key | 100 prompts, current two-model scope; historical three retained; five configurations | Sensitivity to independently generated secrets |
| Privacy identification | 100 synthetic cases, 100 candidates each, six value types | Template-conditioned candidate identification and matched controls |
| Row-secret refresh | 100 prompts, fixed column transform | Stale-demixer transfer after S/a refresh |
| Integrity battery | 12 mutation classes, 100 rounds each | Selected authentication and context rejection checks |
| Arithmetic-profile identification | 100 targets, 100 candidates, two models | Probe-free candidate identification under the arithmetic profile |
| Local ordering | 602 IDs, four conditions | Ordering of recovered bags using a fixed public prior |
| GPT ordering | 602 outputs per model, two GPT models | Global-bag free-text reconstruction |
| Entity-swap compatibility | 553 pairs, 1,106 targets | Given-candidate compatibility and equal-bag controls |
| Prototype battery | Two models, seven completed measurement stages | Original functional, prefix, paging, integrity and timing checks |

Supplementary records include paired-calibrator checks, fully logged probe
sessions and separate M diagnostics, three-configuration attribution/counter
audits, 306 selected blocks under nine numerical paths, first/all-head candidate
comparisons, five complete-session timing repetitions per model, 12,000 known-key
row-secret segment checks, and expanded byte-cache validation on the existing
OR602 cohort (48,160 requests and 96,320 consumer comparisons).

Software-ledger tests retain 100 rounds of eight-process contention, 3,200
competing operations, restart replay and resource checks. Matched cache-operation
timings retain all 800 calls (eight operations, 100 each) on one fixed 512-token
fixture. Cosine distributions reuse the already recorded 602 outputs in each of
six conditions; they are derived metrics, not new user samples.

The fixed-cohort raw records, older accepted outcomes and follow-up files remain
separate. See `metadata/audit/AUDIT_STATUS.json`, the export manifests and the offline inspectors
listed in the README. Model, head and candidate-access differences are recorded
with each study. These retained measurements use CPU or external APIs. GPU runs are cataloged
separately after acceptance; see `docs/GPU_RESULTS.md`. No generalized scaling or unique error-cause guarantee is asserted.
