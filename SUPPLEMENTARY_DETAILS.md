# Supplementary implementation and measurement details

This document accompanies the retained code and data. No new experiments are
introduced here. Every original record and outcome keeps its original scope.
`results/supplementary_reporting_values.json` retains the ten checkpoint
identifiers and twelve configuration-level diagnostic rows.

## Environment and historical version scope

The main runs used Python 3.12.3, PyTorch 2.14.0 CPU, Transformers 4.51.3,
AMD EPYC 9554 and Linux 6.8.0. Their 36,000 prompt records, 72,000 seed
counters and 1,440 progress checkpoints reconcile. The checkpoint inventory
was observed during later preparation. Historical run-bound model/tokenizer
revisions and a contemporaneous dependency lock are unavailable; the inventory
must not be treated as their reconstruction.

The fixed manifest is in `results/main_public/`. Focused records are under
`results/takeover_20261029/corrected/`; acceptance and scope are recorded in
`AUDIT_STATUS.json`. The frozen 602-ID ordering selection supersedes the old
300-ID annotations without rewriting the original rows. Model-dependent paths
use the corresponding input normalization and projection; the grouped-query
head layout does not increase the number of independent prompts.

## Defensive byte format and runtime

The byte profile uses AES-128-CTR with encrypt-then-MAC. HKDF derives a
16-byte encryption key, 32-byte MAC key and 16-byte IV. The 16-byte HMAC tag
covers ciphertext and associated data and is verified before decryption.
The decoder materializes FP32; the trusted runtime adapter restores the native
consumer dtype after authentication.

Epoch derivation uses an epoch domain. Public-object scope includes model,
tokenizer, RoPE, token prefix, layer, K/V role, block index, dtype and policy.
Private tails add principal and a fresh 16-byte request nonce. Variable fields
have 8-byte big-endian length prefixes; coordinates use big-endian encoding.
Associated data include coordinates, dtype, epoch, policy, preceding tag and
final private sequence length. The chain extension binds batch size, head
count/dimension, layer count and profile. The public terminal tag starts the
private chain. The SQLite freshness ledger claims nonces before encryption,
rejects conflicting public payloads and persists across restart. Rollback
protection is a trusted assumption; hardware key release is simulated.

## Matched control and display conventions

The lifecycle control uses 100 fixed targets and both heads. The token-frequency
prior is fitted on 501 other OR prompts after excluding the evaluated IDs and
identical complete-block spans. Its exact expected multiset overlap sums
E[min(X_t,n_t)] over tokens, where X_t is Binomial(16,p_t), n_t is the target
multiplicity and p_t the fitted frequency. No target is excluded.

The percentage color-bin cuts are 0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9
and 99.99. Fractional-error/cosine cells use the same magnitude scale after
multiplication by 100. Counts, intervals and unavailable entries remain white.
These colors do not supply a separate security classification.

## Earlier defensive conditions and valid failures

The original seven-stage battery contains shared-prefix K/V maximum errors
0.49896/0.031205 versus FP32, matching its unprotected bf16 reference.
The earlier storage-precision fixture has 4/4 FP32 exact recoveries and 3/4
bf16-versus-FP32 greedy continuations. They are distinct from comparisons to
native-bf16 cached baselines. Arithmetic-profile own-scope ciphertext equality
is 0/100 for each model.

The fixed100 byte study gives 1,800/1,800 FP32 transitions. Without the runtime
dtype adapter, 600 initial native-bf16 transitions fail dtype consumption and
1,200 later segments are not run. The adapted condition passes all 1,800 native
bf16 transitions, preserving values, dtype and tensor bytes. All conditions keep
the same fixtures and masks/positions. The separate 12,000-check S/a study keeps
4,597 passes and 7,403 failures; its 63 discordant paired segments preclude a
claim of identical functional behavior.

The later fixed100 byte follow-up has 400 requests and 1,200 consumer segments,
200 shared-prefix cases, 200 single-reprotection cases and nine ledger scenarios.
Protected/plain state and logit differences are zero. Full-forward/cached baseline
maxima are 0.000066/0.000044 (FP32 Qwen/Llama) and 0.625/0.296875 (bf16);
these are different comparisons. The expanded OR602 study retains the same
conventions. Its FP32/bf16 baseline maxima are Qwen 0.00042617/1.0469 and
Llama 0.000086457/0.65625. Known-key software validation does not measure a
production scheduler or hardware trust boundary.

## Cache-operation timing details

The matched workload uses one existing 512-token OR602 prompt, four threads,
bf16 storage, 20 warm-ups and 100 calls per operation. Public/private lengths
are 32/480 for shared-prefix storage and 0/512 for no sharing. Operations are
interleaved in seeded shuffled order. Fresh inputs and nonces are prepared
outside timing. Trusted dtype restoration and explicit SQLite connection close
are included. One-time weight fusion costs 0.071 seconds outside timing;
its final-token FP32 logit difference is at most 0.00002. Known-key bf16
unmasked-attention errors of the two official paths are 0.5166/0.4943.

All 800 calls remain. A separate historical arithmetic benchmark reports
11.4017/11.7487 seconds per 1,000 tokens for protection/recovery. That benchmark
has its own workload and does not supply a matched reconstruction-cost ranking.

## Diagnostic accounting and comparison history

The two-model diagnostic follow-up retains median relative sign residuals
0.00058/0.00091, minimum pair norms 0.00319/0.00358 and median head-mean
projection residuals 0.110/0.069 (worst-row 1.045/1.053). The twelve configuration
extrema in the JSON are not per-error attribution.

Earlier logs have 68 worker events on seven head identities and 44 later
attempt rows. The initial low-margin policy logged but did not execute the
change rule; the older later-attempt rows lack submitted arrays. A separate fully
logged execution contains three sessions per model, 126 session-head observations
and 116 distinct input-head observations. Seven of its 44 event inputs equal
primary and 37 differ. Threshold-clear counts are 0/2, 2/5 and 11/18.

The a/S predicate checks cosines at least 0.999 and bijective column matching;
it does not test M. Five target blocks per session give 10,020/10,080 head-token
hits. Individual-head recovery ranges are Qwen 100%, Llama 98.75–100% and
Phi 93.75–100%. Supplementary M reconstruction reports maximum matrix-relative
errors 1.022%, 1.080% and 11.413%, with 32/32/48 coordinate pairs and no skipped
pairs. In the Phi primary session, four of the five largest-angle-error heads
have misses, while head 17 recovers all five blocks. This is partial overlap,
not a demonstrated causal explanation.

Three configurations cover 471,739/516,666 misses. All Qwen/Llama historical
seed counts match; six Phi counts differ by one with zero net change. Historical
per-block predictions were not saved. Absent gold-token-type counts are
5/27,931/359,548 with median recorded gaps 0.0005/0.1611/0.1314. Rounded zeros
do not establish exact ties. Gaps at least 0.0101 yield conservative missed-token
lower bounds 25,083/37,460 and 326,853/434,273. The 306-block numerical comparison
keeps all 2,754 prediction arrays and 2,448 nonbaseline comparisons; none changes.

Earlier cross-transform candidate controls retain 8/200 and 1/200 direct
no-probe hits, and 18/200 and 3/200 wrong-configuration hits. The paired seeds
reuse 100 targets. Separate arithmetic-profile Gram/row-norm/Procrustes controls
are 41/3/9 (Qwen) and 74/37/3 (Llama), each per 100. They are distinct from
both the other arithmetic diagnostic and the later matched-transform,
matched-head comparison. Only combined earlier control timings survive.

## Scaling and ordering record notes

The six scaling settings retain their negative results. Candidate-loss minimum
gaps are 0.04460, 0.003707, 0.000703, 0.01376 for four Qwen ratios and
0.001476/0.000322 for two Llama ratios. The ratio-1.25 Qwen gate matches all
200 old score cells. These gaps are empirical values, not sufficient conditions
for a general scaling guarantee.

The ordering cohort retains all 602 IDs, including 574 distinct truncated spans.
Cached and uncached beam execution were compared at three context lengths.
All GPT responses, including empty and budget-limited outputs, remain. Sol/Luna
use medium/high reasoning, 4,096 completion tokens and temperature zero.
Their exact spans are 4/602 and 0/602; multiset-preserving outputs are 47/602
and 0/602. Cosine uses all-MiniLM-L6-v2 with 256-token truncation and zero for
empty text. See ORDERING_STUDY.md for denominators and saved files.

Independent-column sorting plus row norms does not certify a row permutation:
[[1,1],[-1,-1]] and [[1,-1],[-1,1]] have matching sorted columns and row norms
but different row-vector multisets. K/V attention checks must also preserve the
paired permutations, masks and positions.

See EXPERIMENTS.md, DEFENSE_SUPPLEMENT.md and the retained offline inspectors
for the data inventory and validation entry points. No manuscript files or
private correspondence accompany this documentation.

## Complete supplementary tables

These tables preserve the previously reported values and measurement scope.
They add no samples or experiments. Machine-readable cells are retained in
`results/supplementary_reporting_values.json` under `additional_tables`.

### Intervention comparison

Measured intervention properties. A dash is unmeasured. Public-prefix equality is visible by policy; the trusted boundary is simulated. The first-layer entry describes the transfer-study check; separate full-model continuation outcomes are retained in DEFENSE_SUPPLEMENT.md.

| Property | KV-Cloak comparison | Request-specific S,a | VeilKV byte profile |
| --- | --- | --- | --- |
| Secret scope | Reused structural secrets in the tested configuration | Refresh S,a; retain M | Derived material scoped to cache metadata and private principal/nonce |
| Confidentiality evidence | First-layer Value multiset recovery in the tested implementation | Stale-demixer recovery 1.769% | --- |
| Integrity and replay | --- | --- | 12/12 mutation classes detected; nonce replay rejected after restart |
| Prefix and tail integration | --- | --- | Stored public prefix reused; tail linked to its terminal tag; wrong joins rejected |
| Functional boundary | --- | First-layer joint K/V attention check | 48,160/48,160 requests; FP32/bf16; five mask shapes |

### Closed-set privacy cases by type

One fixed key, 100 synthetic cases, 100 same-type candidates per case. Unique counts strict first-place truths; max m is the largest truth tie group. Top-k entries are percentages under uniform tie-breaking.

| Type | Unique | Max. m | Top-1 | Top-5 | Top-10 |
| --- | --- | --- | --- | --- | --- |
| SSN-like | 17/17 | 1 | 100.00 | 100.00 | 100.00 |
| Card-like | 17/17 | 1 | 100.00 | 100.00 | 100.00 |
| Password | 13/17 | 2 | 88.24 | 100.00 | 100.00 |
| MRN | 17/17 | 1 | 100.00 | 100.00 | 100.00 |
| Name | 16/16 | 1 | 100.00 | 100.00 | 100.00 |
| Address | 15/16 | 2 | 96.88 | 100.00 | 100.00 |

### Saved CPU phase costs

Saved CPU measurements. Setup phases are medians across 4/12 final workers with eight threads each. Decode includes earlier scheduling conditions and diagnostic overhead. Peak RSS is the process peak in reported MB, not dictionary-only allocation.

| Saved quantity | Qwen 0.5B | Llama 1B |
| --- | --- | --- |
| Vocabulary | 151,936 | 128,256 |
| Dictionary (s) | 2.121 | 11.105 |
| Calibration stage (s) | 0.107 | 0.427 |
| Decode (s/1k tokens) | 43.4 | 120.0 |
| Peak RSS (reported MB) | 4,561.6 | 7,588.1 |

## Evaluation setup and diagnostic accounting

`results/supplementary_reporting_values.json` now retains these protocol details
under `evaluation_protocol_details`. They describe the existing accepted studies
and add no samples or measurements.

The main runner generates one complete secret configuration per model/block-size
cell with seed 42. Seeds 5001 and 5002 change target-block permutations while
sharing that configuration. Eight fixed sentences, separate from the targets,
set defender magnitude thresholds. All configured main-suite heads calibrated
and enter the reported counts; whole prompts are the bootstrap units.

Probe lengths are complete tokenizer-output lengths. Searching repeated text can
overshoot a nominal budget. Recorded prefix lengths are one token for Llama and
Phi, two for TinyLlama and zero for the other main configurations. Prefixes stay
in the model input; only full blocks inside the constant run enter calibration.
Target scoring excludes the final incomplete block. Ground-truth secrets serve
only evaluation diagnostics.

The 100-key budget study reuses the same independently generated configurations
at eight budgets from 16 to 2,048 tokens. Reported 95% Wilson endpoints are
57.3–75.4% for 67/100 at 1,024 tokens and 96.3–100% for 100/100 at 2,048.
Intervals are pointwise, not a simultaneous band. The two-head coverage band
does not assume independence between heads; exact occupancy calculations use
independent uniform permutations across blocks.

The entity-swap diagnostic retains all 553 pairs and both targets (1,106 head-0
observations), including 162 equal-global-bag pairs, 25 equal-block-bag pairs,
two identical evaluated token-sequence pairs and eight pairs with different
complete-token lengths. The length-only expected accuracy is 50.723%; block/global
bag candidate matching gives 97.740%/85.353% with half credit for ties. Both
candidate presentations are shuffled per pair. The inputs include arbitrary
words and code identifiers; the task is compatibility with the two supplied
candidates.

The 100 synthetic privacy cases use 100 generated candidates per value type.
Truth indices are sampled without replacement using seed 20261028 and candidate
presentation is independently shuffled per case. Contextual tokens, recovered
blocks, scores, truth positions, control scores and tie groups are retained.
No-cache and template-only controls explicitly assign uniform priors; equal
complete-block candidate lengths make length-only scoring equivalent. Fresh
independent target configurations with the original demixer and Dice scoring
yield expected top-1/top-5/top-10 rates of 1.004%/5.019%/10.039%. All original
cases and outcomes remain in their existing result files.

## Sampling and local ordering settings

The existing fixed-cohort preparation and secondary ordering summaries are
retained in `sampling_and_ordering_details` in the reporting-values JSON.
Preparation scans shards 70–85, selects the first user turn of conversations
from January 2025 onward, and requires English language annotations with toxic
and redacted flags false. These are annotation filters, not a guarantee of
monolingual text. Surrounding whitespace is stripped; 120–4,000 characters are
retained and duplicates removed. Qwen2.5-0.5B length bins are [32,64), [64,128),
[128,256), [256,512) and [512,768), with 600 prompts per bin and seed 20261021.
The saved manifest has 802,528 Qwen tokens. Its observed conversation timestamps
span April 22–July 31, 2025. All text and timestamp/model metadata match source
shards; no texts coincide under NFKC, case folding and whitespace normalization.

Local ordering uses all 602 IDs, representing 574 distinct complete-block spans.
The corrected decoder supplies head-0 bags. The Qwen2.5-0.5B prior uses bf16,
beam width 16 and a model start token. At most 80 reconstructed context tokens
and 16 steps per block keep inputs within 96 tokens. Reordered KV caches reuse
identical prefixes. Cached and uncached execution were compared at three context
lengths; timing may reuse common prefixes across conditions.

All four local conditions remain. Per-prompt position accuracy is 27.776% E2E,
27.776% oracle, 11.283% without reconstructed context and 7.505% for random order.
E2E matches 1/602 complete spans and 2,020/11,248 blocks. Bags are exact across
all blocks in 601/602 IDs, and E2E/oracle output sequences agree in 601/602 IDs.
Equal weighting of distinct spans instead gives 28.258% E2E and 7.548% random
position accuracy. No original prediction, sample ID or measured outcome changed.
