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
