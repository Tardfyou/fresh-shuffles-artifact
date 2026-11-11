# PermLLM native permutation/protocol signature verification

This supplements the first public-profile orbit study with a separate public
privacy protocol implementation: PermLLM, arXiv2405.18744, by Fei Zheng,
Chaochao Chen, Zhongxuan Han and Xiaolin Zheng. The official source repository
is https://github.com/zfscgy/PermLLM at revision
1b120aeb65f18744a1cd73dd419dc15f39a73971. A different same-name repository
contains model pruning and is not the implementation used here.

## Native protection path

The actual LayerNorm/Softmax caller uses `permute_2d_with_seed`: each row's
coordinates are independently permuted, followed by row permutation. Its nominal
space is b! times (d!)^b. All these permutations remain active in this study.
The code was not switched to a pure-row control mode.

We measure both exact raw-output diversity and a public invariant obtained by
sorting coordinates within each row. Sorting preserves each row's value multiset;
it removes coordinate order, rather than estimating a secret matrix. This
signature can retain content-dependent structure even while raw outputs vary.
The finite b-state result concerns the public row-signature orbit, not the full
raw ciphertext space.

We also execute the original `SS_ElementWise__RandPerm` and `SS_Perm` prepare,
offline and online methods with the original three-party `SimulatedCommunication`.
A transparent LayerNorm callback records the n1 input actually passed to the
nonlinearity and then returns the unchanged `F.layer_norm` result. No original
source module is modified.

## Frozen inputs and scope

The120 legal BOS-prefixed probes,10 lexical anchors from5 existing lifecycle
prompt IDs, three repetition patterns and public Llama1B layer0 representations
are exactly reused from `results/orbit_verification_20261111`. There is no new
corpus, full-model forward pass, victim reconstruction or key-recovery method.
The generic LayerNorm protocol is evaluated at the reused input width2048,
using the same callback type as its native caller. This is not full ChatGLM6B
inference, a network benchmark or a full MPC deployment.

The protocol was registered before measurement: b=4/8/16/32, FP32/bf16,
512 operator draws per condition and32 native-protocol draws per anchor/length.
The native sharing/masking arithmetic is FP32. Masks use the registered
x/z scales10/10. All normal observations remain, including distinct raw outputs.

## Results

| Measurement | b=4 | b=8 | b=16 | b=32 |
|---|---:|---:|---:|---:|
| Repeated-input raw outputs,512 draws |512|512|512|512|
| Public row-sorted signatures,512 draws |4|8|16|32|
| Two-group signature control,512 draws |12|226|512|512|
| Distinct-token signature control,512 draws |24|509|512|512|

The summary file is authoritative for all control counts. Every entry has the
same value across the10 anchors and both recorded precisions. At b=4 the
signature controls span all4/12/24 possible row arrangements; larger groups are
sampled rather than enumerated. There are240 operator conditions and122,880
operator observations.

The native three-party test has40 conditions and1,280 trials. All1,280
row-signature checks meet a bound fixed before output observation at one quarter
of the minimum distinct sorted-input-row separation. All1,280 LayerNorm
functional comparisons meet the registered1e-3 tolerance. Maximum sorted-row
error is7.868603283766169e-6; maximum recovered LayerNorm error is
7.47358426451683e-5.

With only32 trials per native-protocol condition, observed position counts are
4/8/14/19. These exactly equal the ideal positions present in those sampled
permutations. There is no observed noise-induced signature merging: every
recorded row tag matches its ideal counterpart. The14/19 counts are incomplete
sampling coverage, not evidence that rounding further shrinks the orbit.
This is checked in `SAMPLING_CORRECTION.json`.

## Independent verification

All raw and public-signature partitions from122,880 operator observations were
replayed, including saved output witnesses. The1,280 native protocol flows were
also replayed against recorded signature tags, distances, bounds, output errors
and message counts. No result is accepted from aggregate counts alone.

The upper-level implication is empirical support for input-dependent observable
invariants across a second public implementation. It does not establish a new
full-system break or general parameter-identification method. Existing theorem
and full-recovery claims remain scoped to the original construction.

## Reproduction

Our sources are in `sources/permllm-orbit/`, records in
`results/permllm_orbit_verification_20261111/`. The upstream repository does not
currently declare a redistribution license, so its modules are fetched at the
fixed revision rather than copied into the artifact. Its license must not be
called BSD; the BSD license belongs to the separate Hidden No More code.

```bash
bash sources/permllm-orbit/fetch_upstream.sh /tmp/permllm-orbit-upstream
python sources/permllm-orbit/audit_check.py \
  --upstream /tmp/permllm-orbit-upstream \
  --inputs results/orbit_verification_20261111/public_embedding_rows.json.gz \
  --results results/permllm_orbit_verification_20261111
```

The actual collection command is `run_check.py` with `--output`, as documented
in the source README. There is no `run_orbit.py --out` entry for this second
study. The saved environment is Torch2.14.0+cpu/Python3.12.3; a new run must use
an independent output directory. No model checkpoint is needed for replay.
