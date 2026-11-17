# PermLLM two-dimensional permutations and native three-party signature checks

Pinned upstream: zfscgy/PermLLM at
1b120aeb65f18744a1cd73dd419dc15f39a73971, corresponding to
PermLLM: Private Inference of Large Language Models within3 Seconds under WAN
(arXiv2405.18744). The original implementation is unchanged. This directory
contains our drivers, registered protocol and reused inputs. Upstream has no
explicit redistribution license; fetch the fixed revision rather than copying
its modules into the release.

Reuse the first study's120 legal text fixtures and public Llama1B layer0 inputs:
10 lexical anchors, b=4/8/16/32, three repetition patterns and FP32/bf16.
There is no new user cohort or model forward pass.

## Measurements

The actual LayerNorm/Softmax path calls permute_2d_with_seed: independently
permute coordinates within every row, then permute rows. Raw outputs therefore
remain diverse. Measure both complete raw-output diversity and the diversity
of the public invariant obtained by sorting values within each row. A b-state
signature does not imply a b-state complete ciphertext space.

Execute the original SS_ElementWise__RandPerm, SS_Perm and
SimulatedCommunication three-party offline/online flows. A transparent LayerNorm
callback records the permuted_x received by n1, then returns the unchanged
F.layer_norm result. Sorted rows of the attacker's own known probe are prototypes.
Before observing outputs, fix the signature-error bound at one quarter of the
minimum prototype separation. Retain per-case signature stability and LayerNorm
recovery errors, with the1e-3 functional threshold. The generic API uses width2048
and native float32 sharing; this is not complete ChatGLM6B inference or deployment
on separate machines.

## Reproduction

From the repository root, fetch the pinned upstream without modifying it:

```bash
git clone https://github.com/zfscgy/PermLLM.git /tmp/permllm-upstream
git -C /tmp/permllm-upstream checkout 1b120aeb65f18744a1cd73dd419dc15f39a73971
```

The experiment uses Torch and numpy; accepted records use Torch2.14.0+cpu.
Transformers4.27, ChatGLM weights and Pyfhel are unnecessary: they belong to
complete-model/HE protocols and are not imported by these operator and generic
nonlinearity checks.

```bash
python run_check.py --upstream /tmp/permllm-upstream \
  --inputs /artifact/results/orbit_verification_20261111/public_embedding_rows.json.gz \
  --output /tmp/permllm-orbit-results
python audit_check.py --upstream /tmp/permllm-upstream \
  --inputs /artifact/results/orbit_verification_20261111/public_embedding_rows.json.gz \
  --results /tmp/permllm-orbit-results
```

Use a new output directory and preserve previous records. The auditor
independently reconstructs every two-dimensional permutation and public
signature and replays the native three-party flow. Retain ordinary functional
failures; a100% result is not a reason to repeat collection.
