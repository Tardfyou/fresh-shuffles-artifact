# Real vLLM KV-page integration: single-GPU supplementary study(v2)

This standalone bundle is independent of the original main GPU bundle. Model
inference runs only through CUDA/vLLM. Reuse the existing ordering602 inputs;
do not collect another user cohort or overwrite an active server bundle.

## Contents and scale

| Component | Fixed scale | Validation |
|---|---|---|
| VeilKV defense | Qwen0.5B/Llama1B x physical blocks16/32 | Per cell:602 ordinary continuations,301 shared-prefix pairs,602 same-private-text isolation pairs,4,816 tamper cases |
| Attack compatibility | Qwen0.5B/block16,602 original inputs | Repeated-input calibration and content recovery at the real KV-page protection boundary |
| CUDA startup checks | Four defense cells and one attack cell,50 cases total | Engine API, page layout, authentication, actual consumption, sharing and record accounting |

The formal queue has25,886 cases. Normal cases generate six greedy tokens;
independent plain and duplicate-plain references must match exactly. The public
prefix has32 tokens from the original lists, followed by the existing no_pad
fixture body; construction is specified in protocol/INPUTS.json. Retain every
input, prediction and normal negative result.

## Actual integration scope

Pin vLLM0.8.5.post1 with V0/UniProc, FlashAttention and eager execution. The worker
reads real physical cache pages and request block tables. After each model step,
it protects valid KV values as host objects and clears the corresponding GPU
pages. Before the next step, authenticate every relevant object and restore its
original physical pages, which the model actually consumes. Record trusted CPU
cryptography and transfer boundaries; do not substitute CPU model experiments.

Apply cache scope before scheduling. The first32 policy-public tokens may share
across principals; private cache identity additionally includes principal and
request. Record actual num_cached_tokens and per-step physical block tables.
Shared pages must coincide and private pages remain disjoint. VeilKV16-token
cryptographic blocks form16/32-token physical pages.

The attack observes the same real-page protection output; secrets and
permutations are not supplied to recovery. A separate legitimate restore adapter
holds the permutation and uses the actual matrix inverse to recover physical row
order. This is an integration adapter, not an upstream-native deobfuscate claim.
Protection still calls official obfuscate. FP32-protected output remains FP32
instead of being requantized to BF16. Report legitimate continuation differences
separately.

The study covers one GPU, greedy generation, host protection storage,
authenticated restoration, real PagedAttention consumption, prefix reuse and
request isolation. It does not cover multiple GPUs, beam/COW, engine CPU swap,
real TEE hardware or production serving. CPU synthetic tests do not replace the
CUDA startup checks.

The v1 source guard produced a false alarm; v2 corrected it and retained the
original state. The original issued bundle includes FIX_V2_ZH.md recovery notes.

## 1. Upload, unpack and configure the environment

Upload veilkv-vllm-live-20261110-v2.zip and run:

```bash
unzip ~/veilkv-vllm-live-20261110-v2.zip
cd veilkv-vllm-live-v2
```

Set the GPU index, existing model cache, output directory and resource fraction
in runtime/settings.json. Reuse the main bundle's vLLM environment:

```bash
export VEILKV_VLLM_PYTHON=/path/to/main-bundle/.venv-vllm/bin/python
bash scripts/bootstrap.sh
```

Alternatively, create a separate environment:

```bash
bash scripts/bootstrap.sh
export VEILKV_VLLM_PYTHON="$PWD/.venv-vllm/bin/python"
```

Pin Torch2.6.0, HF4.51.3 and vLLM0.8.5.post1. Environment/path/resource adaptations
must not change the scientific API implementation. Source preflight without
loading a model is available:

```bash
"$VEILKV_VLLM_PYTHON" -m integration.api_guard
```

## 2. Run startup checks when the GPU is free

The main queue has priority. Wait for other compute processes and do not stop
them. This study loads only0.5B/1B models, not14B/32B. Adjust the vLLM resource
fraction rather than imposing the older bundle's70GiB hard cap.

```bash
"$VEILKV_VLLM_PYTHON" -u -m integration.driver --phase smoke
```

After all50 cases and byte-defense/authentication gates pass, write
SMOKE_GATE.json. Preserve ordinary attack errors and KV-Cloak numerical failures;
record accounting does not require100% attack recovery.

## 3. Formal execution and progress

```bash
nohup "$VEILKV_VLLM_PYTHON" -u -m integration.driver --phase all > runtime/queue.log 2>&1 &
"$VEILKV_VLLM_PYTHON" -m integration.manage status
```

`--phase defense` and `--phase attack` may run separately. Use one GPU serially.
Repeated invocation checks and skips only complete IDs, preserving ordinary
failures. Exceptions remain in attempts.jsonl and incomplete IDs resume.

Ctrl-C may stop this bundle. Keep all files and repeat the same phase to resume.
If the last line is incomplete, preserve its fragment before repairing:

```bash
"$VEILKV_VLLM_PYTHON" -m integration.manage repair --job JOB_ID
```

On scientific-assertion or API failure, preserve the directory, stdout and full
error for review; do not weaken thresholds. Record environment adaptations in
runtime/ENVIRONMENT_ADAPTATIONS.md.

## 4. Handoff

```bash
"$VEILKV_VLLM_PYTHON" -m integration.verify
"$VEILKV_VLLM_PYTHON" -m integration.manage export
```

Return vllm-live-private-handoff.zip with original records, every actual
environment, protected probes, parameter estimates, physical-page/scope evidence,
exceptions and ledger. It is a private acceptance bundle, not an anonymous release.
After GPU acceptance, matching measurements become primary and CPU records remain
auxiliary.

Compare scientific files directly with the included baseline and critical vLLM
interfaces with pinned upstream source. Inputs, precision, model revisions,
algorithms and thresholds are fixed. Local synthetic-tensor and structural tests
do not establish actual A100 execution; the CUDA startup gate does.
