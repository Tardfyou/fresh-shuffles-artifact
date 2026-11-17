# Single80GiB GPU execution bundle: setup, execution and acceptance

This guide records the original standalone execution bundle. In the consolidated
artifact, use `docs/GPU_RESULTS.md` from the artifact root to locate current
accepted studies and their sources.

The bundle reproduces Repeat-to-Reveal and VeilKV measurements on one80GiB
NVIDIA GPU, with one job at a time by default. Samples, revisions, algorithms,
thresholds and metric definitions are fixed. The operator may supervise jobs,
adjust resources, and adapt installation and local paths.

The additional dense models are official Qwen3-14B(actual14.8B) and Qwen3-32B.
The full32B model resides in BF16 with FP32 protection. Its FP32 first-layer
measurement has separate full-checkpoint equivalence evidence and an explicit
execution-scope label. The full14B FP32 model fits on an80GiB card. No multi-GPU
or120GiB memory assumption is used.

## 1. Upload and unpack

Upload the issued ZIP to your GPU server, then run:

```bash
mkdir -p ~/veilkv_gpu
cd ~/veilkv_gpu
unzip ~/veilkv-gpu-80g-experiments-20261109.zip
cd veilkv-gpu-80g
ls
```

The original bundle contains runner, protocol, data, scripts, CLAUDE.md,
SCIENTIFIC_LOCK.json and SCIENTIFIC_BASELINE.zip. It contains no model weights,
manuscript or server password. Weights and two environments require approximately
180-220GB; allow at least250GB including outputs.

## 2. Install the native environment

No free GPU is needed for installation.

```bash
bash scripts/bootstrap_native.sh
```

This creates `.venv-native` with Torch2.6.0/CUDA12.4 and Transformers4.51.3.
Upstream fixes Torch2.6.0, Transformers4.51.1, CUDA, FP32 and eager execution.
This bundle retains Torch and uses the two-patch HF adaptation needed by its
Qwen3 code. The CUDA build works on A100 with the580-series driver; do not
replace the system driver.

If Conda is unavailable, the script uses the existing Python to create a venv.
Handle mirrors, proxies and cache paths within the environment. Do not change
scientific dependencies, precision or algorithms to bypass a gate.

## 3. Download weights before the GPU becomes available

```bash
.venv-native/bin/python -m runner.download_models --workers 2
```

Repositories and revisions are in `protocol/models.json`; downloads can resume.
The14B/32B checkpoints total approximately95GB, and all main models plus the8B
latency model total approximately160GB. Set `cache_dir` in runtime/settings.json
to a disk with enough space. Do not change model IDs/revisions or use quantized weights.

The fixed MMLU and SQuAD inputs contain14,042 and10,570 items. Prompt formats
match the original work; SQuAD demonstrations use a fixed seed. The original
behavior that skipped erroneous items has been removed. The read-only
reference/accepted_cpu_artifact.zip retains prior CPU evidence; do not run its
historical entry points.

## 4. Install the separate vLLM environment

```bash
bash scripts/bootstrap_engine.sh
```

This environment runs unprotected vLLM inference and repeated-prefix baselines;
it does not affect native execution. The original work did not release a
protected-cache vLLM integration. Its PagedAttention compatibility check uses
native HF precision and timing at block sizes16/32/64. This bundle treats those
checks and the vLLM baseline separately. The latter is not evidence that VeilKV
or KV-Cloak is integrated with vLLM; the separate live-integration source tree
provides that study.

## 5. Inspect the runtime when the GPU is available

```bash
nvidia-smi
.venv-native/bin/python -m runner.doctor --gpu 0
```

doctor runs only256x256 FP32/bf16 operations and records the GPU, driver and
dependencies. GPU0 is the default. If another card is used, set the single entry
of `gpu_ids` in runtime/settings.json to its index.

## 6. Run the startup gate

```bash
.venv-native/bin/python -u -m runner.supervisor --phase smoke
```

The complete startup queue must pass before results/SMOKE_GATE.json permits
formal execution. Ordinary negative attack/scaling/S_a results do not fail the
gate. Correctness checks instead cover independent plain references, source
consistency, byte restoration and record accounting. Return scientific-code
errors for correction; do not relax thresholds or remove records.

The original queue has93 execution units, including shards and precision
controls, not93 independent studies. Its60 startup jobs cover each model and
major precision/block-size path. The original per-job schedule is documented
in audit/GPU_QUEUE.csv in the issued bundle.

## 7. Run the formal queue

```bash
nohup .venv-native/bin/python -u -m runner.supervisor --phase all > runtime/all-monitor.log 2>&1 &
```

Alternatively, run phases sequentially, waiting for each to finish:

```bash
.venv-native/bin/python -u -m runner.supervisor --phase core
.venv-native/bin/python -u -m runner.supervisor --phase performance
.venv-native/bin/python -u -m runner.supervisor --phase official_utility
.venv-native/bin/python -u -m runner.supervisor --phase engine_baseline
```

One GPU job runs at a time by default. Wait for a free card and sufficient memory;
do not stop other users' processes. Retain a nominal2GiB reserve and a78GiB
per-process allocation cap. Large-model and timing jobs are exclusive. Small-model
concurrency may be adjusted from1 to2 using max_concurrent_jobs; large models
must not reside together, and multiple GPUs are not supported.

The FP32 primary reproduction uses the corrected calibrator. Historical CPU
precision bridges retain separate legacy/bf16-protection conditions. Full large
models use BF16 weights with FP32 protection and explicit profile labels.
Do not pool different profiles or select only favorable results.

## 8. Inspect, pause and resume

```bash
.venv-native/bin/python -m runner.status
```

Inspect record counts, expected counts, states and ETA from recent measured
progress. Do not infer A100 duration from CPU speed or treat a live PID as proof
of a healthy experiment.

Pause after the current complete observation is saved:

```bash
bash scripts/pause.sh
```

Resume the same queue, skipping only IDs already measured completely:

```bash
bash scripts/resume.sh all
```

If the GPU must be released immediately, stop only this bundle's verified worker PIDs:

```bash
.venv-native/bin/python -m runner.stop_owned
```

Keep all original output. If writing stopped within the last line, preserve the
fragment and repair transport before resuming the same ID:

```bash
.venv-native/bin/python -m runner.repair_transport --job JOB_ID
```

Retain errors, OOMs, download interruptions and incomplete observations.
Do not overwrite normal wrong predictions, empty outputs or functional failures.

## 9. Verify and export

```bash
.venv-native/bin/python -m runner.verify_outputs --completed-only
.venv-native/bin/python -m runner.summarize
.venv-native/bin/python -m runner.export_results
```

The final command creates gpu-results-private-handoff.zip with raw records,
errors, logs and resumable execution state. It is a private acceptance handoff,
not the anonymous release. Do not return only a summary or promote unaccepted
results directly into the manuscript.

## Scientific implementation and data

- The original audit/EXPERIMENT_AUDIT_ZH.md documents CPU/GPU boundaries, scales
  and execution plans for21 study types.
- Byte-defense format v2 adds model/tokenizer/RoPE binding for private-only tails;
  prior CPU evidence retains v1. Fixed test keys are used for known-key software checks.
- Copy inputs before protection to prevent upstream in-place operations from
  contaminating another seed/condition. Save each invocation environment;
  CPU records cannot complete GPU jobs.
- Diagnostic records retain actual calibration estimates, effective defender
  parameters, permutations and failed-block matrices. Ground-truth scoring
  fields are computed after prediction and are not supplied to the decoder.
- Native timing synchronizes CUDA at both boundaries. Trusted CPU cryptography
  and D2H/H2D transfers are included; cache cloning, nonce preparation and
  recovery-input preparation remain outside timing for every method.
- Frozen scientific files are compared directly with the included read-only
  baseline. The queue refuses mismatches. Environment adaptations belong only
  in runtime/ and the two bootstrap scripts.
- Local logic checks do not constitute GPU measurements. The fixed CUDA startup
  gate verifies execution on the actual A100.

See protocol/NUMERICAL_PRECISION.json. Primary protection uses FP32; full BF16
14B/32B models also use FP32-protected caches to avoid the observed marker-rounding
loss. TF32 and low-precision BF16 reductions are disabled. Historical BF16
protection bridges remain separate.
