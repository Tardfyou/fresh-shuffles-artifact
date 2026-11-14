# Accepted GPU results

GPU measurements are primary; CPU measurements remain auxiliary. The same
3,000-prompt WildChat manifest and frozen focused subsets are reused. Hardware
is A100-SXM4-80GB, with PyTorch2.6.0+cu124 and Transformers4.51.3.

The current focused native results supersede the old100/300-prompt GPU
cohorts. See [GPU602_NATIVE.md](GPU602_NATIVE.md) and the per-job overrides in
`results/gpu/index.json`. Main recovery, ordering, byte validation, synthetic
privacy, timings and the original protected live integration retain their
accepted inputs. The older counts below describe that original handoff.

## Navigation

- `results/gpu/index.json`: accepted runs.
- `results/gpu/accepted/a100-80g-20261110-01/<job>/`: main/focused native
  studies, byte validation, block-size controls, utility and timings.
- `results/gpu/accepted/vllm-live-a100-20261110-01/<job>/`: actual engine
  integration, protected-page consumption and content recovery.
- `results/gpu/summaries/<run>/`: acceptance, recounts and plot inputs.
- `sources/gpu-native/` and `sources/vllm-live/`: executable runners,
  dependencies, source interfaces and frozen inputs.

Read each job's numbered `records-*.jsonl.gz` parts in sorted order. Parts
cap expanded size at48MiB, except any individual larger observation. Compression
and machine-path normalization preserve all input tokens, predictions, counts
and valid negative outcomes. Startup jobs are `smoke_*`, separate from formal
counts. Logs, databases and private failure traces stay outside the artifact.

## Profiles

The nine-model/eleven-configuration FP32-protection suite records
91,611,072/91,611,072 hits. Qwen3-14B/32B each record12,480,000/12,480,000.
The14B FP32 run is full-model;32B FP32 uses native first-layer execution.
Both also have full-model BF16 inference with FP32 protection. Corrected
calibration runs on CPU; model cache generation and target decoding run on GPU.

Separate BF16-protection/legacy GPU bridges reach99.999744% on Qwen0.5B,
99.708467% on Llama1B,100% on14B and99.983558% on32B. Precision and
calibrator differ from primary results. All93 formal and60 startup jobs are
complete; the hardware change alone does not explain the difference from CPU.

Native byte validation passes48,160/48,160 requests and96,320 consumer
comparisons. Each large model now adds602 successful byte consumers. Known-key
S/a validation passes23,899/72,240 segments:11,952 original and11,947 refreshed;
functional failures occur in both conditions.

## Actual vLLM integration

vLLM0.8.5.post1 uses V0 UniProc, FlashAttention, eager execution and BF16
engine caches. Four cells cross Qwen0.5B/Llama1B with physical blocks16/32.
Each has100 functional inputs,20 shared pairs,10 isolation pairs and40 tamper
cases. All520 functional cases preserve logits and six generated tokens exactly;
all160 tamper cases reject before model execution. Actual GPU pages are zeroed
between calls, then authenticated and restored for consumption. A32-token public
prefix shares physical pages; private cache identity is scoped before scheduling.
Forty functional cases also transition epochs. Formal cases total780;50 startup
cases are separate.

The Qwen0.5B/b16 attack recovers41,344/41,344 head–tokens on100 targets
from real protected Value pages. Official KV-Cloak obfuscation and storage use
FP32. Its defender-side ordered restore additionally holds the permutation and
actual-matrix inverse; neither enters the attacker. Its six-token output agrees
with plain inference on90/100 targets, and its logits differ. This functional
result is separate from100% content recovery. The unprotected repeat-output baseline is retained. Three602-prompt reruns per block give555/602 and539/602 agreement;
disabling prefix caching yields602/602 in all six runs. FP32 eliminates differences on the10 diagnostic
inputs at both blocks. The15 selected original diagnostic first-divergence events involve near-tied leading
score groups (gaps0–0.125). This supports prefix-cache-associated BF16 greedy
sensitivity. See `results/gpu/accepted/vllm-repeat-diagnosis-20261110-01/README.md`
for the B-to-C condition correction. It is not a protected-integration correctness
test and does not relax the exact paired VeilKV checks.

## Timings

Llama3.1-8B/FP32/512tokens has128MiB of KV data. Median protect/recover
latencies are445.39/327.29ms for VeilKV,32.74/44.33ms for unfused KV-Cloak
and21.63/32.71ms fused; plain prefill is499.92ms. VeilKV includes CPU crypto
and GPU–CPU transfers; KV-Cloak keeps resident GPU caches. Added latency is
non-negligible. All100 repetitions and20 warm-ups are defined in the protocol.
These are latency measurements, not power measurements.

## Inspection

```bash
python tools/check_artifact_catalog.py
python tools/summarize_gpu_results.py \
  --results results/gpu/accepted/a100-80g-20261110-01 \
  --output /tmp/gpu-recount.json
python tools/inspect_gpu_results.py --root .
```

Source trees include setup guides and pinned requirements. Native rerun source
incorporates the authorized78GiB resource policy; actual record headers retain
the original science version. The author confirmed no other native source edits.
The returned live43-file baseline exactly matches issuedv2. Source versions and
actual record versions remain distinct. Multi-GPU, beam/COW, production service
and real TEE were not exercised. Native block-size/utility controls and live
engine integration are distinct studies.
