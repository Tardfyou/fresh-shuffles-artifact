# Accepted GPU results

GPU measurements are primary; CPU measurements remain auxiliary. The same
3,000-prompt WildChat manifest and frozen focused subsets are reused. Hardware
is A100-SXM4-80GB, with PyTorch2.6.0+cu124 and Transformers4.51.3.

The current focused native results supersede the old100/300-prompt GPU
cohorts. See [GPU602_NATIVE.md](GPU602_NATIVE.md) and the per-job overrides in
`results/gpu/index.json`. Main recovery, ordering, byte validation, synthetic
privacy retains its accepted inputs. The separate live602
integration is also accepted; see GPU602_LIVE.md. The original single-fixture GPU timings are replaced by the accepted stratified study below.

## Navigation

- `results/gpu/index.json`: accepted runs.
- `results/gpu/accepted/a100-80g-20261110-01/<job>/`: retained main recovery,
  probe, ordering, privacy, byte validation, utility and calibration studies.
- `results/gpu/accepted/gpu602-native-nonlive-20261114/<job>/`: current
  focused native studies and unprotected engine references.
- `results/gpu/accepted/gpu602-vllm-live-20261114/<job>/`: actual engine
  integration, protected-page consumption and content recovery.
- `results/gpu/summaries/<run>/`: acceptance, recounts and plot inputs.
- `results/gpu/accepted/latency-lengths-1k-20261115/<job>/`: current
  length-stratified timing observations.
- `results/gpu/accepted/vllm-repeat-diagnosis-20261110-01/`: original
  selected numerical diagnostics, separate from the current whole-cohort results.

| Source tree | Role |
|---|---|
| `sources/gpu-native/` | Frozen original main execution sources; use retained jobs listed in the result index |
| `sources/gpu602-native/` | Current50-job focused native plan and inputs |
| `sources/vllm-live/` | Current602-prompt protected engine plan and inputs |
| `sources/latency-lengths-1k/` | Current20-job stratified latency plan and inputs |
| `sources/vllm-repeat-diagnosis/` | Original unprotected engine diagnostics |

These source trees include their dependency files and frozen inputs. The
original93-unit plan and its source version remain as historical execution
metadata; do not use that plan's retired smaller-cohort or fixed512-token jobs
as the current study. The included sources are sufficient without an external
execution ZIP. Scientific baseline ZIPs, including their embedded CPU reference
archives, are required by source-lock verification and are retained unchanged.

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
calibrator differ from primary results. The original execution plan contained
93 formal units and60 startup units. After the authorized replacements, the
original result directory retains53 formal and39 startup jobs. The native602
run adds50 formal and10 startup jobs; live602 adds5 and5; the current timing
run adds20 and20. Selected original engine diagnostics remain separate.
The hardware change alone does not explain the difference from CPU.

Native byte validation passes48,160/48,160 requests and96,320 consumer
comparisons. Each large model now adds602 successful byte consumers. Known-key
S/a validation passes23,899/72,240 segments:11,952 original and11,947 refreshed;
functional failures occur in both conditions.

## Actual vLLM integration

vLLM0.8.5.post1 uses V0 UniProc, FlashAttention, eager execution and BF16
engine caches. Four cells cross Qwen0.5B/Llama1B with physical blocks16/32.
Each has602 functional inputs,301 shared pairs,602 isolation pairs and4816 tamper
cases. All6020 functional cases preserve logits and six generated tokens exactly;
all19264 tamper cases reject before model execution. Actual GPU pages are zeroed
between calls, then authenticated and restored for consumption. A32-token public
prefix shares physical pages; private cache identity is scoped before scheduling.
Forty functional cases also transition epochs. Formal cases total25886;50 startup
cases are separate.

The Qwen0.5B/b16 attack recovers359,936/359,936 head–tokens on602 targets
from real protected Value pages. Official KV-Cloak obfuscation and storage use
FP32. Its defender-side ordered restore additionally holds the permutation and
actual-matrix inverse; neither enters the attacker. Its six-token output agrees
with plain inference on562/602 targets, and its logits differ. This functional
result is separate from100% content recovery. The unprotected repeat-output baseline is retained. Three602-prompt reruns per block give555/602 and539/602 agreement;
disabling prefix caching yields602/602 in all six runs. FP32 eliminates differences on the10 diagnostic
inputs at both blocks. The15 selected original diagnostic first-divergence events involve near-tied leading
score groups (gaps0–0.125). This supports prefix-cache-associated BF16 greedy
sensitivity. See `results/gpu/accepted/vllm-repeat-diagnosis-20261110-01/README.md`
for the B-to-C condition correction. It is not a protected-integration correctness
test and does not relax the exact paired VeilKV checks.

## Timings

Current timings use the original3000 texts, sampled uniformly with replacement
within five actual-token length groups for each pinned model tokenizer. Four
profiles cover Llama3.1-8B/FP32 at blocks16/32/64 and Qwen0.5B/bf16 at block16.
Each method/operation/group has1000 timed calls, plus20 warm-ups. The full
return has180000 formal and360 startup observations; all are retained.

On the primary Llama/block16 profile, median VeilKV protection/recovery rises
from103.17/37.05ms in the32–63-token group to554.69/494.02ms in the512–767
group. Unfused KV-Cloak medians span32.44–33.05/50.38–52.06ms across groups.
Times are raw milliseconds per call; normalized times use actual lengths, with
no pooling across models, precisions, block sizes or length groups.
CPU cryptography and GPU–CPU transfers remain inside byte-profile timings.
See [LATENCY_LENGTHS_1K.md](LATENCY_LENGTHS_1K.md) for the paired-input rules,
timing boundaries, per-profile data and offline inspection.

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
The live602 return did not include an actual server source or baseline snapshot;
its supervisor attests unchanged science files and a resource-only8-to32 CPU-thread
change. The declared43-file smoke lock passes, and the returned records bind to
the issued602 inputs, version and unchanged inference protocol. Actual server
source byte comparison is not claimed. The distributed live source uses the602
inputs and the declared32-thread setting; the runtime snapshot was not supplied. Multi-GPU, beam/COW, production service
and real TEE were not exercised. Native block-size/utility controls and live
engine integration are distinct studies.
