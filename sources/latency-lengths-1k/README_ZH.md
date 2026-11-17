# Five-length latency study with1,000 calls per operation

This guide records the issued performance-only execution bundle. It does not
rerun recovery, privacy, utility or live-vLLM integration. Models, precisions,
four protection methods, operations and timing boundaries match the original
performance study. Replace its single512-token fixture with five length groups
and1,000 formal calls per operation in each group.

## Sampling and counts

Reuse the main3,000 real texts. Each tested model's pinned tokenizer assigns
complete inputs to32-63,64-127,128-255,256-511 and512-767 tokens. Preserve source
text without truncation or padding its wording. Within each group, draw1,000
inputs uniformly with replacement; every method uses exactly the same schedule.
There are no content-category quotas. Repeated texts remain, so1,000 counts
timed calls rather than distinct inputs. Schedules and distinct-input counts
are frozen in protocol/SAMPLING.json and protocol/schedules/.

Retain four profiles: Llama-3.1-8B/FP32 at KV-Cloak block sizes16/32/64, and the
original Qwen2.5-0.5B/bf16/block16 bridge. Five groups per profile give20 formal jobs.

Each job measures protection and recovery for unfused KV-Cloak, fused KV-Cloak,
VeilKV with a public prefix and authenticated encryption without sharing, plus
the original unprotected-model prefill. Nine series x1,000 calls give9,000
observations per job and180,000 formal observations. Each series has20 separate
warmups. Startup checks cover every profile/group with two calls per series:
360 observations, separate from formal measurements.

## Fixed measurement rules

- Use one80GB GPU serially, with a78GiB allocation cap and16 CPU threads by
  default. Reuse the existing native environment and model cache.
- Time protection and recovery separately, synchronizing CUDA at both ends.
  VeilKV/AEAD include CPU cryptography and GPU-CPU transfers; KV-Cloak caches
  remain on GPU.
- Cache construction, configuration preparation, input copying, nonce creation
  and prior protection of recovery inputs remain outside timing. Fused VM inputs
  are also prepared outside timing; this does not measure fused-model prefill.
- Build each sampled input's actual cache. Process the fixed32-token public
  prefix canonically and then the private text, reusing the accepted byte-consumer
  construction. This avoids floating-point byte conflicts for identical prefixes
  under different complete-prefill lengths. Preparation is untimed, and all four
  methods receive the same cache. Prepare independent operation inputs and
  randomize protection/recovery call order. Native unprotected prefill retains
  its separate timing stage on the same input schedule.
- Public content remains at most two complete16-token blocks(32 tokens);
  no-sharing uses zero public tokens. Record actual public/private lengths.
  A32-token input can be entirely public: this is a fixed prefix length, not a
  fixed sharing fraction.
- Record actual token counts and KV bytes. Compute seconds/1,000 tokens and
  ms/GiB from each actual length/size, not a fixed512 denominator. Retain raw
  timings, medians, p95 and p99 separately by model/precision/block/length group.
- Use a new request nonce for each operation. Preserve errors and completed
  timings; do not rerun unfavorable values. Resume interrupted formal work by
  observation key without splicing in old100-call records.

## Execution

Unpack into a new directory and reuse the original native environment/cache:

```bash
unzip veilkv-latency-lengths-1k-20261114.zip
cd veilkv-latency-lengths-1k
python3 run.py configure --old-native /path/to/main-gpu-bundle --gpu 0
nohup python3 -u run.py run > runtime/operator.log 2>&1 &
python3 run.py status
```

`--old-native` may instead point to the configured GPU602 bundle; the tool
locates its native directory. For another environment entry, pass
`--python /path/to/native-venv/bin/python --cache-dir /path/to/cache/hub`.
Do not replace the venv entry with its resolved symlink target.

The20 CUDA startup jobs precede formal execution. Wait for other GPU work;
do not compete with the running live integration. Create runtime/pause.flag to
pause; remove it and repeat the same run command to continue. After fixing a
verified environment failure, use `run --retry-blocked` for incomplete work,
retaining every existing record.

```bash
python3 run.py verify
python3 run.py export
```

Return the complete latency-lengths-private-handoff-*.zip for acceptance.
Export requires the actual server source snapshot, issued scientific baseline,
fixed inputs, all timings, environments and errors. For incomplete execution,
use `export --partial`; it does not mark the run complete. Do not manually
select only the results directory.

Preparation performed offline tokenization/input/protocol checks and small CPU
synthetic-cache/real-cryptography interface checks, not GPU timing. At issuance,
the manuscript and anonymous result archive were unchanged. Accepted results are
now documented in docs/LATENCY_LENGTHS_1K.md in the consolidated artifact.
