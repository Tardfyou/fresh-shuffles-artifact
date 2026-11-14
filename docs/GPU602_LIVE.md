# Accepted602-prompt live vLLM integration

`gpu602-vllm-live-20261114` replaces the original100-target live cohort.
The same frozen602 IDs are tested afresh, without splicing old measurements.
Four cells cross Qwen2.5-0.5B/Llama-3.2-1B with physical blocks16/32.
Each cell contains602 ordinary requests,301 shared-prefix pairs,602 isolation
pairs and4816 tamper cases. A separate Qwen/b16 attack covers602 targets.
Formal cases total25886;50 startup cases remain separate.

All6020 functional cases preserve recorded cache roundtrips, logits and
six-token outputs exactly. All19264 tamper cases reject before model execution.
Shared physical pages, disjoint private pages, identical public ciphertext and
different private ciphertext are checked from the retained page tables and
payloads. The first10 ordinary cases in each cell rotate epochs after two
engine steps:40 total, retaining the original trigger rule.

The attack recovers359936/359936 token occurrences across two heads. The
separate defender-held KV-Cloak ordered-restore adapter preserves six-token
outputs in562/602 cases and has nonexact logits. All40 mismatched outputs remain;
this adapter result is distinct from exact token-multiset recovery and from
VeilKV functional validation.

The return contains records, environments, the unchanged outer plan and a
supervisor no-scientific-change statement, but no actual server source/baseline
snapshot. Its smoke declares a direct43-file lock pass. The supplied note says
CPU threads changed8 to32; the actual runtime-settings file and thread value
were not recorded in the environments. Acceptance uses independent complete
case/input/score/consumer/page checks and that source statement; actual server
source byte comparison is not claimed. `sources/vllm-live/` retains the issued
602 source and inputs with the declared32-thread resource setting.

Original live measurements and their old archive were removed under the
author's cleanup instruction once this complete replacement was accepted.
Selected original unprotected-engine numerical diagnostics remain separate,
alongside the newer602-target reference results. The pending multi-length
1k performance supplement is not part of this return.

```bash
python tools/inspect_gpu_live602.py --root .
python tools/inspect_gpu_results.py --root .
python tools/check_artifact_catalog.py --root .
```
