# Large-model CPU extension

Qwen3-14B and Qwen3-32B each reuse the original 3,000 WildChat IDs, all eight
KV heads and two target-permutation seeds. Model forward/cache extraction is
FP32, official protection is BF16, and calibration uses the retained legacy
algorithm. Actual dependencies are Torch 2.14.0+cpu and Transformers 4.51.3.
The measured scope is first-layer content-multiset recovery, with exact
native-full checkpoint cache gates.

| Location | Contents |
|---|---|
| Each model directory | All 3,000 raw records and their summary |
| `protocol/` | Fixed model revisions and measurement protocol |
| `provenance/` | Initial/shard environments, jobs, calibrations and cache gates |
| `source_reference/` | Preserved measurement-source snapshot before sharding |
| `validation/` | Independent acceptance and source-reference comparisons |

`provenance/origins.json` records the disjoint concatenation. The source-reference
comparison documents the launch module's generalized configuration-layer count;
it equals the original source model's layer count in every retained run.

The 14B count is 12,480,000/12,480,000; the 32B count is
12,477,685/12,480,000. All 2,315 misses, involving 1,333 prompt IDs, remain.
Counts pool heads and seeds; there are 3,000 independent input IDs per model.
Bootstrap intervals resample whole prompts within the original length bins.

Inspect from the artifact root without running a model or API:

```bash
python tools/inspect_cpu_large_models.py --root .
```
