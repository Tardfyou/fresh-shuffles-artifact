# Current602-prompt native GPU results

This accepted run reuses exactly the existing602 ordering IDs; it does not
resample, concatenate old observations, or alter the measurement methods.
All50 formal jobs and10 startup jobs are complete:49,364 and38 observations.
Records, their invocation environments and all normal negative outcomes are
in `results/gpu/accepted/gpu602-native-nonlive-20261114/`. `PAPER_VALUES.json`
and `INPUT_AND_OUTCOME_CHECKS.json` in its summary directory give the recounts.
The issued source is in `sources/gpu602-native/`; the returned partial package
contains no actual server source snapshot. Its supervisor attests unchanged
scientific files, and every returned job/target manifest matches the issuance.
A byte comparison of actual server code is therefore not claimed.

- Fusion:359,936/359,936 tokens across two heads on602 inputs.
- Complete-secret rotation:359,936/359,936 before,1,482/359,936 with stale
  estimates and359,936/359,936 after recalibration.
- Row-secret rotation:6,288/359,936 with stale estimates.
- Six scaling settings:adapted recovery100% in all six; four improve and two
  remain equal. No prompt worsens. Denominators are359,936 for Qwen and
  1,429,120 for Llama.
- Five complete secret configurations per model:all602 inputs recover100%.
- S/a continuation:11,952/36,120 original and11,947/36,120 refreshed segments
  pass;206 pass only originally and201 only after refresh. All12,336 FP32
  mixed-mask-block checks fail; other FP32 checks pass23,544/23,784.
- Native block sizes16/32/64:all3,612 greedy decisions agree per size.
- Large-model byte consumers:602/602 cases pass for each14B/32B model.
- Unprotected engine references:three C rounds give555/602 and539/602 for
  blocks16/32; all six A rounds give602/602. The separately registered simple
  baseline gives558/602 and541/602; it does not reset before each pair.

The selected original leading-score/FP32 diagnostics remain a separate diagnostic
study. Their original two100-input baseline jobs are retained solely to bind
those diagnostics; the new run supplies current whole-cohort reference counts.
Original actual protected-page vLLM results remain current while their602-prompt
rerun is pending. This return does not replace those live measurements.
Main3,000-target recovery,602-target ordering and byte validation, synthetic
privacy, candidate-message pairs, probe-secret trials and timing repetitions
were not rerun here. CPU auxiliary data retain their historical scope.

The author authorized removal of superseded smaller GPU target cohorts. Their
replacement mapping is in `metadata/gpu602/RETIRED_COHORTS.json`; no retained
3,000/602 result or current live-engine measurement was removed.

```bash
python tools/inspect_gpu602_native.py --root .
python tools/check_artifact_catalog.py --root .
```
