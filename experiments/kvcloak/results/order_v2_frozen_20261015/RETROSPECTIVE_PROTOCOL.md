# Ordering experiment: retrospective protocol record

**Status: RETROSPECTIVE, not independently archived before execution.**
This record was assembled after the experiments on October 15, 2026, using
the October 8 task instructions, CONFIG constants in the code, and code review
records. It was not independently preregistered before execution. This process
limitation is retained explicitly. Each evidence reference below identifies a
record or implementation that already existed before execution.

## 1. Fixed experimental conditions

Evidence: CONFIG in `exp_order_v2.py`, implemented before execution.

- Model: Qwen2.5-0.5B; b=16; beam_width=16; top-1 output; context limit96; two iterations.
- Seeds: calibration42; obfuscation9000 plus sample index; sampling dev=20260929,
  main=dev+1, boundary=dev+2. Main and boundary use independent seeds and do not
  overlap dev. This was corrected on October 7, 2026, and the first samples were
  checked to differ.

## 2. Sampling design

Evidence: `samples_main/boundary.jsonl`, with source text and groups per sample.

- Main: three categories x four length groups(b1/b2/b4/b8) x50 samples;
  `generate_sample_of_length` enforces the lengths.
- Boundary A:100 pairs with the same token multiset but different directions.
  Both directions use the same wrapper context and identical token counts;
  only the body positions differ. The October 13 check records36 tokens/two
  blocks, equal Counters and two position differences.
- Boundary B:100 random token sequences.
- Dev:60 samples(3x4x5), already inspected and excluded from formal results.

## 3. Controls and metrics

Evidence: `run_experiment/evaluate` in `exp_order_v2.py`, implemented before execution.

- Conditions: end2end, random_order, oracle_ms and no_cross_block.
- Metrics: position_accuracy(primary), exact_match, block_exact,
  trigram_overlap, multiset_conservation, pa_by_block and random_baseline_pa
  (sum(c^2)/b^2).
- Ground truth must not select a key, candidate or seed. Main results report top-1 only.

## 4. Known data limitations

- Per-token predictions and the original bag Counters were not saved.
  Regeneration would require a deterministic rerun with the fixed seeds,
  approximately35 hours for the complete study; it was not performed.
- multiset_conservation compares predictions with the recovered input bags.
  Per-sample equality between recovered and true bags was checked only in dev
  (60/60 oracle=end2end); main retains aggregate equality only.
- After resumed segments, random_order uses the resumed RNG state, as recorded
  during the bug-fix process.

## 5. Execution incidents and handling

Evidence: `resume_info_*.json`. No independent pre-execution version record
binds the code; the retained source files are authoritative.

- October 7: a full-disk incident lost the complete output; execution was
  repeated with incremental recording and outputs moved to another disk.
- Three processes wrote concurrently. Identical seeds produced identical
  content; sample-ID deduplication retains600 unique rows and zero malformed rows.
- Initial pair samples were all skipped because they had fewer than16 tokens.
  A shared wrapper corrected the inputs, followed by `--resume` for those samples.
