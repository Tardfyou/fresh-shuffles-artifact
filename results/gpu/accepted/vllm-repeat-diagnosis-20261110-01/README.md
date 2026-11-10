# Repeated-generation diagnostics

These observations reuse the original100 inputs; no original baseline records
are replaced. A disables prefix caching; C enables it and resets before each
pair. Three full-cohort runs per block size reproduce all original C output
transcripts; all A pairs agree. E is FP32 on the10 distinct diagnostic IDs at
both block sizes. F also requests prompt log-probabilities, a different execution
condition. D compares two calls after an unmeasured warm-up.

Important source/label correction: stage1_B shares the exact C branch and
resets only once before each pair. It is a repeated C observation, not the
registered cold-cold intervention. Its records are preserved under their
original labels; claims excluding cold/warm numerical differences are withdrawn.
At15 first divergences, competing winners belong to the top-two score groups;
one event has a tied second-score group extending to a third list entry. Thus
strict top-two-list swaps occur14/15, and tie-aware leading-group switches15/15.
Recorded top-two gaps round to0,0.0625 or0.125. These are log-probability
observations, not full-logit traces or an isolated kernel cause.

The protocol records intended tests and initial hypotheses, not accepted
conclusions. No general determinism guarantee,7–8% correctness allowance, or
quantitative concurrency/failure-probability claim is established. Live VeilKV
continues to require exact paired bytes/logits and matching outputs.

Recount from the artifact root:

```bash
python tools/audit_vllm_repeat.py \
  --diagnosis results/gpu/accepted/vllm-repeat-diagnosis-20261110-01 \
  --original results/gpu/accepted/a100-80g-20261110-01 \
  --output /tmp/vllm-repeat-audit.json
```
