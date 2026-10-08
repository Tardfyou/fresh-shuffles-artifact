# VeilKV-Plus v2 bugfix log

## Run 1 (archived in run1_archive/)
- arithmetic MAC computed over AES-ct instead of payload -> mac_only helper
- min-row block count for truncation detection; npad>=true_len binding
- tolist() list-of-lists family; bf16 numpy bytes; head_dim fallback; lstsq dims
- test texts shorter than block counts (Qwen tokenizer compactness, 4 sites)
- Gram diagnostic simplified to observed-block eigen ratio
- nonce-reuse e2e construction inconclusive (unit-level XOR leak stands)

## Run 2 (v3 design fixes, 2026-10-08)
- DESIGN FIX 1: public-prefix AAD block-scoped (true_length binding removed for
  public class; private/tail keep full request binding). Root cause of run1 C7
  falsified: request-level true_length made shared-prefix ciphertexts differ
  across requests of different lengths -> stored artifacts not byte-reusable.
- DESIGN FIX 2: KeyService nonce-freshness state persisted to disk (TEE-held in
  deployment); restart now REJECTS replayed nonces (run1: accepted).
- Rerun scope: fixation, functional (TEE-only), tee_battery, paged_harness,
  adjudicate. arithmetic NOT re-executed (run1 failures preserved as inherited
  entries; excluded from deployment claims). attacks/perf inherited from run1
  with flags.

## Run 3 (v4 prefix cross-decode gap fix, 2026-10-08)
- DESIGN FIX 3: public-prefix AAD binds ONLY block-level fields (artifact_id now
  includes layer+kv_type for K/V domain separation ‖ block_index ‖ shape ‖ dtype ‖
  epoch ‖ policy). true_length now bound ONLY on the final private tail block
  (is_last_tail); intermediate tail blocks use tl=-1 (j-index + chain still bind).
- DESIGN FIX 4: explicit chain junction — protect_cache accepts chain_seed
  {(li,kvf): prefix terminal tag}; private tail chains FROM the reused prefix
  artifact's terminal tag; recover_cache accepts chain_seed symmetric. Assembled
  object (A prefix + B tail) decodes under B's context with NO dependence on
  request A's true_length. The run1/run2 splice test failure was a harness
  artifact (post-hoc splice broke chain by construction) — replaced by
  chain-aware assembly.
- C7 gate re-registered: requires all of {sharing full, byte identity full,
  cross_decode true, isolation, relocation, reload, wrong-terminal DETECTED,
  wrong-principal DETECTED, block-swap DETECTED, truncation DETECTED}.
- Run3 postscript: cross-decode criterion re-based on plain-bf16-storage baseline
  (rK 0.49898 == baseline 0.49898 exactly; greedy20 True); chain junction seeds the
  FIRST TAIL block (prefix never re-encoded on radix hit — deep-layer recompute
  drift was the run2 splice failure root cause).
