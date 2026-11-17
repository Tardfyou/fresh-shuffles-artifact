# Orbit verification in a public permutation implementation

Source: the Hidden No More authors' public repository leopardracer/vma-external,
pinned in protocol/UPSTREAM.json. Call only the original permute_states
function, not the vocabulary-matching attack. GELO's default dense mixing is
not a pure permutation; this study does not claim to break it.

Inputs reuse10 single-token English lexical anchors(two per category) from the
existing lifecycle100 cohort. Construct legal repeated text and retain Llama's
default BOS. This is not a new100-input natural-language recovery cohort.
Test b=4/8/16/32, three repetition patterns and FP32/bf16. Protected inputs are
Llama-3.2-1B layer0 embedding lookups; no full-model forward or post-attention
representation is evaluated. In profile S, row multiplicities predict an orbit
of b!/product(n_i!). BOS plus b-1 identical rows therefore predicts b states;
no KV-Cloak marker is inserted.

D and SD refresh coordinate permutations at each invocation, so their complete
outputs can remain diverse. Do not extend the b-state S result to the complete
SD randomization. Observing collapse does not establish another parameter
identification or content recovery attack.

## Execute with the frozen inputs

Reuse the main GPU environment and cached Llama1B checkpoint. Only a small
embedding subset and permutation tensors are needed; do not load the complete
model, start a service or call an API. Use a new output directory and preserve
existing records.

```bash
python run_orbit.py --cache /path/to/model_cache/hub --device cuda:0 --out results/gpu_operator
python audit_orbit.py --results results/gpu_operator
```

The same script with `--device cpu` performs CPU operator checks, not complete
CPU inference. `--prepare` builds inputs during preparation; formal collection
does not resample them.

The protocol and fixtures were frozen before measurement. Retain all registered
cells and ordinary negative outcomes. Record each input/precision/profile
observation once. Save every trial seed, complete permutation, output-state ID
and exact partition comparison. Public embedding rows are saved separately;
the auditor reconstructs every output partition rather than trusting summaries.
The recording wrapper only records randperm outputs. Before formal sampling,
32 fixed examples must match the unchanged function without affecting RNG state
or outputs.

This study checks input-stabilizer effects in another public profile. It does
not change the paper's formal analysis or primary experiments. Report the
conditions, sources and all positive and negative results together.
