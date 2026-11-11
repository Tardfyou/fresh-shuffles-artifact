# Orbit verification in a public permutation profile

This supplementary experiment calls the unchanged `permute_states` function
released by the authors of Hidden No More. The executable public repository is
https://github.com/leopardracer/vma-external at revision
82ed23ce17d9501880922320f243e4d4e741bd5f. Its README identifies the paper and
retains the earlier ritual-net repository address. This is a second public
permutation configuration, not a new full-system attack.

## Inputs and fixed protocol

The experiment reuses lexical material from the existing WildChat lifecycle100
selection. Two eligible single-token English lexical units per heuristic category
produce10 anchors, taken deterministically from5 source prompts. These generate
120 structured, legal text probes:10 anchors ×4 lengths ×3 repetition patterns.
They are not120 independent user prompts or an extra natural-language recovery
cohort. The default Llama tokenizer adds BOS, and every raw text is checked to
tokenize exactly as declared. BOS is the distinguished row; no KV-Cloak marker is
added to the external implementation.

Public Llama-3.2-1B layer0 embeddings are read directly from its pinned checkpoint.
The code performs no model forward pass, parameter estimation, dictionary matching
or content recovery. The saved input representations permit offline checking.
The current measured device is CPU; a CUDA invocation of this exact operator
experiment is supported by the supplied runner but is not claimed as measured.

The registration predates measurement and fixes the source,10 anchors,
b=4/8/16/32, FP32/bf16, input patterns,1024 S draws and128 D/SD draws per cell.
A recorder retains every row/coordinate permutation without changing random
sampling;32 seeded examples match the untouched function exactly. All720 cells
and307,200 observations are retained.

## Results

For sequence permutation S, the observed state counts agree in every anchor and
datatype. Counts use exact tensor equality, with no distance threshold merging.

| Input row pattern, including BOS | b=4 | b=8 | b=16 | b=32 |
|---|---:|---:|---:|---:|
| One repeated lexical unit |4|8|16|32|
| Two repeated lexical groups |12|273|1021|1024|
| All token rows distinct |24|1009|1024|1024|

Each entry is the number of distinct outputs among1024 sampled row permutations.
At b=4 all24 original permutations actually appear, so the4/12/24 comparison
covers the complete group. Larger groups are sampled, not enumerated:40320,
20922789888000 and263130836933693530167218012160000000 nominal permutations
at b=8/16/32. The exact multiplicity formula is b!/product(n_i!). A repeated
lexical unit plus one BOS row therefore has at most b distinct S outputs, and
all b are observed. The two-group b=8 orbit has280 possible outputs, of which
273 are sampled; no uncovered states are silently treated as impossible.

D and SD refresh the hidden-coordinate permutation at each invocation and
retain128/128 different outputs in every tested cell. Thus the finite b-state
claim applies to the sequence-permutation component/profile; it does not describe
all randomization in the full SD configuration.

## Interpretation

These records verify the input-stabilizer effect in another publicly evaluated
permutation implementation. They provide empirical evidence that nominal group
size can greatly exceed the observable row orbit. They do not establish a new
parameter-identification argument, content recovery, or a break of GELO's default
dense mixing or STIP's hidden-coordinate permutation. GELO's fresh dense mixing
and its existing repetition discussion motivated choosing a more directly
applicable sequence-permutation profile.

## Reproduction

Sources, registered protocol and frozen texts are in `sources/orbit-verification/`.
All observations are in `results/orbit_verification_20261111/`; read every numbered
`checks-*.jsonl.gz` part. The independent auditor reconstructs the output
partitions from the saved permutations and public input rows, rather than trusting
only the summary.

```bash
python sources/orbit-verification/audit_orbit.py \
  --results results/orbit_verification_20261111
```

For a new GPU operator run, use an existing environment with torch, safetensors
and the pinned Llama checkpoint (no full model is loaded):

```bash
python sources/orbit-verification/run_orbit.py \
  --cache /path/to/model_cache/hub --device cuda:0 --out /tmp/orbit-gpu-results
python sources/orbit-verification/audit_orbit.py --results /tmp/orbit-gpu-results
```

The baseline GPU attack/defense suite and all previous CPU data remain unchanged.
The BSD license of the external implementation is retained beside its copied
function; model-input fixture provenance is recorded in the protocol.
