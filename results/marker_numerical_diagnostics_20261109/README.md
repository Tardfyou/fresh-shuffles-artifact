# Marker numerical diagnostics

These data reuse the same fixed WildChat manifest. The layout audit reads all
saved Qwen/Llama failing blocks and aligns labels using saved correct blocks.
Layout labels are newly reconstructed from the original CPU implementation
and seeds, rather than inserted into historical logs.

Known-key arithmetic checks use 446 head-block observations: 66 Qwen blocks
from seven prompt IDs and 380 Llama blocks from 252 prompt IDs. They contain
266 previously failed blocks and 180 previously correct blocks. The earlier
306 numerical-path blocks are retained. The extension adds one saved failure
from each of the first 20 distinct prompt IDs per failing Llama head, with
observation keys deduplicated. Thus prompts and heads recur; these are targeted
diagnostics, not a population estimate of token recovery.

The fixture and protocol were frozen before collection. The final records add
an official-forward equivalence check to the same arithmetic conditions. All
446 BF16 forward outputs match the official implementation bitwise. Earlier
collection/trace stages remain in the private raw archive; their common fields
agree exactly. All six numerical conditions and every selected block remain.

Only known public token arrays and directly generated transform parameters are
used. There is no calibration, secret recovery, marker-hypothesis search, or
open-vocabulary decoding. Candidate sets contain only token IDs already present
in the saved truth or baseline prediction. The unused Key thresholds are zero;
Value thresholds follow the original eight calibration sentences. This leaves
the first-layer Value parameters and random-draw order intact.

The recorded original marker-row vector, transformed vector, stored marker and
rounded sum allow coordinate absorption to be checked directly. Once a nonzero
component is rounded away during marker insertion, subtracting the marker does
not restore it. Row mixing adds further rounding error. The measured precision
controls establish this numerical error source. Parameter-estimation effects
are not individually decomposed by these known-key conditions.

Offline saved-record verification:

    python tools/inspect_marker_numerics.py --root .

To regenerate the layout audit with the public model tokenizers already cached:

    python tools/audit_known_marker_locations.py --root . --cache /path/to/hf/hub --out /tmp/marker-layout.json

To repeat the known-key arithmetic conditions, first copy PROTOCOL.json and
fixtures.jsonl to a new empty directory, then run:

    python tools/known_key_marker_fidelity.py --project . --cache /path/to/hf/hub --out /tmp/new-marker-check

The runner refuses to overwrite checks.jsonl. It uses public checkpoint tensors
and the retained official implementation. Environment revisions and actual
observation accounting are supplied in this directory. The main-table counts
remain unchanged.
