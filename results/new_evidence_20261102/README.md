# Reviewed follow-up observations

This directory contains saved probe inputs/states/diagnostics, fixed-candidate
identification scores, numerical-path predictions, six scaling-point records,
and rounded score-gap histograms. It adds data and offline analysis only.
The fixed prompt IDs belong to the existing spectrum and need-ratio cohorts.

The a/S predicate uses cosine >= 0.999 and bijective column matching. It does
not include M/u/w. M errors were measured in supplementary reconstructed sessions,
not recovered from historical logs. Content means aggregate heads within a
session. The numerical-path v2 files add full predictions for the same selected
306 blocks. Predictions retain within-block order as emitted; correctness uses
the token multiset. Two seeds reuse the same targets. A positive aggregate
scaling gain can coexist with individual prompt regressions; every record remains.

From the artifact root:

    python tools/inspect_new_evidence_records.py --root .

For the earlier error-rate presentation with matplotlib and installed Arial:

    python tools/plot_scaling_observations.py --data results/new_evidence_20261102/summary.json --out /tmp/scaling-panels

The histogram unit is one distinct gold-token type per failing block, classified
by presence in the predicted token set. These are not multiplicity-weighted miss
counts. Distribution statistics and finite numerical-path stability are reported
as observations; physical error causes and general scaling guarantees are unproven.

The current paper uses the paired-gain/outcome view from tools/plot_paper_gallery.py and results/visual_summary_20261102/data.json. Both views use the same six recorded settings.
