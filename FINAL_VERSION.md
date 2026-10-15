# Current manuscript and evidence

Updated 2026-10-15. The supplied paper/build/main.pdf has 18 pages: 13 main-text pages and 5 pages of references/appendices.

The main raw result remains 263,029/263,904 correlated head-token opportunities across ten models and twelve configurations. The ordering follow-up adds 600 synthetic main metric IDs and 300 boundary metric IDs, without changing that denominator. Mean main position accuracy is 22.3% versus 7.4% randomized ordering and 18.9% without cross-block context; logs record 18/600 exact spans.

The nominal generation minima are not exact-length buckets: 109/600 spans overshoot and 581 main spans are distinct. Raw token predictions and recovered counters were not retained. Oracle conservation supplies indirect whole-span content agreement under the inspected source; it does not certify per-block correctness. Direction-pair scores are position accuracy, and none of the 100 pairs holds every block bag fixed. No zero factual-leakage claim is made. The protocol and source snapshots are explicitly retrospective.

Evidence, all physical result rows, the offline-tokenizer audit and source snapshots are in paper/evidence/order_v2/. The acceptance report is paper/review/ORDER_V2_ACCEPTANCE_20261015.md. Output-directory usernames were anonymized in the source snapshots; the source filenames and normalization notes are recorded.

Figures and tables retain the measured F258 aesthetic; the ordering comparison is Figure 6 and its nominal-group table is Table IV. The Figure 5 performance baseline is unfused KV-Cloak, not the original paper's weight-fused serving path. No new model runs were performed during manuscript integration.
