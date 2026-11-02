# Figure inputs

These plot inputs are direct selections and aggregates of accepted saved records.
All 602 semantic scores per condition, all 100 privacy cases, all 800 timing
observations and all six scaling settings remain. The original experiment files
are retained separately. Figures do not create new experimental observations.

From the artifact root, with matplotlib/numpy/scipy and Arial installed:

    python tools/plot_paper_gallery.py --data results/visual_summary_20261102/data.json --out /tmp/paper-panels

The author-selected #00B887 / #FF5B61 endpoints apply to tables and the key
matrix. Other panels reproduce the specific palettes and forms of USENIX
Security Approximate Caches Figures 11/22 and NDSS PROMPTPEEK Figures 10/11/13.
A colour encodes a magnitude or named series, not statistical significance.
