# English documentation and preserved original data

Release documentation, operation guides, code comments and docstrings are in
English, including copies inside scientific baseline and CPU reference ZIPs.
Historical filenames such as README_ZH.md remain for compatibility with frozen
file lists; their contents are now English.

Original experimental text is not translated. Fixed benchmark inputs, selected
WildChat messages and saved model outputs may contain source-language text,
including mixed-language requests. Their text, token arrays, IDs, predictions,
scores and negative outcomes are preserved. A source-language dataset annotation
does not imply that every character in a selected message is English.

Localization takes place in publication copies after measurement. For Python,
the executable AST and fixed string values are checked against the accepted
version, excluding translated docstrings. Baseline ZIPs receive the same comment
and documentation translations as their corresponding source copies, so direct
source-lock checks continue to work. Runtime records and their original version
labels are not rewritten to claim that translated comments were present during
the server runs. Upstream licenses and revision identifiers remain intact.

Inspect the language boundary without running a model or calling an API:

```bash
python -B tools/check_english_materials.py --root . --report /tmp/english-materials.json
python -B tools/check_artifact_catalog.py --root .
```

The language checker reports retained original-data locations separately from
documentation or annotation failures. It does not remove, translate, resample
or re-score scientific records.
