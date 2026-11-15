# Third-party sources and data

KV-Cloak source is retained at commit 6b40f36edb2f337557543e7e60b10022308883d4 from https://github.com/SiO-2/kvcloak. Its original license is included at vendor/kvcloak/LICENSE.

The fixed 3,000-prompt manifest comes from allenai/WildChat-4.8M, using the existing recorded 2025-traffic selection, toxic/redacted filters and seed20261021. Source-selection records and the prior accepted cohort are retained in reference/accepted_cpu_artifact.zip. WildChat-4.8M is released under ODC-By; attribution and dataset terms continue to apply. The manifest does not include assistant responses, IP-derived fields or request headers.

MMLU and SQuAD are used only for the original work's accuracy controls. Dataset repositories, immutable revisions, complete test records and fixed examples are declared in protocol/utility_datasets.json and data/utility/. Source URLs: https://huggingface.co/datasets/cais/mmlu and https://huggingface.co/datasets/rajpurkar/squad. Their upstream data terms apply.

Models are not redistributed. runner.download_models downloads pinned public checkpoints under each upstream model's own terms. Any access requirements must be satisfied by the operator.

The historical CPU artifact is retained read-only for audit. Active GPU entry points are runner/, with their reviewed scientific dependencies in science_base/; do not run historical entry points from the reference archive.
