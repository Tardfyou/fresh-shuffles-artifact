# Sources and data

vLLM reference: https://github.com/vllm-project/vllm, tag v0.8.5.post1,
revision 3015d5634e74d59704e2b39bab0dbe2e6f86a38a. Apache2.0 license retained
in source_reference/LICENSE. Reference files validate installed upstream seams.

KV-Cloak: https://github.com/SiO-2/kvcloak, revision
6b40f36edb2f337557543e7e60b10022308883d4; license in vendor/kvcloak/LICENSE.

Inputs reuse the existing lifecycle100 IDs from the 3,000-prompt
allenai/WildChat-4.8M manifest. Dataset source and ODC-By terms:
https://huggingface.co/datasets/allenai/WildChat-4.8M.
The fixed32-token public prefix also comes from that manifest; its construction
and source ID are in protocol/INPUTS.json. No new user dataset is collected.

Weights are not redistributed; pinned checkpoints retain upstream terms.
No paper, slides, credentials, server addresses, model weights, local databases
or runtime logs are included in this execution ZIP.
