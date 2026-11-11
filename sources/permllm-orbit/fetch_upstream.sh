#!/usr/bin/env bash
set -euo pipefail
permllm_checkout="${1:-/tmp/permllm-orbit-upstream}"
if [ ! -d "$permllm_checkout/.git" ]; then
  git clone https://github.com/zfscgy/PermLLM.git "$permllm_checkout"
fi
git -C "$permllm_checkout" checkout 1b120aeb65f18744a1cd73dd419dc15f39a73971
