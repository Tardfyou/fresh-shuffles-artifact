#!/usr/bin/env bash
set -euo pipefail
VLLM_PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PACK_ENV="${VEILKV_VLLM_PYTHON:-$VLLM_PACK_ROOT/.venv-vllm/bin/python}"
if [ ! -x "$VLLM_PACK_ENV" ]; then
  if command -v conda >/dev/null 2>&1; then
    conda create --prefix "$VLLM_PACK_ROOT/.venv-vllm" python=3.10 -y
  else
    python3 -m venv "$VLLM_PACK_ROOT/.venv-vllm"
  fi
fi
"$VLLM_PACK_ENV" -m pip install -r "$VLLM_PACK_ROOT/requirements.txt"
cd "$VLLM_PACK_ROOT"
"$VLLM_PACK_ENV" -m unittest discover -s tests -v
"$VLLM_PACK_ENV" -m pip freeze > runtime/installed-packages.txt
printf '%s\n' 'Environment ready. Run the registered CUDA smoke queue when the GPU is free.'
