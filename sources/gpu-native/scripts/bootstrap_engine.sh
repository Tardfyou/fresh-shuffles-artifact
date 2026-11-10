#!/usr/bin/env bash
set -euo pipefail
GPU_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ENGINE_PREFIX="$GPU_BUNDLE_ROOT/.venv-vllm"
if [ ! -x "$GPU_ENGINE_PREFIX/bin/python" ]; then
  if command -v conda >/dev/null 2>&1; then
    conda create --prefix "$GPU_ENGINE_PREFIX" python=3.10 -y
  else
    python3 -m venv "$GPU_ENGINE_PREFIX"
  fi
fi
"$GPU_ENGINE_PREFIX/bin/python" -m pip install --upgrade pip
"$GPU_ENGINE_PREFIX/bin/python" -m pip install -r "$GPU_BUNDLE_ROOT/requirements-engine.txt"
"$GPU_ENGINE_PREFIX/bin/python" -m pip freeze > "$GPU_BUNDLE_ROOT/runtime/engine-installed-packages.txt"
