#!/usr/bin/env bash
set -euo pipefail
GPU_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_NATIVE_PREFIX="$GPU_BUNDLE_ROOT/.venv-native"
if [ ! -x "$GPU_NATIVE_PREFIX/bin/python" ]; then
  if command -v conda >/dev/null 2>&1; then
    conda create --prefix "$GPU_NATIVE_PREFIX" python=3.10 -y
  else
    python3 -m venv "$GPU_NATIVE_PREFIX"
  fi
fi
"$GPU_NATIVE_PREFIX/bin/python" -m pip install --upgrade pip
"$GPU_NATIVE_PREFIX/bin/python" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
"$GPU_NATIVE_PREFIX/bin/python" -m pip install -r "$GPU_BUNDLE_ROOT/requirements-native.txt"
cd "$GPU_BUNDLE_ROOT"
"$GPU_NATIVE_PREFIX/bin/python" -m unittest discover -s tests -v
"$GPU_NATIVE_PREFIX/bin/python" -m pip freeze > runtime/native-installed-packages.txt
if [ "${GPU_RUN_SMOKE:-0}" = 1 ]; then
  "$GPU_NATIVE_PREFIX/bin/python" -m runner.doctor --gpu 0
else
  printf '%s\n' 'Environment ready. When the GPU is free, run .venv-native/bin/python -m runner.doctor --gpu 0'
fi
