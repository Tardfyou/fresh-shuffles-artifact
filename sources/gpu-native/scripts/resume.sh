#!/usr/bin/env bash
set -euo pipefail
GPU_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -f "$GPU_BUNDLE_ROOT/runtime/pause.flag"
cd "$GPU_BUNDLE_ROOT"
exec .venv-native/bin/python -m runner.supervisor --phase "${1:-core}" --retry-blocked
