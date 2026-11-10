#!/usr/bin/env bash
set -euo pipefail
GPU_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
touch "$GPU_BUNDLE_ROOT/runtime/pause.flag"
printf '%s\n' 'Pause requested. Workers retain the current complete observation, then release GPU memory.'
