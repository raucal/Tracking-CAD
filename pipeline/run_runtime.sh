#!/usr/bin/env bash
# Compatibility entry; all dispatch and paths live in launcher.py/settings.py.
set -euo pipefail
cad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$cad_root/run.sh" "$@"
