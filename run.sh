#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Python 3 is needed only to read configuration; inference uses the selected runtime.
exec "${CAD_TRACKING_BOOTSTRAP_PYTHON:-python3}" "$cad_root/pipeline/launcher.py" "$@"
