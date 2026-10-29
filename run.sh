#!/usr/bin/env bash
set -euo pipefail
sam_projects="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$sam_projects/SAM6D/pipeline/run_runtime.sh" "$@"
