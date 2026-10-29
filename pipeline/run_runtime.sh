#!/usr/bin/env bash
# Isolated environment for SAM-6D, ASAP, and Windows exports.
set -euo pipefail
pddl_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
pddl_python="$pddl_root/.venvs/assembly-runtime/bin/python"
if [[ ! -x "$pddl_python" ]]; then
    echo "Falta el entorno de ejecución: $pddl_python" >&2
    exit 2
fi
pddl_base="$("$pddl_python" -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$pddl_base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$pddl_root/.venvs/assembly-runtime/bin:$PATH"
export MPLCONFIGDIR="$pddl_root/.runtime/matplotlib"
export XDG_CACHE_HOME="$pddl_root/.runtime/cache"
export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1
export FOUNDATIONPOSE_PYTHON="${FOUNDATIONPOSE_PYTHON:-$pddl_root/.venvs/foundationpose/bin/python}"
export FOUNDATIONPOSE_ROOT="${FOUNDATIONPOSE_ROOT:-$pddl_root/FoundationPose}"
export FOUNDATIONPOSE_NATIVE="${FOUNDATIONPOSE_NATIVE:-$pddl_root/SAM6D/pipeline/native/foundationpose}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$pddl_root/.runtime/cache/torch_extensions}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
pddl_mode="${1:-}"
if [[ -z "$pddl_mode" ]]; then
    echo "Uso: bash SAM6D/pipeline/run_runtime.sh {install|ui|detect|sam|suite|scores|asap|catalog|export} [argumentos]"
    exit 2
fi
shift
case "$pddl_mode" in
    install)
        exec bash "$pddl_root/SAM6D/pipeline/install_models.sh" "$@"
        ;;
    sam)
        if [[ "${1:-}" == render ]]; then
            exec "$pddl_python" "$pddl_root/SAM6D/SAM-6D/pddl_pipeline.py" "$@" \
                --blender-install-path "$pddl_root/.runtime/blender" --render-size 256 --render-samples 16
        fi
        exec "$pddl_python" "$pddl_root/SAM6D/SAM-6D/pddl_pipeline.py" "$@"
        ;;
    asap)
        exec "$pddl_python" "$pddl_root/ASAP/pddl_pipeline.py" "$@"
        ;;
    suite)
        exec "$pddl_python" "$pddl_root/SAM6D/pipeline/sam6d_suite.py" "$@"
        ;;
    catalog)
        exec "$pddl_python" "$pddl_root/ASAP/pipeline/catalog_generator.py" "$@"
        ;;
    scores)
        exec "$pddl_python" "$pddl_root/SAM6D/pipeline/score_all_cads.py" "$@"
        ;;
    detect)
        exec "$pddl_python" "$pddl_root/SAM6D/pipeline/run_detection.py" "$@"
        ;;
    ui)
        exec "$pddl_python" "$pddl_root/SAM6D/pipeline/sam6d_gui.py" "$@"
        ;;
    export)
        exec "$pddl_python" "$pddl_root/ASAP/pipeline/export_results.py" "$@"
        ;;
    *)
        echo "Modo no reconocido: $pddl_mode" >&2
        exit 2
        ;;
esac
