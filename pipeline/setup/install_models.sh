#!/usr/bin/env bash
# Restore the supported model weights and compile the PEM CUDA extension if needed.
set -euo pipefail
cad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
sam_python="${CAD_TRACKING_PYTHON}"
sam_base="$("$sam_python" -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$sam_base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${CAD_TRACKING_PYTHON%/python}:$PATH"
sam_cache="$SAM6D_CHECKPOINT_CACHE"
"$sam_python" -m pip install --no-cache-dir --constraint "$cad_root/pipeline/setup/constraints-assembly-runtime.txt" -r "$cad_root/pipeline/setup/requirements-sam6d.txt"
"$sam_python" "$cad_root/pipeline/tools/download_runtime_assets.py" --cache "$sam_cache"
"$sam_python" "$cad_root/pipeline/tools/download_pose_assets.py" --cache "$sam_cache"
if ! compgen -G "${CAD_TRACKING_NATIVE}/pointnet2/_ext*.so" > /dev/null; then
    if [[ -z "${CUDA_HOME:-}" ]]; then
        cuda_compiler="$(command -v nvcc || true)"
        [[ -n "$cuda_compiler" ]] || { echo 'Set CUDA_HOME or add nvcc to PATH for compilation.' >&2; exit 2; }
        export CUDA_HOME="$(dirname -- "$(dirname -- "$cuda_compiler")")"
    fi
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$sam_python" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
    export MAX_JOBS="${MAX_JOBS:-2}"
    cd "${CAD_TRACKING_SAM6D}/SAM-6D/Pose_Estimation_Model/model/pointnet2"
    "$sam_python" setup.py build_ext --build-temp "$cad_root/.cache/pointnet2-build" --build-lib "${CAD_TRACKING_NATIVE}"
fi
exec bash "$cad_root/run.sh" sam doctor
