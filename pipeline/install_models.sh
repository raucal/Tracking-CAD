#!/usr/bin/env bash
# Restore the supported model weights and compile the PEM CUDA extension if needed.
set -euo pipefail
sam_projects="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
sam_python="$sam_projects/.venvs/assembly-runtime/bin/python"
sam_base="$("$sam_python" -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$sam_base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$sam_projects/.venvs/assembly-runtime/bin:$PATH"
sam_cache="${SAM6D_CHECKPOINT_CACHE:-/dev/shm/pddl-assembly-runtime/checkpoints}"
"$sam_python" -m pip install --no-cache-dir -r "$sam_projects/SAM6D/pipeline/requirements-pose.txt"
"$sam_python" "$sam_projects/SAM6D/pipeline/download_runtime_assets.py" --cache "$sam_cache"
"$sam_python" "$sam_projects/SAM6D/pipeline/download_pose_assets.py" --cache "$sam_cache"
if ! compgen -G "$sam_projects/SAM6D/pipeline/native/pointnet2/_ext*.so" > /dev/null; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
    export MAX_JOBS="${MAX_JOBS:-2}"
    cd "$sam_projects/SAM6D/SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2"
    "$sam_python" setup.py build_ext --build-temp /dev/shm/pddl-assembly-runtime/pointnet2-build --build-lib "$sam_projects/SAM6D/pipeline/native"
fi
exec bash "$sam_projects/SAM6D/run.sh" sam doctor
