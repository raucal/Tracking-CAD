#!/usr/bin/env bash
# Install FoundationPose without changing the SAM-6D environment or model checkout.
set -euo pipefail
cad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
shared_python="${CAD_TRACKING_PYTHON}"
target="${FOUNDATIONPOSE_PYTHON%/bin/python}"
export LD_LIBRARY_PATH="$("$shared_python" -c 'import sys; print(sys.base_prefix)')/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x "$target/bin/python" ]]; then
    "$shared_python" -m venv --system-site-packages "$target"
fi
shared_packages="$("$shared_python" -c 'import site; print(site.getsitepackages()[0])')"
"$target/bin/python" - "$shared_packages" <<'PY'
from pathlib import Path
import site, sys
Path(site.getsitepackages()[0], 'assembly_runtime.pth').write_text(sys.argv[1] + '\n')
PY
"$target/bin/python" -m pip install --no-cache-dir --constraint "$cad_root/pipeline/setup/constraints-foundationpose.txt" -r "$cad_root/pipeline/setup/requirements-foundationpose.txt"
# FoundationPose uses PyTorch3D transforms and nvdiffrast rendering.
# Its CUDA rasterizers are not required: the CPU extension avoids duplicate kernels.
PYTORCH3D_FORCE_NO_CUDA=1 MAX_JOBS=2 CFLAGS=-g0 CXXFLAGS=-g0 \
    "$target/bin/python" -m pip install --no-cache-dir --constraint "$cad_root/pipeline/setup/constraints-foundationpose.txt" --no-build-isolation --no-deps \
    git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9
"$target/bin/python" -m pip install --no-cache-dir --constraint "$cad_root/pipeline/setup/constraints-foundationpose.txt" --no-build-isolation --no-deps \
    git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8
native="${CAD_TRACKING_NATIVE}/foundationpose"
cmake -S "${FOUNDATIONPOSE_ROOT}/mycpp" -B "$native/build-cadtracking" \
    -DPYTHON_EXECUTABLE="$target/bin/python" \
    -Dpybind11_DIR="$("$target/bin/python" -m pybind11 --cmakedir)" \
    '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG -g0'
cmake --build "$native/build-cadtracking" --parallel 2
cp "$native"/build-cadtracking/mycpp*.so "$native/"
echo "FoundationPose installed in $target. Weights are loaded from paths.foundationpose_weights in config/settings.json."
