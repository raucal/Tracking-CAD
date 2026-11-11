# Installation

[README](../README.md) · [Reference](REFERENCE.md) · [Provenance](PROVENANCE.md)

Run the commands below from the project root unless stated otherwise. This guide covers environments and data; normal application use is described in the README.

## Existing server

The configured server already has environments, weights, and native extensions:

```bash
bash run.sh settings
bash run.sh sam doctor
bash run.sh ui
```

The installation uses `../.venvs/assembly-runtime` for SAM6D/PEM and `../.venvs/foundationpose` for FoundationPose. The former depends on Conda's `dinobot_env` Python; the latter shares libraries with the base runtime. Preserve these environments and their dependencies. Copying a venv to another machine is insufficient: executables and `.pth` files can retain paths to the original installation.

## Requirements for another machine

The recorded environment is Linux, Python 3.10, PyTorch 2.5.1 with CUDA 12.4, torchvision 0.20.1, and NumPy 1.26.4. Native extensions were compiled with CUDA Toolkit 12.5. Package constraints are in `pipeline/setup/`.

You need:

- The full CADTracking repository and its model submodules. The official model repositories alone do not include this interface or its adapters.
- An NVIDIA GPU and compatible driver with enough memory for the selected models. PEM and FoundationPose require CUDA.
- A C/C++ compiler, CMake, OpenMP, Eigen3, Boost system/program_options, and CUDA Toolkit with `nvcc` to build extensions.
- Blender 3.3.1 as an executable for template rendering; the `bpy` Python package is not a replacement.
- RealSense SDK and USB permissions when using a camera. Saved-file analysis does not require one.
- Space for environments, builds, weights, prepared data, and results. SAM/DINOv2/PEM weights in the recorded installation occupy approximately 4.9 GiB, in addition to FoundationPose.

The following recipe documents setup on another machine. A fresh installation was not performed as part of this documentation update; validate driver/compiler/extension compatibility using the checks below.

## Dependencies

Dependencies are split between system software, Python environments, and external assets. The recorded host uses Ubuntu 22.04.5 LTS. Other Linux systems need equivalent packages and compatible GPU build tools.

| Layer | Required components | How they are provided |
| --- | --- | --- |
| Bootstrap | Bash, Git (including submodule access), Python 3, Conda for the recipe below, and pip/venv | Install on the destination machine before using the launcher. Git is also used to apply the validated SDU compatibility patch. |
| GPU/build | NVIDIA driver, CUDA Toolkit with `nvcc`, C/C++ toolchain, CMake >=3.15, OpenMP, Eigen3, Boost system/program_options | System packages/toolkit; Python requirements do not install them. On Ubuntu, build packages include `build-essential`, `cmake`, `libeigen3-dev`, `libboost-system-dev`, and `libboost-program-options-dev`; verify OpenMP support in the selected compiler. |
| Base Python runtime | PyTorch/torchvision CUDA wheels, NumPy, image/mesh processing, Hydra, SAM6D/PEM, camera and report dependencies | [requirements-sam6d.txt](../pipeline/setup/requirements-sam6d.txt), constrained by [constraints-assembly-runtime.txt](../pipeline/setup/constraints-assembly-runtime.txt). Install with `bash run.sh install` after creating the base runtime. |
| FoundationPose Python runtime | Shared base libraries, Open3D CPU, Warp, Kornia, transformations, pybind11, and supporting packages | [requirements-foundationpose.txt](../pipeline/setup/requirements-foundationpose.txt), constrained by [constraints-foundationpose.txt](../pipeline/setup/constraints-foundationpose.txt). The installer additionally builds PyTorch3D, nvdiffrast, and `mycpp`. |
| CAD template renderer | Blender 3.3.1 executable, including its own `bpy` and `mathutils` | Separate Blender installation selected by `paths.blender`; not installed into the application venv. |
| Camera, when used | RealSense SDK, device access/USB rules, and `pyrealsense2` | SDK/system configuration plus the base Python requirements. A physical camera is unnecessary for saved RGB-D analysis. |
| Optional tools | `manifold3d` for `--repair-topology`; `rtree` for full mesh-proximity audits | [requirements-optional.txt](../pipeline/setup/requirements-optional.txt); not required for routine detection/tracking. These packages have no tested version pins in the recorded environment. |
| Assets | Pinned model sources, checkpoints, prepared data, source CADs, and assembly inventories | Submodules and separately supplied/downloaded assets; see below. These are not pip packages. |

To enable the optional tools in an already configured base runtime:

```bash
../.venvs/assembly-runtime/bin/python -m pip install --constraint pipeline/setup/constraints-assembly-runtime.txt -r pipeline/setup/requirements-optional.txt
```

The constraints are snapshots of an existing environment, including packages used by other local projects. They restrict versions when requested; they are not a request to install every listed package. Use the feature-specific requirements and installers above. Neither the web interface nor saved-demo inference requires a robot or its controller packages. Preserved SDU robot demos have additional hardware/software requirements described in their [integration README](../integrations/foundationpose_sdu/README.md).

## CPU-only development checks

After cloning with submodules, use a separate test environment if you only need to work on the application and run simulated tests:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=pipeline .venv/bin/python -B -m pytest -p no:cacheprovider pipeline/tests alignment_pose_cad/tests -q
```

These tests need neither prepared data nor weights, CUDA, Blender, RealSense or robot hardware. Shared fixtures in the root `conftest.py` provide synthetic CAD/RGB-D and an explicit temporary HTTP catalog. One topology-repair test is optional when `manifold3d` is absent. CPU tests do not validate native GPU builds or physical hardware. The GitLab CI job follows the same dependency split; its actual remote run requires the referenced submodule commit to be published first.

## Clone the project

```bash
git clone --recurse-submodules https://github.com/raucal/Tracking-CAD.git
cd Tracking-CAD
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status
```

Use the pinned revisions, rather than `update --remote`. Model sources and SDU adaptations are documented in [Provenance](PROVENANCE.md).

## Create the base runtime

With Conda available:

```bash
conda create -n cad-runtime python=3.10 pip
conda activate cad-runtime
mkdir -p ../.venvs
python -m venv --system-site-packages ../.venvs/assembly-runtime
../.venvs/assembly-runtime/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

These are CUDA 12.4 wheels, as listed in the [PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251). CPU-only wheels do not support the pose backends in this configuration.

Edit the corresponding fields in `config/settings.json`, retaining the other settings:

```json
{
  "paths": {
    "runtime_python": "../../.venvs/assembly-runtime/bin/python",
    "foundationpose_python": "../../.venvs/foundationpose/bin/python"
  }
}
```

JSON paths are relative to the file containing them. Change these paths if you choose another environment location. Constraints pin package versions; they do not supply system compilers, weights, or Blender.

## Configure paths and build tools

```bash
bash run.sh settings
nvidia-smi
nvcc --version
```

Set `paths.sam6d` to `../models/SAM-6D`, `paths.foundationpose` to `../models/FoundationPoseOfficial`, and `paths.blender` to the real Blender executable, relative to `config/settings.json`. The distributed Blender path points to the sibling `../.runtime/blender/` installation.

If `nvcc` is not on PATH, set `CUDA_HOME` to the CUDA Toolkit directory and add its `bin` directory to PATH. Otherwise the installer detects the toolkit through `nvcc`. `TORCH_CUDA_ARCH_LIST` can select the PEM build architecture; by default it is inferred from the available GPU. `runtime.build_jobs` sets `MAX_JOBS`; some FoundationPose build steps are separately limited to two jobs.

## Install SAM6D and PEM

```bash
bash run.sh install
```

The installer installs `pipeline/setup/requirements-sam6d.txt` with `constraints-assembly-runtime.txt`, downloads or reuses SAM ViT-H, DINOv2, FastSAM, and PEM weights in `checkpoints/`, creates relative weight links, builds PointNet2 in `pipeline/native/pointnet2/` if missing, and runs diagnostics.

ViT-B and ViT-L require their own compatible checkpoints. FoundationPose weights and example data/templates are separate steps.

## Install FoundationPose

Obtain the official refiner and scorer weights following [FoundationPose data preparation](https://github.com/NVlabs/FoundationPose#data-prepare). No training dataset is needed. The expected layout is:

```text
models/FoundationPoseOfficial/weights/
├── 2023-10-28-18-33-37/
│   ├── model_best.pth
│   └── config.yml
└── 2024-01-11-20-02-45/
    ├── model_best.pth
    └── config.yml
```

In the existing installation these weights are kept in `checkpoints/foundationpose/` and linked from the model checkout. Then run:

```bash
bash run.sh install-foundationpose
```

This creates the venv if missing, shares base runtime libraries through `assembly_runtime.pth`, installs FoundationPose requirements with their constraints, and builds native extensions. PyTorch3D uses `v0.7.9`; nvdiffrast is pinned to `729261dc64c4241ea36efda84fbf532cc8b425b8`. `mycpp` is built in `pipeline/native/foundationpose/`.

The existing PyTorch3D binary came from a temporary source directory that is no longer present. Its version/binary were audited, but rebuilding on another machine still requires validation against the selected PyTorch version and compiler. Do not manually patch the official FoundationPose checkout: SDU compatibility is applied by the runtime adapter.

## Prepare data and templates

If `data/prepared/` is complete, reuse it. To prepare a replacement dataset, edit `data/dataset.json` to point to the original CADs, RGB-D captures, and catalog, then write to a new directory:

```bash
bash run.sh prepare --config data/dataset.json --output data/prepared-new
bash run.sh inspect --manifest data/prepared-new/manifest.json
bash run.sh sam render --manifest data/prepared-new/manifest.json
```

Preparation creates centered CADs, RGB images, raw depth, calibration, and a manifest. Rendering creates 42 views per CAD with masks and metric XYZ coordinates. `sam render --part CAD_KEY` selects a single part.

Select the new directory through `paths.prepared` in settings and `manifest` in the suite profile. See [Assembly selection](REFERENCE.md#assembly-selection) for separate assembly datasets. Original data paths may point to the sibling `PDDL/` project; copying CADTracking alone does not supply those sources.

`--repair-topology` requires optional `manifold3d`. Full template proximity audits may require `rtree`; routine detection and tracking do not require running that audit. Prepared data is not rebuilt on startup, including when FoundationPose uses manual initialization. See [Storage and cleanup](REFERENCE.md#storage-and-cleanup).

The Blender helper currently fixes rendering to 256×256 pixels and 16 samples. Changing only `runtime.render_size` or `runtime.render_samples` does not change its output because those values are not yet forwarded to the helper.

### Assets required after cloning

A Git clone includes code, configuration and the two selected exported evidence bundles. It does **not** include the following ignored or external assets:

| Asset | Needed for | What the recipient must do |
| --- | --- | --- |
| `data/prepared/` (about 352 MiB) | Current UI startup, example captures, catalog matching, and suite processing | Receive the complete folder separately or reconstruct it from original inputs. It contains the manifest, centered CADs, 45 RGB-D examples, and 42 template views for each of the 10 CADs. |
| Source CADs and `assembly.json` inventories under the sibling `PDDL/Assembly CAD/` | Example CAD selection and assembly discovery, identification, and suite validation | Receive the relevant assembly folders and point `assembly_registry`/`cad_dir` to them. The current code expects these inventories even when prepared data is present. |
| Original photos, raw depth, calibration metadata, and catalog selected in `data/dataset.json` | Rebuilding prepared data with `prepare` | Supply these sources and update their paths before preparation. They are not needed merely to read an exported report. |
| `checkpoints/` | Model inference | Download through the SAM6D installer and obtain the FoundationPose weights as described above. |
| Python environments, Blender, and native extensions | Running the application and models | Install/build them for the recipient's machine; do not copy local venvs or compiled `.so` files as a portable installation. |

The distributed dataset configuration currently points to `PDDL/DINOPDDL/realsense_photos/`. That directory is absent from the inspected working copy, although prepared captures remain available. Consequently, rebuilding those examples from scratch requires locating/restoring the originals and updating `photos_dir`. The source CAD and catalog paths were present when checked. Do not delete the working prepared data on the assumption that the configured source photographs are available.

For a handoff, provide the complete prepared folder and the corresponding assembly CAD/inventory folders separately from Git. Preserve their documented sibling layout or update the configuration and any external paths stored in the manifest. Prepared data alone does not supply those source CADs or inventories. No external data bundle is published by this repository yet.

For **report viewing only**, the complete exported folder in `results/reports/` is sufficient: it contains its own images and reference gallery and needs none of the runtime assets above. Keep its resource folders beside `index.html`.

### Repository exclusions

The root [.gitignore](../.gitignore) keeps weights, prepared datasets, captures, individual executions, interface sessions, caches, native builds, venvs, personal `.local.json` profiles, environment secrets, and editor files local. Only the two selected evidence bundles, their logs, the report index and `results/dashboard` are eligible for version control. Other suites and exports remain ignored; see [retention policy](RESULTS.md). Rules do not remove files already tracked and do not alter the independent Git indexes of model submodules. Review changes within a submodule separately before committing there.

## Configure RealSense

Install the SDK and USB rules using the [official librealsense instructions](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md). `pyrealsense2` is included in runtime requirements.

Connect the camera to the machine running the service, including for remote browser access. Check that the interface lists it and no other process owns it. Installation commands do not start acquisition or move the robot.

## Validate the installation

```bash
bash run.sh settings
bash run.sh sam doctor
bash run.sh ui --preload none
```

Inspect CUDA, checkpoints, FoundationPose availability, and example CADs in the web interface. Stop it with Ctrl+C, then use `bash run.sh ui` for normal startup.

```bash
../.venvs/assembly-runtime/bin/python -m pip check
../.venvs/foundationpose/bin/python -m pip check
bash run.sh test
```

Adapt Python paths if you changed environment locations. Diagnostics establish that components load; assess pose quality by inspecting the CAD projection on your scene.

## Installer files

| File in `pipeline/setup/` | Purpose |
| --- | --- |
| `install_models.sh` | SAM6D dependencies/weights and PEM extension. |
| `install_foundationpose.sh` | FoundationPose environment and extensions. |
| `requirements-sam6d.txt` | Data, UI, ISM, PEM, camera, and rendering dependencies. |
| `requirements-foundationpose.txt` | FoundationPose-specific dependencies. |
| `requirements-optional.txt` | Optional topology repair and mesh-proximity audit packages. |
| `constraints-assembly-runtime.txt` | Recorded base runtime package versions. |
| `constraints-foundationpose.txt` | Recorded FoundationPose environment versions. |

The two environments have different package requirements and versions. Keep their dependency lists separate and validate both backends when changing them.
