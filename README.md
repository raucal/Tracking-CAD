# CAD Tracking

Identify a CAD part in an RGB-D image, estimate its 6D pose, and track it with an Intel RealSense camera. The browser application and maintained documentation use English.

SAM ViT-H generates masks. **DINOv2 ViT-L/14** supplies the global and patch descriptors used by SAM6D ISM. The identification evaluator compares every CAD on the same primary mask, chooses the highest eligible full ISM score, and registers that CAD with **FoundationPose**. FoundationPose does not rerank CAD identities. The older suite uses **PEM** and remains a separate experiment.

| Guide | Contents |
| --- | --- |
| [Installation](docs/INSTALLATION.md) | Fresh-machine setup, GPU/native builds, environments, weights and data handoff. |
| [Reference](docs/REFERENCE.md) | Configuration, image/camera workflows, tracking, commands and troubleshooting. |
| [Evaluation](docs/EVALUATION.md) | Label-blind SAM6D + FoundationPose evaluation, score meanings and reproduction. |
| [Retained results](docs/RESULTS.md) | The two selected reports, their protocols and retention policy. |
| [Provenance](docs/PROVENANCE.md) | Model revisions, SDU compatibility and upstream notices. |

## Clone and configure

```bash
git clone --recurse-submodules <your-repository-url> CADTracking
cd CADTracking
```

For an existing checkout, run `git submodule update --init --recursive`. Submodules are pinned to commits; do not use `--remote` to reproduce an experiment. When publishing local submodule changes, push the SAM-6D fork commit before the parent commit that references it.

Paths in `config/settings.json` are relative to that file. The supplied layout expects sibling `.venvs/` and `.runtime/` directories. Select other locations with a settings JSON and `bash run.sh --settings /path/to/settings.json ...`, or set `CAD_TRACKING_SETTINGS`. The installers use the selected environments; they do not bootstrap a missing base Python automatically.

**A Git clone is not a complete inference installation.** Weights, prepared templates, captures, Blender, native extensions and environments are external assets. The example `data/dataset.json` also points to CAD inventories and a catalog in the sibling `PDDL/` project. Configure your own sources or obtain the [asset handoff bundle](docs/INSTALLATION.md#assets-required-after-cloning). The original example photographs are unavailable in this working copy, so preserve `data/prepared/`; rebuilding it from the configured sources is currently impossible.

### Shared DINOv2 checkpoint in this workspace

The local DINOv2 ViT-L/14 checkpoint is stored once at `../PDDL/DINO/torch_hub/checkpoints/dinov2_vitl14_pretrain.pth`. The `paths.dinov2_checkpoint` entry in `config/settings.json` points directly to that file; no checkpoint symlinks are needed. The bytes and SAM6D model are unchanged; DINOPDDL can load the same weights for reference/stage comparison. Each process still has its own RAM/VRAM allocation.

This is a local asset arrangement, excluded from Git. Model **source repositories** are pinned Git submodules, not symlinked folders. All model locations are explicit entries in `config/settings.json`: `dinov2_checkpoint`, `sam_vit_h_checkpoint` (also `vit_l` and `vit_b`), `fastsam_checkpoint`, `pem_checkpoint`, and `foundationpose_weights`. Relative paths start at the settings file directory. Adjust these before installation on another machine. Installers reuse or download the configured files directly; evaluation provenance hashes each physical checkpoint once.

The SAM ViT-H, two FoundationPose checkpoints (refiner and scorer), optional FastSAM and optional PEM weights are different models. Both FoundationPose checkpoints are required: the refiner improves candidate poses and the scorer selects between them. Deleting either disables the supported FoundationPose pipeline. They were retained so the available backends continue to work. Standalone DINOv2 does not supply SAM masks or FoundationPose poses.

## Dependencies and installation

The validated inference environment uses **Linux, Python 3.10, PyTorch 2.5.1 / torchvision 0.20.1 with CUDA 12.4, and NumPy 1.26.4**. Native extensions were built with CUDA Toolkit 12.5. CPU tests do not certify GPU inference or hardware behaviour.

| Component | Dependencies / installation |
| --- | --- |
| System | Bash, Git, C/C++ compiler, CMake, OpenMP, Eigen3, Boost, NVIDIA driver and CUDA Toolkit with `nvcc`. |
| SAM6D application | [requirements.txt](requirements.txt), including the base dependencies under `pipeline/setup/`; versions are constrained by the recorded runtime snapshot. |
| FoundationPose | [requirements-foundationpose.txt](pipeline/setup/requirements-foundationpose.txt), in its separate environment; the installer builds PyTorch3D, nvdiffrast and `mycpp`. |
| CAD template rendering | **Blender 3.3.1 executable**, configured through `paths.blender`; installing `bpy` into the application Python is insufficient. |
| Live camera | RealSense SDK, USB/device access and `pyrealsense2`. Saved RGB-D evaluation does not require a connected camera. |
| Robot alignment | [alignment_pose_cad/requirements.txt](alignment_pose_cad/requirements.txt), including `ur-rtde==1.6.5`; optional and hardware-specific. |
| Development / CPU checks | [requirements-dev.txt](requirements-dev.txt), plus compatible PyTorch/torchvision installed for the test machine. |
| Optional mesh repair | `manifold3d`; optional proximity audits also use `rtree`. See [optional requirements](pipeline/setup/requirements-optional.txt). |

After creating the base runtime and configuring its paths as described in [Installation](docs/INSTALLATION.md):

```bash
# Install the application dependencies, download/reuse weights, and build PEM extensions.
bash run.sh install

# Create/configure the separate FoundationPose environment and build its extensions.
bash run.sh install-foundationpose

# Optional robot feature, in the application environment.
../.venvs/assembly-runtime/bin/python -m pip install -r alignment_pose_cad/requirements.txt
```

`pip install -r requirements.txt` installs Python packages only. It does not replace the system, CUDA, Blender, model checkout, native build or data preparation steps. Do not merge the FoundationPose and application environments or copy existing venvs to another machine.

## Start the browser application

```bash
bash run.sh ui
```

Open **http://127.0.0.1:8770**. Keep the terminal running; stop with **Ctrl+C**. On a remote machine, forward port 8770 through VS Code or SSH:

```bash
ssh -N -L 8770:127.0.0.1:8770 user@server
```

For saved data, expand **Try prepared assembly data**, choose a capture and CAD, then **Run example**. For live input, choose **Live Intel RealSense**, click **Open camera**, select the CAD or assembly, choose FoundationPose, then **Locate and start fast tracking**. **Start live detection** instead repeats complete analyses.

The [robot alignment extension](alignment_pose_cad/README.md) adds capture, preview and explicit execution controls. It starts with motion disabled and requires verified device-specific calibration and limits.

## Evaluate identification and pose

Stop competing inference jobs on the GPU before benchmarking. Use a fresh output directory:

```bash
# Smoke case: SAM + DINOv2 ISM + FoundationPose.
bash run.sh evaluate --output results/evaluations/my_smoke --frame cassette_far

# All manifest cases; default corner filter enabled.
bash run.sh evaluate --output results/evaluations/my_full

# Additional checks and plots; reuse measurements, not inference.
../.venvs/assembly-runtime/bin/python -B pipeline/tools/audit_identification.py \
  --run results/evaluations/my_full
```

The scene-specific filter excludes `03a_vertical_option_a` only when its box centre has **x > 0.72 × width and y > 0.65 × height**. Other CADs and this CAD elsewhere remain eligible. Use `--no-corner-filter` for the unfiltered protocol. The filter was chosen after inspecting these data, so its results are not an independent held-out validation.

Expected manifest labels are used only after prediction. ISM scores are not probabilities. FoundationPose scores have a separate scale; only the winner receives a pose score. Without pose ground truth and symmetry annotations, pose accuracy is not measured.

For the historical **PEM** protocol, use `bash run.sh suite run --manifest /path/to/manifest.json`. This is distinct from `evaluate`; do not combine its scores, timing or pose results with the FoundationPose protocol.

## Open the retained reports

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory results
```

Open **http://127.0.0.1:8766/** and choose one of the two reports. Forward port 8766 when using SSH. The latest main table marks the selected CAD in green and other CADs with **absolute ISM difference < 0.10** in orange; green indicates selection, not correctness.

Only the two selected report bundles are versioned. Other runs, caches and exports stay ignored. See [RESULTS.md](docs/RESULTS.md) for exact retained paths, historical-format compatibility and cleanup records.

## Run tests

```bash
# Existing configured application environment; includes pipeline and robot-extension tests.
bash run.sh test -q

# Direct equivalent; synthetic fixtures need no PDDL checkout, captures, GPU or robot.
PYTHONPATH=pipeline ../.venvs/assembly-runtime/bin/python -B -m pytest \
  -p no:cacheprovider pipeline/tests alignment_pose_cad/tests -q
```

For a machine without an inference installation, follow the [CPU-only setup](docs/INSTALLATION.md#cpu-only-development-checks). The GitLab CI job installs CPU PyTorch and development dependencies and initializes submodules. Tests use temporary synthetic CAD/RGB-D data and simulated hardware. Optional topology tests are skipped if `manifold3d` is absent. GPU registration and real hardware still require separate acceptance checks.

## Repository layout and compatibility

`pipeline/` contains the application, backends, tools, tests and installers. `alignment_pose_cad/` isolates the optional robot feature. `integrations/` preserves upstream adapters and notices; `models/` holds pinned submodules. Configuration and guides live in `config/` and `docs/`. Shared synthetic test fixtures live in the root `conftest.py`.

Maintained messages and prose are English. Historical evidence retains original filenames, identifiers, CSV/JSON schemas and immutable source snapshots where required for provenance. Upstream source/documentation and license notices are preserved rather than rewritten. The historical `termal_fan` capture name is an identifier, not a UI label.

The model repositories retain their own licenses. Obtain checkpoints under their authors’ terms. This cleanup does not relicense upstream code or grant additional rights to it.
