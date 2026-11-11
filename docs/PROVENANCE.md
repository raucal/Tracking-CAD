# Sources and provenance

[README](../README.md) · [Installation](INSTALLATION.md) · [Reference](REFERENCE.md)

CADTracking combines external models with its own application layer. Model sources are pinned Git submodules; weights and machine-specific environments are installed separately.

## Model revisions

| Component | Source | Pinned revision |
| --- | --- | --- |
| SAM-6D official base | [JiehongLin/SAM-6D](https://github.com/JiehongLin/SAM-6D) | `1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2` |
| SAM-6D integration fork | [raucal/SAM-6D](https://github.com/raucal/SAM-6D), branch `cadtracking` | `ab672d3` (2026-11-11 integration cleanup) |
| FoundationPose official | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | `a1b694b83e633c2cb6115b9063d940a687759392` |
| SDU adaptation source | [SDU foundation_pose](https://gitlab.sdu.dk/robot_learning/vision/foundation_pose) | `f4b04cd30bdae0b72a0a548192965782b5553587` |

The parent repository's Git tree is authoritative for submodule revisions. The `branch` entries in `.gitmodules` identify development branches; `git submodule update --remote` can advance revisions and should not be used to reproduce this installation.

The SAM-6D fork's commits `36865b0` and `ad2365a` add or adapt preparation, inference, suites, model compatibility, and PEM pose visualization. Commit `ab672d3` shares the scoped corner rule with the application, reuses atomic JSON output and translates the fork-owned adapters. The published algorithms and models belong to their original authors.

## SDU compatibility

`models/FoundationPoseOfficial/` retains the official source. The separate [SDU integration](../integrations/foundationpose_sdu/README.md) contains:

- `files/`: five original SDU files preserved byte for byte: `README.md`, `real_sense_interface.py`, `run_video_demo.py`, `run_single_pose_robot_demo.py`, and `grasp_config_ui.py`.
- `sdu.patch`: SDU changes to official files, including `Utils.py`, environment, gitignore, and documentation.
- `provenance.json`: revisions, inventory, and compatibility hashes.
- `LICENSE.upstream`: the retained NVIDIA license.
- `runtime.py` and `run.py`: CADTracking adapters; `pipeline/foundationpose_sdu.py` bridges the application to them.

At runtime, the adapter validates the official `Utils.py`, applies SDU's dtype/device compatibility changes to a cached copy, validates the result by hash, and loads it in memory. It does not edit the official checkout. The derived copy in `.cache/foundationpose_sdu/` is regenerable. Documentation and environment changes in the patch are retained as evidence and are not applied to the runtime. `estimater.py` matches the official base.

`files/UR5eGripper/` came from local, untracked installation assets rather than the SDU commit; that distinction is recorded in `provenance.json`. Twenty byte-identical mesh copies have been removed from these local assets. XML mesh names are preserved explicitly while filenames point to canonical geometry; the SHA-256 mapping and saved byte count are recorded under `supporting_assets.deduplication` in `provenance.json`. Three personal grasp configuration JSON files remain local and unpublished. The preserved SDU README is an upstream GitLab template, not the installation guide for CADTracking.

A legacy local `models/FoundationPose/` directory may remain protected by filesystem permissions. It is ignored, unused by the current configuration, and absent from new clones.

## CADTracking code and licenses

`pipeline/sam6d_ui/` implements the browser application, model integration, resident workers, camera capture, tracking, and recovery. `pipeline/suites/`, `pipeline/tools/`, `config/`, and `run.sh` provide evaluation, tooling, configuration, and the launcher. This code is maintained in Tracking-CAD, outside the model submodules.

The submodules retain their own licenses and documentation. Checkpoints must be obtained under their authors' instructions and licenses. Separating the SDU files does not change the original license conditions. Both submodules are fetched from GitHub; cloning CADTracking does not require access to SDU's GitLab.

For clone commands and environment setup, see [Installation](INSTALLATION.md#clone-the-project).
