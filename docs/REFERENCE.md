# CAD Tracking reference

[README](../README.md) · [Installation](INSTALLATION.md) · [Provenance](PROVENANCE.md)

Run all commands from the project root. Use this document for operational details; environment setup and source attribution are maintained in the linked guides.

## Commands

`run.sh` selects the configured runtime before importing application modules. Use `--help` on the launcher or a subcommand for its full argument list. Installation commands perform changes directly; follow the installation guide before running them.

| Command | Purpose |
| --- | --- |
| `bash run.sh ui` | Start the browser application. Use `--port 8771` for another port or `--preload none` to skip startup model loading. |
| `bash run.sh evaluate --output results/evaluations/ID` | Run label-blind SAM6D + FoundationPose identification; see [evaluation](EVALUATION.md). |
| `bash run.sh suite run` | Run the suite profile, including PEM by default, and export a report. |
| `bash run.sh suite report --output results/suites/ID` | Export saved results without rerunning inference; `--zip` also creates a ZIP. |
| `bash run.sh suite pose --output results/suites/ID` | Add PEM to saved detections; requires CUDA and dataset resources. |
| `bash run.sh detect --example control_board --part 05_control_board --mode rgbd --pose` | Analyze one example; defaults to `results/runs/`. |
| `bash run.sh scores --run results/suites/ID` | Compare CAD hypotheses on the same primary region. |
| `bash run.sh sam doctor` | Check SAM6D/PEM dependencies, CUDA, weights, and templates without inference or camera acquisition. |
| `bash run.sh sam render` | Render 42 catalog views per CAD with Blender; `--part CAD_KEY` selects one part. |
| `bash run.sh sam segment --frame control_board` | Advanced model-adapter ISM run; defaults under `data/prepared/sam6d/results/`. |
| `bash run.sh sam pose --frame control_board` | Advanced PEM run on that adapter's existing detections. |
| `bash run.sh prepare --config data/dataset.json --output data/prepared-new` | Prepare CADs/RGB-D and a manifest from available originals; does not render templates. |
| `bash run.sh inspect` | Check dataset depth and CAD frames; `--output FILE` saves findings and `--preview` requests an overview image. |
| `bash run.sh audit --output-dir results/template-audit` | Audit rendered templates; may require optional mesh-proximity dependencies. |
| `bash run.sh settings` | Display effective paths. |
| `bash run.sh install` | Install SAM6D/PEM dependencies, weights, and PointNet2. |
| `bash run.sh install-foundationpose` | Install the FoundationPose environment and extensions; obtain weights separately. |
| `bash run.sh test` | Run automated tests without a real camera. |

## Processing and scores

SAM or FastSAM proposes masks. DINOv2/ISM compares proposals with rendered CAD views. PEM or FoundationPose uses an accepted detection, depth, and calibration to estimate pose. FoundationPose can also start from a manual box and update pose between SAM checks.

RGB-only analysis uses semantic and appearance information. Metric pose requires aligned RGB-D, calibration, and correct CAD units. For RGB-D:

```text
ISM = (semantic + appearance + geometry * visibility) / (2 + visibility)
```

The score is not a calibrated probability or a measure of pose accuracy. DINOv2 supplies global semantic descriptors and local appearance patches. The semantic aggregation `avg_5` averages the five highest view similarities; it need not equal the best individual view score. ISM uses the semantic winning view for appearance and geometry; the audit's appearance winner is additional evidence and does not alter the original decision.

Geometry is a 2D bounding-box IoU: rotate the CAD cloud to the selected view, translate it using the mean RGB-D point inside the mask, project it through camera intrinsics, and compare its box with the SAM box. It does not optimize pose or measure surface distances. Visibility is estimated from patch correspondences rather than depth occlusion tests. The original IoU implementation has a batch-wide non-overlap condition that can zero the whole batch; the audit preserves that result and reports areas for inspection.

Implementation sources are `model/detector.py`, `model/loss.py`, `utils/trimesh_utils.py`, and `utils/bbox_utils.py` under `models/SAM-6D/SAM-6D/Instance_Segmentation_Model/`.

The UI reuses resident SAM/DINOv2 and FoundationPose workers. The default preload selects SAM; FoundationPose is also initialized by the service's preload workflow. Suites reuse ISM within a run and estimate PEM in a separate worker, with separate loading/timing. A suite benchmark is not a real-time tracking benchmark.

## Configuration

| File | Scope |
| --- | --- |
| `config/settings.json` | Runtime paths, models, data, weights, caches, and build settings. |
| `config/interface.json` | Server, inference, camera, tracking, and initial example selection. |
| `config/suite.json` | Suite dataset, frames, inference parameters, and destinations. |
| `data/dataset.json` | Source CADs, captures, catalog, names, and assembly selection for preparation. |

Paths in JSON files resolve relative to the containing file; CLI paths resolve relative to the current directory. Absolute paths and `~` are supported. `--settings FILE` overrides `CAD_TRACKING_SETTINGS`; `ui --config FILE` and `suite run --config FILE` select profiles, also selectable through `CAD_TRACKING_INTERFACE_CONFIG` and `CAD_TRACKING_SUITE_CONFIG`.

Model files are selected by `paths.dinov2_checkpoint`, `paths.sam_vit_h_checkpoint` (or `sam_vit_l_checkpoint` / `sam_vit_b_checkpoint`), `paths.fastsam_checkpoint`, and `paths.pem_checkpoint`. `paths.foundationpose_weights` selects the root containing both complementary FoundationPose checkpoints and their `config.yml` files. These paths point directly to physical files/directories; installers do not create checkpoint symlinks.

Custom interface/suite profiles inherit omitted fields and reject unknown keys. `tracking.quality` supports partial overrides. CLI options override profiles, and web controls can override parameters per request. Settings require the complete paths/runtime configuration; example snippets are edits, not complete replacements. Startup segmentor preload and initial inference segmentor are separate settings.

For example, save this as `config/interface.local.json`:

```json
{
  "server": {"port": 8771, "preload": "fastsam"},
  "inference": {"segmentor": "fastsam", "mode": "rgbd", "pose_backend": "foundationpose", "pose": true},
  "tracking": {"redetect_interval_s": 10}
}
```

```bash
bash run.sh ui --config config/interface.local.json
bash run.sh ui --config config/interface.local.json --port 8772
bash run.sh --settings config/settings.json ui
```

The following tables describe the distributed JSON values. Effective values depend on the selected profile, CLI options, and web controls.

### config/settings.json

| Field | Distributed value | Meaning |
| --- | --- | --- |
| `paths.sam6d` | `"../models/SAM-6D"` | SAM-6D integration fork, containing the original SAM-6D/ source directory. |
| `paths.foundationpose` | `"../models/FoundationPoseOfficial"` | Pinned official FoundationPose checkout; weights are linked under weights/. |
| `paths.prepared` | `"../data/prepared"` | Prepared manifest, CADs, examples, and templates. |
| `paths.results` | `"../results"` | General result root; interface/suite profiles can select other destinations. |
| `paths.captures` | `"../captures"` | Camera captures and recordings; the interface profile sets the effective location. |
| `paths.checkpoints` | `"../checkpoints"` | Persistent model weights. |
| `paths.native` | `"../pipeline/native"` | Compiled PEM and FoundationPose extensions. |
| `paths.runtime_python` | `"../../.venvs/assembly-runtime/bin/python"` | SAM6D/PEM runtime Python executable. |
| `paths.foundationpose_python` | `"../../.venvs/foundationpose/bin/python"` | FoundationPose Python executable, sharing base runtime libraries. |
| `paths.blender` | `"../../.runtime/blender/blender-3.3.1-linux-x64/blender"` | Blender executable for CAD template rendering. |
| `paths.file_root` | `"../.."` | Default server file-browser root; the UI profile can limit it. |
| `paths.reference_cache` | `"../.cache/reference_features"` | Reference embeddings used by comparison tools. |
| `paths.ui_reference_cache` | `"../.cache/ui_reference_features"` | Reference embeddings reused by the interface. |
| `paths.runtime_cache` | `"../../.runtime/cache"` | Runtime caches and Torch extension builds. |
| `paths.matplotlib_cache` | `"../../.runtime/matplotlib"` | Matplotlib cache/configuration. |
| `paths.dataset` | `"../data/dataset.json"` | Example source dataset description. |
| `runtime.threads` | `16` | CPU threads; not an inference frame rate. |
| `runtime.render_size` | `256` | Adapter render-size argument; the current Blender helper still fixes 256 px. |
| `runtime.render_samples` | `16` | Adapter sample-count argument; the current Blender helper still fixes 16 samples. |
| `runtime.build_jobs` | `2` | Native build parallelism, exported as MAX_JOBS. |

### config/interface.json

| Field | Distributed value | Meaning |
| --- | --- | --- |
| `server.port` | `8770` | Local HTTP port, bound to 127.0.0.1. |
| `server.preload` | `"sam"` | Startup segmentor: sam, fastsam, or none. |
| `server.backbone` | `"vit_h"` | Preloaded SAM variant: vit_h, vit_l, or vit_b; ignored by FastSAM. |
| `server.data_dir` | `"../results/interface"` | Interface jobs, cache, logs, and tracking sessions. |
| `server.capture_dir` | `"../captures"` | Saved RealSense images and recordings. |
| `server.file_root` | `"../.."` | Server file-browser root. |
| `inference.mode` | `"rgb"` | rgb or aligned/calibrated rgbd; pose requires rgbd. |
| `inference.device` | `"cuda"` | auto, cpu, or cuda; pose requires CUDA. |
| `inference.backbone` | `"vit_h"` | SAM variant: vit_h, vit_l, or vit_b; ignored by FastSAM. |
| `inference.cad_units` | `"mm"` | Input CAD physical units: mm, cm, or m. |
| `inference.segmentor` | `"sam"` | sam or fastsam. |
| `inference.pose` | `false` | Enable pose estimation with the selected backend; requires RGB-D and CUDA. |
| `inference.pose_backend` | `"pem"` | pem or foundationpose; selecting FoundationPose enables pose. |
| `inference.fastsam_conf` | `0.05` | FastSAM proposal threshold (0–1), separate from ISM confidence. |
| `inference.fastsam_iou` | `0.9` | FastSAM IoU threshold (0–1). |
| `inference.min_score` | `0.3` | Minimum retained ISM score; must not exceed confidence. |
| `inference.confidence` | `0.4` | Final ISM acceptance threshold (0–1); not a calibrated probability. |
| `inference.stability` | `0.97` | SAM mask stability threshold (0–1), overriding the model YAML. |
| `inference.pred_iou` | `0.88` | SAM predicted mask-quality threshold (0–1). |
| `inference.points_per_batch` | `64` | SAM prompts per batch (1–128); affects memory and throughput. |
| `inference.nms` | `0.25` | Per-object non-maximum suppression IoU threshold (0–1). |
| `inference.aggregation` | `"avg_5"` | Semantic view aggregation: avg_5, mean, median, or max. |
| `camera.width` | `640` | RGB-D width: 640 or 1280 with the corresponding height. |
| `camera.height` | `480` | RGB-D height: 480 or 720 for 640×480 or 1280×720. |
| `camera.fps` | `15` | Acquisition rate: 15 or 30 FPS; not guaranteed inference throughput. |
| `camera.preview_width` | `640` | Maximum JPEG preview width. |
| `camera.preview_height` | `480` | Maximum JPEG preview height. |
| `camera.jpeg_quality` | `70` | Preview JPEG quality (1–100); does not change saved RGB-D. |
| `tracking.identify_assembly` | `false` | Automatically identify the part among the selected assembly's prepared CADs. |
| `tracking.redetect_interval_s` | `10.0` | Periodic SAM check interval; 0 disables periodic checks while preserving automatic loss recovery. |
| `tracking.recovery_retry_s` | `1.0` | Pause after failed recovery attempts; the first attempt is immediate. |
| `tracking.max_detection_age_s` | `5.0` | Maximum SAM frame age at completion; detections predating loss are discarded. |
| `tracking.quality_bad_frames` | `3` | Consecutive geometrically inconsistent frames required to confirm loss. |
| `tracking.quality.enabled` | `true` | Enable depth and motion checks; these are heuristic, not calibrated confidence. |
| `tracking.quality.sample_points` | `1024` | CAD surface samples computed once per part. |
| `tracking.quality.depth_tolerance_mm` | `20.0` | Maximum depth difference for a supported surface sample. |
| `tracking.quality.min_depth_samples` | `20` | Minimum visible samples with depth for evaluation. |
| `tracking.quality.min_depth_support` | `0.35` | Minimum supported fraction after excluding partial occlusions. |
| `tracking.quality.min_valid_depth_fraction` | `0.25` | Minimum valid-depth fraction on the projected surface. |
| `tracking.quality.min_visible_fraction` | `0.1` | Minimum projected front-surface fraction inside the image. |
| `tracking.quality.max_occlusion_fraction` | `0.7` | Maximum tolerated occlusion fraction. |
| `tracking.quality.translation_slack_mm` | `20.0` | Position allowance before applying the speed limit. |
| `tracking.quality.max_translation_speed_mm_s` | `1000.0` | Maximum CAD-center speed between accepted poses. |
| `tracking.quality.rotation_slack_deg` | `20.0` | Angular allowance for orientation noise. |
| `tracking.quality.max_rotation_speed_deg_s` | `360.0` | Maximum angular speed between accepted poses. |
| `tracking.quality.max_motion_gap_s` | `0.5` | Skip motion-jump checks beyond this gap; continue depth evaluation. |
| `examples.frame` | `"control_board"` | Initially selected example capture. |
| `examples.cad` | `"05_control_board"` | Initially selected example CAD key. |

### config/suite.json

| Field | Distributed value | Meaning |
| --- | --- | --- |
| `manifest` | `"../data/prepared/manifest.json"` | Prepared dataset manifest (schema_version 1). |
| `output_root` | `"../results/suites"` | Suite result root; new runs use UTC timestamp directories. |
| `export_root` | `"../results/reports"` | Exported HTML/CSV reports and optional ZIPs. |
| `device` | `"auto"` | auto, cpu, or cuda; PEM requires CUDA. |
| `min_score` | `0.4` | Minimum retained ISM score (0–1), no greater than confidence. |
| `confidence` | `0.5` | Final ISM acceptance threshold (0–1). |
| `stability_score` | `0.97` | Effective SAM stability threshold; overrides the model YAML's 0.85. |
| `segmentor` | `"sam"` | sam or fastsam. |
| `backbone` | `"vit_h"` | SAM variant; ignored by FastSAM. |
| `threads` | `16` | CPU threads, not inference FPS. |
| `reference_cache` | `"../.cache/reference_features"` | Reference embedding cache for comparison tools. |
| `pose` | `true` | Calculate PEM for accepted detections; --no-pose overrides it. |
| `frame` | `[]` | Capture names; an empty list selects all. CLI --frame can be repeated. |
| `assembly` | `"printed_assembly"` | Selected assembly; --assembly overrides it. |

### data/dataset.json

| Field | Distributed value | Meaning |
| --- | --- | --- |
| `assembly_id` | `"converter_demo_v1"` | Example assembly identifier. |
| `cad_dir` | `"../../PDDL/Assembly CAD/Assembly parts/printed_assembly"` | Original CAD directory. |
| `photos_dir` | `"../../PDDL/DINOPDDL/realsense_photos"` | Original RGB, raw depth, and calibration directory. |
| `catalog` | `"../../PDDL/DINOPDDL/catalogs/printed_assembly_catalog.yaml"` | YAML catalog relating parts, CAD keys, and options. |
| `photo_aliases` | `{"front_housing_2": "front_housing", "termal_fan": "thermal_core"}` | Example selection aliases; not classification labels. |
| `display_names` | `{"03a_vertical_option_a": "sensor_module", "03b_vertical_option_b": "communications_module"}` | Visible-name corrections preserving CAD IDs and geometry. |
| `assembly` | `"printed_assembly"` | Selected assembly; --assembly overrides it. |
| `assembly_registry` | `"../../PDDL/Assembly CAD"` | Directory containing per-assembly inventories, or a legacy registry JSON. |

## Images and CAD input

Select server files or upload them, choose a CAD, declare mm/cm/m, and select RGB or RGB-D. Enable PEM or choose FoundationPose for pose estimation, then click **Detect object**. The result includes scores, timings, the actual device, and available pose/projection. Example names come from the prepared manifest; photo filenames are not ground-truth classification labels.

| Input | Requirement |
| --- | --- |
| RGB | Supported color image, up to 4 megapixels. |
| CAD | STL, OBJ, or PLY with explicit physical units. |
| Depth | Uncolored 16-bit PNG or `uint16` NPY, matching the RGB dimensions and alignment. |
| Calibration | Intrinsics and depth scale in supported BOP/SAM6D or RealSense metadata JSON. |

BOP/SAM6D `depth_scale` means **millimeters per raw depth unit**, not meters per unit. Preparation converts RealSense metadata where appropriate.

For manual initialization, select FoundationPose and **Manual box · FoundationPose only**, freeze a capture, and mark the object's rectangle. This is an approximate initial mask, not a precise segmentation. SAM/DINOv2 is bypassed for localization; after loss, manually locate the object again. Start with `ui --preload none` to skip SAM preload in a manual-only session; FoundationPose loads on first use and its worker is reused.

## Camera and tracking

Select **Live Intel RealSense**, a device, and connect. Controls support 640×480 and 1280×720 at 15 or 30 FPS, subject to device/SDK support.

| Control | Behavior |
| --- | --- |
| **Analizar fotograma actual** | Analyze the latest RGB-D frame when the request is sent. |
| **Start live detection** | Repeat complete analyses at the available inference rate. |
| **Locate and start fast tracking** | Initialize FoundationPose, then update pose on the latest frames. |
| Capture/export controls | Save RGB-D or download result originals/overlays. |
| Video recording controls | Save RGB-D, calibration, timestamps, and a preview video; recording-tracking results can also be downloaded. |
| **Open compact viewer** | Open a compact camera/pose window; it does not reduce inference cost. |

Acquisition keeps the latest frame rather than queuing every frame for inference. Camera and tracking FPS are different measurements; intermediate frames are skipped when processing is slower than acquisition.

### Recovery and periodic checks

Automatic tracking runs SAM immediately after confirmed loss and every 10 seconds by default as a backup. Tracking errors and numerically invalid poses trigger recovery. Depth/motion checks require three inconsistent frames by default; suspect poses are withheld while loss is confirmed. Checks use projected front-surface depth and CAD-center motion, allowing partial occlusions.

Only one SAM detection can be pending. Stale detections and those predating loss are discarded. Recovery uses a fresh frame, shows live camera video while waiting, and retries after the configured pause without reloading models. The interval can change during tracking; `0` disables periodic checks but retains automatic loss recovery. Manual initialization keeps SAM disabled.

These checks are heuristic: geometrically plausible drift can pass. Tune them for object size, depth noise, symmetry, and motion.

### Automatic part identification

Choose **Identify assembly part** as the CAD source and select an assembly. SAM6D compares image proposals with all its prepared CADs and selects the highest accepted score. This requires RGB-D, millimeter units, and complete assembly templates. No accepted score means no declared part.

Single-frame analysis identifies once; live detection repeats it. Fast FoundationPose tracking maintains pose between checks and confirms a detected part change with a new frame before registering and publishing the new CAD pose. The status shows the current part, score, and change count. One object is tracked: this is not simultaneous tracking of all visible parts. Set `tracking.identify_assembly` to `true` for this mode by default.

### CAD and axes viewer

Fast FoundationPose tracking includes **CAD and axes on a black background**, also available in the compact viewer as **CAD + axes · black background**. It renders the current model pose without the camera image, using the initialized GPU. The camera overlay and CAD view use the same centered axes: X red, Y green, Z blue. The object remains in camera coordinates rather than being artificially recentered. The view hides when pose is lost, reports rendering time, and does not rerun SAM.

## Assembly selection

Canonical inventories are `assembly.json` files beside CADs in `PDDL/Assembly CAD/Assembly parts/<assembly>/`. `assembly_registry` points to `PDDL/Assembly CAD` and also supports the legacy JSON registry. See the [shared assembly guide](../../PDDL/Assembly%20CAD/README.md).

The UI discovers available inventories and lets you select an assembly and its original STL part. `data/dataset.json` selects the preparation assembly; `prepare --assembly printed_assembly_2` overrides it without editing the JSON. Each assembly needs its own captures and templates.

`config/suite.json` and `suite run --assembly NAME` select the benchmark assembly. When using the default manifest setting with another assembly, the launcher selects `data/assemblies/<assembly>/prepared/manifest.json`. That manifest must belong to the selected assembly. Non-default assembly suites go under `results/suites/<assembly>/`; existing results are not moved.

## Suites

A suite evaluates the selected manifest captures against the CAD catalog. Suites use PEM for pose; selecting FoundationPose in the UI does not change the suite backend.

```bash
# Preview the effective command without inference
bash run.sh suite run --dry-run

# Select captures; repeat --frame as needed
bash run.sh suite run --frame hmi_panel --frame control_board

# Override the profile's segmentor or pose settings
bash run.sh suite run --segmentor fastsam
bash run.sh suite run --no-pose
bash run.sh suite run --device cpu --no-pose
```

A run creates a new timestamp directory. `--output` can select another **new** directory; a nonempty output is rejected. A custom suite JSON can override only selected fields, for example `{"confidence": 0.6, "frame": ["hmi_panel"]}`, and be selected with `--config config/suite.local.json`.

| Option | Effect |
| --- | --- |
| `run`, `report`, `pose` | New inference, report export, or PEM enrichment respectively. `report` and `pose` require `--output`. |
| `--config`, `--manifest`, `--assembly` | Profile, prepared dataset, and assembly selection. |
| `--output` | New run directory for `run`, existing run for `report`/`pose`. |
| `--export-dir` | Separate report destination; rejects files unrelated to the exporter. |
| `--frame` | Capture name; repeat to select several, omit for the profile selection. |
| `--device`, `--segmentor`, `--backbone` | Device and segmentor choices described in the configuration tables. |
| `--min-score`, `--confidence`, `--stability-score` | Retention, acceptance, and SAM stability thresholds; require `0 <= min-score <= confidence <= 1`. |
| `--threads`, `--reference-cache` | CPU threads and reference embedding cache. |
| `--pose`, `--no-pose` | Enable or disable PEM over the profile. |
| `--view-details`, `--no-view-details` | Enable (default) or skip the extra per-view audit after a new run. |
| `--overwrite-pose` | Recalculate saved poses, preserving `pose.before.json`. |
| `--dry-run` | Print the new-run command without running it. |
| `--no-export` | Keep results without generating the report. |
| `--zip` | Also create a report ZIP; it is not created by default. |

To use saved results, replace `ID` with an actual directory name:

```bash
bash run.sh suite report --output results/suites/ID
bash run.sh suite report --output results/suites/ID --zip
bash run.sh suite pose --output results/suites/ID
bash run.sh suite pose --output results/suites/ID --overwrite-pose
```

Each suite retains `run.json`, case outputs, and an exact `dataset_manifest.json` snapshot. Hash checks prevent mixing pose and ISM outputs from different runs. Dataset resources such as templates remain in prepared data and are needed for further processing. The launcher also validates the configured dataset manifest before report/pose commands.

### Per-view audit

New suites audit every CAD view on the **same saved primary mask** per capture: DINOv2 semantic similarity, appearance, winning views, and geometry intermediates. This is extra computation, timed separately in `ism_view_details.json`. It is not the full set of SAM proposals and does not change original detection results. Captures without a saved primary region have no measurements.

To enrich an old suite without rerunning SAM or PEM:

```bash
bash run.sh scores --run results/suites/ID --view-details
bash run.sh suite report --output results/suites/ID
```

The tool refuses to overwrite an existing audit, verifies the source suite hash, and checks that the recalculated original-class score differs by no more than 0.001. For inspection in a different environment, `--view-details --allow-score-drift` records discrepancies explicitly without replacing original scores; `--device cpu` runs the audit without CUDA. The export includes `ism_vistas.csv`. Regenerating HTML alone does not run models.

## Reports and pose interpretation

Serve `results/dashboard` as shown in the README or open an exported folder's `index.html`. **Matrices and coordinate axes** opens the pose gallery; `#poses` links directly to it. Each complete export updates the dashboard symlink.

The report includes saved scores/components, timings, images, masks, CAD comparisons, reference views, and available poses. Captures without accepted detections may have no pose; PEM errors are shown separately.

| Report file | Content |
| --- | --- |
| `index.html` | Offline report, requiring its accompanying resources. |
| `confianza_sam6d_por_imagen.csv`, `confianza_sam6d_por_cad.csv` | ISM scores/components by image and CAD. |
| `comparacion_todos_los_cad.csv` | Saved CAD comparisons. |
| `poses_pem.csv`, `resultados_poses.json` | Poses, transforms, and metadata. |
| `resultados_completos.json` | Exported original ISM result. |
| `ism_view_details.json`, `ism_vistas.csv` | Additional per-view audit. |
| `cases/` | RGB, depth, masks, overlays, audits, and per-case results. |
| `referencias_cad/` | Exported reference images. |

**R** is a 3×3 rotation, **t** a translation, and **T** a 4×4 homogeneous transform. Compare backends using `T_camera_from_input_cad_mm`, which maps the original CAD to OpenCV camera coordinates (X right, Y down, Z forward) with millimeter translation.

The CAD origin can lie outside its geometry or mask. PEM's overlay uses the original CAD frame, while FoundationPose's viewer uses a centered model frame; axes can therefore appear offset even when the exported original-CAD transforms are comparable. A displaced CAD contour/cloud requires checking the estimate, units, intrinsics, depth, and transforms.

PEM's combined score can incorporate ISM and ranks pose hypotheses; it is neither another detection probability nor metric accuracy. `estimated` means an estimate was produced, not validated against ground truth.

## Storage and cleanup

| Location | Role and retention |
| --- | --- |
| `data/dataset.json` | Original dataset description; cannot be regenerated if deleted. |
| `data/prepared/manifest.json` | Prepared index read by UI and suites; not rebuilt at startup. |
| `data/prepared/sam6d/models/`, `frames/` | Centered CADs (PLY for PEM, OBJ for rendering), RGB, raw `uint16` depth, and calibration. Rebuild with `prepare` only if originals remain. |
| `data/prepared/sam6d/templates/` | 42 views per CAD with RGB, masks, poses, and metric XYZ. Explicitly rebuild with `sam render`; embeddings do not replace them. |
| `checkpoints/`, `models/`, `pipeline/native/` | Weights, pinned model source, and system-specific compiled extensions. Keep for operation. |
| `pipeline/sam6d_ui/`, `suites/`, `tools/`, `workers/` | Application, evaluations/reports, utilities, and PEM worker. |
| `pipeline/setup/`, `settings.py`, `launcher.py`, `tests/` | Installers, configuration loading, runtime selection, and tests. |
| `captures/images/ID/`, `captures/videos/ID/` | User-saved RGB-D, calibration, metadata, and recordings. Keep wanted captures. |
| `results/interface/jobs/{sam6d,foundationpose}/ID/` | Individual web analyses. Removable after the interface and workers stop. |
| `results/interface/live/ID/`, `logs/` | Tracking session summaries, SAM checks/recovery records, and worker logs. Removable after stopping the service. |
| `results/interface/cache/cad/` | Custom-CAD references; missing references are rebuilt when that CAD is analyzed again. |
| `results/interface/camera/` | Compatibility links to captures; not an independent copy of recordings. |
| `results/runs/` | Individual CLI analyses and local startup checks. |
| `results/suites/` | New local suite runs; ignored. Historical redundant source suites were removed after preserving self-contained exports. |
| `results/reports/` | Selected 22-image detailed bundle. Other exports stay local and ignored. |
| `results/evaluations/` | Selected 45-image FoundationPose bundle. Other evaluations stay local and ignored. |
| `results/dashboard` | Symlink to the latest report, not another report copy. |
| `.cache/` | Regenerable embeddings and SDU compatibility copies; removing them adds future startup/computation work. |

Git retains only the two bundles listed in [RESULTS.md](RESULTS.md). Other suites and exports, individual runs, interface sessions, captures, prepared data, caches, weights and native builds are ignored. **Ignoring files does not delete them or prevent disk accumulation.** Stop processing before removing local execution artifacts; preserve suite/report folders and wanted captures. Failed jobs retain `error.log`; suites retain `execution.log` and per-part `pose.log`. The HTTP server logs to the terminal.

An exported report can be copied to another computer and viewed without models, CUDA, Python, source suites, or Internet. Copy the **entire real report folder**, preserving its relative symlinks (for example with `cp -a` or a tar archive), including `cases/` and `referencias_cad/`; `index.html` or the dashboard symlink alone is insufficient. Reprocessing or rebuilding an export still requires the application and relevant data. See [Installation](INSTALLATION.md#prepare-data-and-templates) for dataset reconstruction. ASAP planning code and data remain in the sibling ASAP project.

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| Address already in use | Use the existing service, stop it from its terminal, or select another `--port`. |
| Remote browser cannot connect | Keep the service running and forward its port through SSH/VS Code. |
| RealSense missing or busy | Check server connection, SDK/USB permissions, and other camera users. |
| Missing weights, runtime, or templates | Run `settings` and `sam doctor`; follow Installation. Set `paths.foundationpose_weights` to the directory containing both dated weight/config directories. |
| Missing pose matrices/axes | In the historical PEM report open `#poses` and inspect saved errors. In the FoundationPose report open the case overlay and `poses.json`. Newly generated suites can use `suite pose`. |
| Correct mask but poor CAD projection | Check CAD units, aligned depth, intrinsics, and transforms. A good mask does not guarantee a good pose. |
| Smooth stream but slow detections | Acquisition and inference have different rates. Use fast FoundationPose tracking and inspect stage timings. |
| Repeated recovery | Inspect `pose_quality`, depth support, and motion limits; tune the profile for the scene. |
| Missing data after cleanup | Restore the dataset description and rebuild from originals using Installation. |
| Native `.so` fails on another machine | Verify Python/PyTorch/CUDA compatibility and rebuild in that runtime. |

`inspect` checks depth and CAD frames; historical JPEGs can decode differently across libraries when byte hashes are absent. New preparation records source and PNG hashes. `audit` checks template XYZ/scale and may need optional mesh-proximity dependencies. Auxiliary audit files are written only when an output is requested.

For a custom one-off RGB-D analysis:

```bash
bash run.sh detect --rgb PHOTO.png --cad PART.stl --depth DEPTH.png --camera CAMERA.json --mode rgbd --pose --cad-units mm
```
