# Audit and findings

36/45 predictions match the manifest CAD. Top-3: 40/45; top-5: 41/45. Completed poses: 45; identification failures: 0; pose failures: 0.

The original audit passed 1192/1192 correspondence, formula, ranking, pose and export checks. See [verification.json](verification.json). These checks certify exported measurements, not physical pose accuracy.

Median margin for correct predictions: 0.107635; for incorrect predictions: 0.015649. Scores and margins are not calibrated probabilities. High-scoring mistakes are recorded in summary.json: wrong_high_scores.

Refinement from eligible semantic ranking to eligible full ISM changed 6 identities. Recovered: ['io_rot_side']. Lost: ['cassette_far', 'io_rot_far', 'sensor_rot_far']. Correct semantic predictions: 38; correct final predictions: 36. FoundationPose registers the winner and never changes identity.

No primary mask among the nine incorrect cases has its centre in the excluded corner. Without annotated masks, mask accuracy cannot be measured. Inspect the contact sheet and per-case overlays.

## Protocol and limitations

- SAM ViT-H supplies masks; DINOv2 ViT-L/14 supplies descriptors. Every CAD is evaluated on the same primary region with the complete ISM formula. Expected labels are joined after prediction.

- The scene-specific rule excludes only `03a_vertical_option_a` when its bounding-box centre has x > 0.72 × width and y > 0.65 × height. It was requested after inspecting these data. Filtered results are not an independent held-out validation.

- The upstream IoU implementation tests `(wh_inter > 0).all()` across the batch; one disjoint pair can zero every IoU. Evaluation uses independent `paired_iou`; identical and disjoint pairs return [1, 0]. Historical outputs retain the original behaviour.

- `process_saved_demos.py` supplies known CAD identity and therefore performs a different task. The historical `suite` command uses PEM. Neither is merged into this evaluator’s predictions or pose scores.

- Prepared RGB, depth, calibration and CAD hashes were verified at execution. Some original `source_*` paths are absent. Physical depth alignment and calibration accuracy cannot be certified from arrays alone.

- Poses map original CAD coordinates into OpenCV camera coordinates, with translation in millimetres. FP scores range from 23.125 to 136.15625, on a separate scale from ISM. No pose ground truth or symmetry annotations are available; ADD/ADD-S, VSD and angular accuracy are not measured.

- Per-stage times overlap with subtotals. Loading, cold starts, cache hits and competing GPU work affect latency. The wireframe includes hidden edges and is an inspection aid.

## Evidence and retention

The two selected bundles are documented in [docs/RESULTS.md](../../../docs/RESULTS.md). Original JSON/CSV, images, logs and executable source snapshots retain their bytes and historical schemas. This English presentation was refreshed without model inference. Paths inside snapshots describe the original machine; unselected runs were removed. Before/after filter results remain in [comparison.md](comparison.md) and filter_comparison.json.

The first 22-image suite contains assigned-detection maxima and separate per-view diagnostics, with PEM poses. It has a different capture set and no original per-photo ground truth. No causal accuracy gain or directly comparable timing against that suite is claimed.

## Reproduction

See [report.md](report.md) for the measured command and protocol. Configure local asset paths first. Historical source snapshots document the exact original implementation; current maintained code may have a newer Git revision.

## Visualizations

[Confusion matrix](confusion_matrix.png), [score heatmap](score_heatmap.png), [margins](margins.png), [latency](latency.png) and [mask contact sheet](mask_contact_sheet.jpg). Each case also includes masks, pose overlays and a CAD projection.
