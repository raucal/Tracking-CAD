# SAM6D identification and FoundationPose evaluation

Run from `CADTracking/` after installing the configured GPU environments, weights and prepared dataset:

```bash
EVAL_RUN="results/evaluations/sam6d_fp_$(date -u +%Y%m%dT%H%M%SZ)"
bash run.sh evaluate --output "$EVAL_RUN"
../.venvs/assembly-runtime/bin/python -B pipeline/tools/audit_identification.py --run "$EVAL_RUN"
```

Add `--frame cassette_far` for a smoke case and use another output directory for the complete evaluation. Existing directories are never overwritten. The retained latest report is `results/evaluations/sam6d_fp_corner_full_20261111/index.html`.

SAM generates fresh masks. The original ISM assignment selects a primary region, subject to the documented corner rule. Every CAD is then scored on that same mask with the full RGB-D formula. The highest eligible score selects the final CAD; ties use `category_id`. FoundationPose registers only the winner and never reranks CAD identities. FoundationPose scores for other CADs remain null with an explicit reason; their measured ISM scores remain available.

The default `--corner-filter` excludes only `03a_vertical_option_a` when its xywh box centre lies at x>0.72 of image width and y>0.65 of image height. It applies during proposal selection and final ranking. Measured scores remain intact: excluded candidates have `eligible=false`, `filter_reason`, `raw_rank` and `rank=null`. Margins and top-k use eligible candidates. Use `--no-corner-filter` to reproduce the unfiltered policy. The UI and SAM6D worker share this rule in `pipeline/region_filters.py`.

The expected CAD is joined strictly after prediction and pose. The manifest labels themselves are not independently verified. A primary mask can select background or a different part; this remains an evaluation error. Because the corner heuristic was requested after inspecting this dataset, its results are not held-out evidence.

## Scores and pose conventions

`ISM = (semantic + appearance + geometry * visible_ratio) / (2 + visible_ratio)`.

Semantic scores aggregate the top five clipped DINOv2 cosine similarities over 42 CAD views. Appearance uses patch similarities on the semantic winner view. Geometry is projected 2D bounding-box IoU, not a 6D surface fit. Visibility is a patch-similarity fraction. These components and the composite score lie in [0,1] but are not calibrated probabilities. FoundationPose exposes a pose-ranking network score on a different scale; do not mix it with ISM as a probability.

The suite uses independent per-row IoU to avoid the historical upstream batch-wide intersection condition. This correction is limited to the evaluation process.

Pose outputs are `T_camera_from_model_mm` and `T_camera_from_input_cad_mm`, with OpenCV camera X right, Y down and Z forward. Translation is in millimetres. FoundationPose consumes meshes and depth in metres internally; conversion occurs exactly once in each direction. The original-CAD transform is `T_camera_from_model_mm @ inverse(T_local_from_sam_mm)`.

No sufficient pose ground truth or symmetry annotations are available. A successful registration does not establish accuracy; ADD/ADD-S, VSD and angular accuracy are not reported.

## Outputs and reproducibility

`run.json` records configuration, command, model/source hashes, versions, device, reference cache hits, rankings, poses, timing and input correspondence checks. Only signature-verified CAD reference features are reused; segmentation, image matching and registration run afresh. Seed 0 and category-specific sampling seeds are recorded. GPU numerical results need not be bit-identical.

`index.html` displays the complete score matrix and case overlays. The selected CAD cell is green; other cells with absolute score difference strictly below 0.10 are orange. Colour indicates selection/proximity, not correctness. CSV/JSON include wide and long scores, pose matrices, confusion matrix, per-CAD metrics and latency. Non-evaluated candidates retain nulls and reasons. `audit_identification.py` verifies outputs and produces plots without rerunning inference.

The historical PEM suite has a separate detailed report with per-view diagnostics. Its input set and protocol differ; see [retained results](RESULTS.md). Historical schema field names and source snapshots remain unchanged for compatibility. Reports are presentations of saved measurements, not new GPU runs.
