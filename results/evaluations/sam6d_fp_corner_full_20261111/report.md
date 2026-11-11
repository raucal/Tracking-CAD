# SAM6D + FoundationPose evaluation

36 of 45 predictions match the manifest CAD.

Manifest: 45; valid: 45; selected: 45; identification: 45; completed poses: 45.

Protocol: fresh SAM masks; settings.region_filter controls primary-region and final-CAD selection. Every CAD retains its measured score on the same region. Excluded candidates have eligible=false, filter_reason and rank=null. The highest eligible score wins (ties: category_id). Margins and top-k use eligible candidates only. FoundationPose registers the winner. Expected CAD labels are joined only after prediction.

ISM = (semantic + appearance + geometry * visible_ratio) / (2 + visible_ratio). Semantic: mean of the five highest cosine similarities across 42 views. Appearance: patch similarity on the best semantic view. Geometry: 2D bounding-box IoU, not a 6D fit. Visible ratio: fraction of compatible reference patches. Semantic/appearance are clipped to [0,1]; IoU and visible ratio are in [0,1]. These are not probabilities. FoundationPose exposes a pose-ranking network score without probability calibration or a scale comparable to ISM.

Pose: T_camera_from_model_mm and T_camera_from_input_cad_mm; OpenCV camera coordinates: X right, Y down, Z forward; translation in mm. FP receives meshes and depth in metres and converts output to mm. T_camera_from_input = T_camera_from_model @ inv(T_local_from_sam_mm). No pose ground truth or symmetry annotations are available; pose accuracy is not measured.

Cache: only CAD references with verified signatures/hashes; hits and keys are in run.json. SAM, ranking and pose run afresh. Seed 0; sampling by category_id. The first image includes cold-start costs. Other GPU processes can affect latency. Exact CUDA numerical reproducibility is not guaranteed.

Historical detailed suite: scores were maxima among assigned detections, with gaps for unassigned classes, rather than a full ranking on one mask. It used PEM and a different IoU/protocol. No causal improvement or directly comparable latency is claimed.

Artifacts: scores_wide.csv/json (full table), scores_long.csv/json (image × CAD), run.json (poses, timings, rankings and provenance), summary.json, confusion_matrix.csv, per_cad.csv, latency.csv, cases/<frame_id>/mask_overlay.png, pose_overlay.png and cad_mesh_overlay.png.

## Metrics

```json
{
  "total_manifest": 45,
  "valid_manifest": 45,
  "selected": 45,
  "processed_identification": 45,
  "complete_with_pose": 45,
  "failed_identification": 0,
  "failed_pose": 0,
  "invalid": 0,
  "omitted": 0,
  "matches": 36,
  "top1_on_predictions": 0.8,
  "top1_on_selected": 0.8,
  "failure_reasons": [],
  "topk": {
    "1": {
      "matches": 36,
      "denominator_predictions": 45,
      "accuracy": 0.8
    },
    "3": {
      "matches": 40,
      "denominator_predictions": 45,
      "accuracy": 0.8888888888888888
    },
    "5": {
      "matches": 41,
      "denominator_predictions": 45,
      "accuracy": 0.9111111111111111
    }
  },
  "latency": {
    "appearance_s": {
      "count": 45,
      "mean": 0.00019809766155150203,
      "median": 0.0001967529533430934,
      "p95": 0.0002120202174410224,
      "sum": 0.00891439476981759
    },
    "assignment_s": {
      "count": 45,
      "mean": 0.00016785610932856798,
      "median": 0.00016069202683866024,
      "p95": 0.00020636820700019597,
      "sum": 0.007553524919785559
    },
    "comparison_export_s": {
      "count": 45,
      "mean": 7.232359558757808e-05,
      "median": 6.999308243393898e-05,
      "p95": 9.523318149149417e-05,
      "sum": 0.003254561801441014
    },
    "dino_s": {
      "count": 45,
      "mean": 0.08376830050773505,
      "median": 0.08186784200370312,
      "p95": 0.12274130312725902,
      "sum": 3.769573522848077
    },
    "foundationpose_pipeline_s": {
      "count": 45,
      "mean": 1.0458713037763825,
      "median": 0.9744714320404455,
      "p95": 1.4970755449030542,
      "sum": 47.064208669937216
    },
    "geometry_s": {
      "count": 45,
      "mean": 0.006303079224502047,
      "median": 0.00023945595603436232,
      "p95": 0.0002965362044051288,
      "sum": 0.2836385651025921
    },
    "input_s": {
      "count": 45,
      "mean": 0.008380830225845178,
      "median": 0.008669941918924451,
      "p95": 0.009223042754456401,
      "sum": 0.377137360163033
    },
    "postprocess_s": {
      "count": 45,
      "mean": 1.2247596924529514,
      "median": 1.2336627850309014,
      "p95": 1.8735126741928976,
      "sum": 55.114186160382815
    },
    "processing_s": {
      "count": 45,
      "mean": 2.3612155732595257,
      "median": 2.366990206995979,
      "p95": 3.0316523247864096,
      "sum": 106.25470079667866
    },
    "projection_s": {
      "count": 45,
      "mean": 0.0012013777366114988,
      "median": 0.0011680269381031394,
      "p95": 0.0014268919359892605,
      "sum": 0.05406199814751744
    },
    "proposal_filter_s": {
      "count": 45,
      "mean": 0.0005400065099820494,
      "median": 0.0005290090339258313,
      "p95": 0.0006435254123061896,
      "sum": 0.024300292949192226
    },
    "ranking_all_cads_s": {
      "count": 45,
      "mean": 0.0030154910197274553,
      "median": 0.002987446030601859,
      "p95": 0.0032278437865898013,
      "sum": 0.13569709588773549
    },
    "ranking_query_features_s": {
      "count": 45,
      "mean": 0.010821597531644835,
      "median": 0.010755933006294072,
      "p95": 0.010960633982904255,
      "sum": 0.48697188892401755
    },
    "sam_s": {
      "count": 45,
      "mean": 1.0340130026374632,
      "median": 1.0344708079937845,
      "p95": 1.0493206548504532,
      "sum": 46.53058511868585
    },
    "semantic_s": {
      "count": 45,
      "mean": 0.001647859728998608,
      "median": 0.00156739994417876,
      "p95": 0.0016686962451785802,
      "sum": 0.07415368780493736
    },
    "total_s": {
      "count": 45,
      "mean": 3.5523598274112578,
      "median": 3.4770580830518156,
      "p95": 4.412149821571075,
      "sum": 159.8561922335066
    }
  },
  "pose_accuracy_measured": false,
  "pose_accuracy_limitation": "No pose ground truth or symmetry annotations in manifest; registration success is not accuracy.",
  "wall_s": 175.45636599394493,
  "semantic_to_final_changed": 6,
  "assignment_to_final_changed": 6,
  "semantic_correct": 38,
  "per_cad": [
    {
      "cad_key": "00_rear_mounting_plate",
      "support": 5,
      "predictions": 5,
      "correct": 4,
      "recall_on_selected": 0.8,
      "precision": 0.8
    },
    {
      "cad_key": "01_front_housing",
      "support": 5,
      "predictions": 4,
      "correct": 4,
      "recall_on_selected": 0.8,
      "precision": 1.0
    },
    {
      "cad_key": "02_control_panel",
      "support": 5,
      "predictions": 5,
      "correct": 5,
      "recall_on_selected": 1.0,
      "precision": 1.0
    },
    {
      "cad_key": "03_option_cassette",
      "support": 5,
      "predictions": 3,
      "correct": 3,
      "recall_on_selected": 0.6,
      "precision": 1.0
    },
    {
      "cad_key": "03a_vertical_option_a",
      "support": 5,
      "predictions": 4,
      "correct": 2,
      "recall_on_selected": 0.4,
      "precision": 0.5
    },
    {
      "cad_key": "03b_vertical_option_b",
      "support": 0,
      "predictions": 3,
      "correct": 0,
      "recall_on_selected": null,
      "precision": 0.0
    },
    {
      "cad_key": "04_io_terminals",
      "support": 5,
      "predictions": 3,
      "correct": 3,
      "recall_on_selected": 0.6,
      "precision": 1.0
    },
    {
      "cad_key": "05_control_board",
      "support": 5,
      "predictions": 7,
      "correct": 5,
      "recall_on_selected": 1.0,
      "precision": 0.7142857142857143
    },
    {
      "cad_key": "06_power_unit",
      "support": 5,
      "predictions": 6,
      "correct": 5,
      "recall_on_selected": 1.0,
      "precision": 0.8333333333333334
    },
    {
      "cad_key": "07_thermal_core",
      "support": 5,
      "predictions": 5,
      "correct": 5,
      "recall_on_selected": 1.0,
      "precision": 1.0
    }
  ],
  "confused_pairs": [
    {
      "expected": "00_rear_mounting_plate",
      "predicted": "03a_vertical_option_a",
      "count": 1
    },
    {
      "expected": "01_front_housing",
      "predicted": "03b_vertical_option_b",
      "count": 1
    },
    {
      "expected": "03_option_cassette",
      "predicted": "03a_vertical_option_a",
      "count": 1
    },
    {
      "expected": "03_option_cassette",
      "predicted": "05_control_board",
      "count": 1
    },
    {
      "expected": "03a_vertical_option_a",
      "predicted": "00_rear_mounting_plate",
      "count": 1
    },
    {
      "expected": "03a_vertical_option_a",
      "predicted": "03b_vertical_option_b",
      "count": 1
    },
    {
      "expected": "03a_vertical_option_a",
      "predicted": "05_control_board",
      "count": 1
    },
    {
      "expected": "04_io_terminals",
      "predicted": "03b_vertical_option_b",
      "count": 1
    },
    {
      "expected": "04_io_terminals",
      "predicted": "06_power_unit",
      "count": 1
    }
  ],
  "wrong_high_scores": [
    {
      "name": "sensor_side",
      "expected": "03a_vertical_option_a",
      "predicted": "00_rear_mounting_plate",
      "score": 0.4925619661808014,
      "margin": 0.0340552031993866
    },
    {
      "name": "sensor_rot_side",
      "expected": "03a_vertical_option_a",
      "predicted": "05_control_board",
      "score": 0.46993914246559143,
      "margin": 0.001695185899734497
    },
    {
      "name": "cassette_side",
      "expected": "03_option_cassette",
      "predicted": "03a_vertical_option_a",
      "score": 0.46315574645996094,
      "margin": 0.07523703575134277
    },
    {
      "name": "io_far",
      "expected": "04_io_terminals",
      "predicted": "03b_vertical_option_b",
      "score": 0.45131194591522217,
      "margin": 0.03617125749588013
    },
    {
      "name": "cassette_far",
      "expected": "03_option_cassette",
      "predicted": "05_control_board",
      "score": 0.45011407136917114,
      "margin": 0.025578737258911133
    },
    {
      "name": "io_rot_far",
      "expected": "04_io_terminals",
      "predicted": "06_power_unit",
      "score": 0.41911253333091736,
      "margin": 0.013402849435806274
    },
    {
      "name": "front_rot_side",
      "expected": "01_front_housing",
      "predicted": "03b_vertical_option_b",
      "score": 0.4028916656970978,
      "margin": 0.004908144474029541
    },
    {
      "name": "sensor_rot_far",
      "expected": "03a_vertical_option_a",
      "predicted": "03b_vertical_option_b",
      "score": 0.3683335781097412,
      "margin": 0.003152608871459961
    },
    {
      "name": "rear_rot_side",
      "expected": "00_rear_mounting_plate",
      "predicted": "03a_vertical_option_a",
      "score": 0.3323538899421692,
      "margin": 0.01564854383468628
    }
  ],
  "margins_correct": {
    "count": 36,
    "mean": 0.10195928894811207,
    "median": 0.10763539373874664,
    "p95": 0.22035955637693405,
    "sum": 3.6705344021320343
  },
  "margins_wrong": {
    "count": 9,
    "mean": 0.023316618469026353,
    "median": 0.01564854383468628,
    "p95": 0.0596107244491577,
    "sum": 0.20984956622123718
  }
}
```

## Reproduction

```bash
/home/raca/projects/.venvs/assembly-runtime/bin/python /home/raca/projects/CADTracking/pipeline/tools/evaluate_identification.py --corner-filter --output results/evaluations/sam6d_fp_corner_full_20261111
```
