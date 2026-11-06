"""Compute unfiltered SAM-6D-formula hypotheses for every CAD on one saved region per image."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from settings import ISM, PREPARED, path, display_names, run_dataset
sys.path.insert(0, str(ISM))
from model.utils import Detections  # noqa: E402
from run_suite_pddl import choose_device, compare_all, hash_files, load_references, measured, write_json  # noqa: E402


def score_hypotheses(model, detections, patches, batch, semantic, views):
    """No class assignment or confidence cutoff: same mask, all CADs, original score formula."""
    count = len(model.ref_data["descriptors"])
    objects = torch.arange(count, device=patches.device)
    repeated = Detections({"boxes": detections.boxes.repeat(count, 1),
                           "masks": detections.masks.repeat(count, 1, 1)})
    repeated_patches = patches.repeat(count, 1, 1)
    appearance, reference = model.compute_appearance_score(views[0], objects, repeated_patches)
    uv = model.project_template_to_image(views[0], objects, batch, repeated.masks)
    geometry, visible = model.compute_geometric_score(uv, repeated, repeated_patches, reference, model.visible_thred)
    # Upstream compute_iou returns scalar 0.0 when its batch overlap test fails.
    geometry = torch.as_tensor(geometry, device=appearance.device).expand_as(appearance)
    final = (semantic[0] + appearance + geometry * visible) / (2 + visible)
    if not torch.isfinite(final).all():
        raise ValueError("Una hipótesis produjo un score no finito; no se sustituirá por cero.")
    return {"score": final, "semantic": semantic[0], "appearance": appearance,
            "geometry": geometry, "visible_ratio": visible}


def score_view_details(model, query, patches, detections, batch, views):
    """Additional audit on the primary mask; the original ISM uses semantic winners only."""
    records = []
    for obj in range(len(model.ref_data["descriptors"])):
        raw = model.matching_config.metric(query, model.ref_data["descriptors"][obj:obj + 1])[0, 0]
        appearance = []
        for view in range(len(raw)):
            a, _ = model.compute_appearance_score(
                torch.tensor([view], device=patches.device),
                torch.tensor([obj], device=patches.device), patches)
            appearance.append(float(a[0]))
        records.append({"semantic_by_view": raw.cpu().tolist(), "appearance_by_view": appearance,
                        "semantic_winner": int(views[0, obj]),
                        "appearance_winner": int(np.argmax(appearance))})
    objects = torch.arange(len(records), device=patches.device)
    masks = detections.masks.repeat(len(records), 1, 1)
    uv = model.project_template_to_image(views[0], objects, batch, masks)
    translations = model.Calculate_the_query_translation(masks, batch["depth"][0],
                                                         batch["cam_intrinsic"][0], batch["depth_scale"])
    for i, record in enumerate(records):
        projected = torch.cat((uv[i].amin(0), uv[i].amax(0))).cpu().tolist()
        box = detections.boxes[0].cpu().tolist()
        intersection = max(0, min(projected[2], box[2]) - max(projected[0], box[0])) * max(0, min(projected[3], box[3]) - max(projected[1], box[1]))
        union = (projected[2]-projected[0])*(projected[3]-projected[1]) + (box[2]-box[0])*(box[3]-box[1]) - intersection
        record["geometry_details"] = {"projected_box_xyxy_px": projected, "proposal_box_xyxy_px": box,
            "intersection_px2": intersection, "union_px2": union,
            "translation_m": translations[i].cpu().tolist(), "point_count": uv.shape[1],
            "visible_threshold": float(model.visible_thred)}
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-score-drift", action="store_true", help="For view audits only: keep measured recomputations with explicit original-score deltas")
    parser.add_argument("--view-details", action="store_true", help="Audit every reference view on the saved primary mask")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reference-cache", type=Path, default=path("reference_cache"))
    args = parser.parse_args()
    if args.allow_score_drift and not args.view_details:
        parser.error("--allow-score-drift requiere --view-details")
    source = args.run.resolve()
    original = json.loads((source / "run.json").read_text())
    manifest_path, manifest_base = run_dataset(original, source)
    if hash_files([manifest_path]) != original["manifest_sha256"]:
        raise ValueError("El manifest cambió desde la inferencia original.")
    manifest = json.loads(manifest_path.read_text())
    by_name = {f["name"]: f for f in manifest["frames"]}
    corrections = display_names()
    target = source / ("ism_view_details.json" if args.view_details else "all_cad_primary_region.json")
    if target.exists():
        raise ValueError("El diagnóstico ya existe; se preserva el resultado anterior.")
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    os.chdir(ISM)
    start = time.perf_counter()
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_run": str(source), "source_run_sha256": hash_files([source / "run.json"]),
              "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "scope": "All CAD hypotheses on the SAME saved primary region of each image; no semantic assignment or score cutoff.",
              "is_additional_diagnostic": True, "original_predictions_modified": False,
              "score_is_probability": False, "score_formula": "(semantic + appearance + geometry * visible_ratio) / (2 + visible_ratio)",
              "allow_score_drift": args.allow_score_drift, "torch": str(torch.__version__), "aggregation": None, "setup_timing": {}, "frames": []}
    write_json(target, report)
    try:
        with measured(report["setup_timing"], "model_load_s", device):
            with initialize_config_dir(version_base=None, config_dir=str(ISM / "configs")):
                cfg = compose(config_name="run_inference.yaml")
            # Reuse saved masks; no need to load or rerun SAM.
            model = instantiate(cfg.model, segmentor_model=None)
            model.eval()
            report["aggregation"] = model.matching_config.aggregation_function
            model.descriptor_model.model.to(device)
            model.descriptor_model.model.device = device
        with measured(report["setup_timing"], "references_s", device):
            report["references"] = load_references(model, manifest["parts"], manifest_base, device, args.reference_cache)
        for number, frame in enumerate(original["frames"], 1):
            print(f"[{number}/{len(original['frames'])}] Scores de los diez CAD: {frame['name']}", flush=True)
            folder = source / "cases" / frame["name"]
            saved = json.loads((folder / "detection_ism.json").read_text())
            candidate = frame["candidate"]
            if not candidate:
                report["frames"].append({"name": frame["name"], "status": "no_saved_region", "scores": []})
                continue
            if not saved or saved[0]["part_key"] != candidate["part_key"] or saved[0]["score"] != candidate["score"]:
                raise ValueError("La región primaria guardada no coincide con la predicción original.")
            image_start = time.perf_counter()
            with Image.open(folder / "rgb.png") as source_rgb:
                rgb = source_rgb.convert("RGB")
            with Image.open(folder / "mask_000.png") as source_mask:
                mask = torch.from_numpy((np.array(source_mask) > 0).astype(np.float32))[None].to(device)
            frame_meta = by_name[frame["name"]]
            with Image.open(manifest_base / frame_meta["depth"]) as source_depth:
                depth = torch.from_numpy(np.array(source_depth).astype(np.int32))[None].to(device)
            camera = json.loads((manifest_base / frame_meta["camera"]).read_text())
            batch = {"depth": depth,
                     "cam_intrinsic": torch.tensor(camera["cam_K"], dtype=torch.float32, device=device).reshape(1, 3, 3),
                     "depth_scale": torch.tensor([camera["depth_scale"]], device=device)}
            x, y, w, h = candidate["bbox"]
            detections = Detections({"boxes": torch.tensor([[x, y, x + w, y + h]], device=device), "masks": mask})
            timing = {}
            with measured(timing, "query_features_s", device):
                query, patches = model.descriptor_model(np.array(rgb), detections)
            with measured(timing, "all_cad_scores_s", device):
                semantic, views, _ = compare_all(model, query, device)
                values = score_hypotheses(model, detections, patches, batch, semantic, views)
                cpu_values = {key: tensor.cpu().tolist() for key, tensor in values.items()}
                details = score_view_details(model, query, patches, detections, batch, views) if args.view_details else None
            scores = [{"part_key": part["key"], "part_name": corrections.get(part["key"], part["name"]),
                       **{key: value[i] for key, value in cpu_values.items()},
                       **(details[i] if details else {})}
                      for i, part in enumerate(manifest["parts"])]
            if details:
                for item in scores:
                    geometry_image = rgb.copy()
                    draw = ImageDraw.Draw(geometry_image)
                    g = item['geometry_details']
                    draw.rectangle(g['proposal_box_xyxy_px'], outline='#ff9a24', width=3)
                    draw.rectangle(g['projected_box_xyxy_px'], outline='#1689ed', width=3)
                    relative = Path('geometry_audit') / (item['part_key'] + '.png')
                    (folder / relative).parent.mkdir(parents=True, exist_ok=True)
                    geometry_image.save(folder / relative)
                    g['overlay'] = relative.as_posix()
            original_hypothesis = next(s["score"] for s in scores if s["part_key"] == candidate["part_key"])
            delta = abs(original_hypothesis - candidate["score"])
            if delta > 0.001 and not args.allow_score_drift:
                raise ValueError(f"El score de la clase original cambió {delta:.6f}; revisar referencias antes de publicar.")
            rgb.crop((int(x), int(y), int(x + w), int(y + h))).save(folder / "primary_region.png")
            timing["total_s"] = time.perf_counter() - image_start
            report["frames"].append({"name": frame["name"], "status": "scored", "bbox": candidate["bbox"],
                                     "original_prediction": candidate["part_key"], "original_score": candidate["score"],
                                     "original_recognized": frame["recognized"], "original_score_delta": delta,
                                     "original_score_matches": delta <= .001,
                                     "source_mask_sha256": hash_files([folder / "mask_000.png"]),
                                     "timing": timing, "scores": scores})
            write_json(target, report)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "error"
        report["error"] = str(error)
        raise
    finally:
        report["wall_s"] = time.perf_counter() - start
        write_json(target, report)


if __name__ == "__main__":
    main()
