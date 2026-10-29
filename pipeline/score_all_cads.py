"""Compute unfiltered SAM-6D-formula hypotheses for every CAD on one saved region per image."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ISM = Path(__file__).resolve().parents[2] / "SAM6D/SAM-6D/SAM-6D/Instance_Segmentation_Model"
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
    final = (semantic[0] + appearance + geometry * visible) / (2 + visible)
    if not torch.isfinite(final).all():
        raise ValueError("Una hipótesis produjo un score no finito; no se sustituirá por cero.")
    return {"score": final, "semantic": semantic[0], "appearance": appearance,
            "geometry": geometry, "visible_ratio": visible}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reference-cache", type=Path, default=Path("/dev/shm/pddl-assembly-runtime/reference_features"))
    args = parser.parse_args()
    source = args.run.resolve()
    original = json.loads((source / "run.json").read_text())
    manifest_path = Path(original["manifest"])
    if not manifest_path.is_file():
        manifest_path = Path(__file__).parent / "prepared/manifest_at_execution.json"
    if hash_files([manifest_path]) != original["manifest_sha256"]:
        raise ValueError("El manifest cambió desde la inferencia original.")
    manifest = json.loads(manifest_path.read_text())
    by_name = {f["name"]: f for f in manifest["frames"]}
    corrections = json.loads(Path(__file__).with_name("cad_name_corrections.json").read_text())["names"]
    target = source / "all_cad_primary_region.json"
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
              "setup_timing": {}, "frames": []}
    write_json(target, report)
    try:
        with measured(report["setup_timing"], "model_load_s", device):
            with initialize_config_dir(version_base=None, config_dir=str(ISM / "configs")):
                cfg = compose(config_name="run_inference.yaml")
            # Reuse saved masks; no need to load or rerun SAM.
            model = instantiate(cfg.model, segmentor_model=None)
            model.eval()
            model.descriptor_model.model.to(device)
            model.descriptor_model.model.device = device
        with measured(report["setup_timing"], "references_s", device):
            report["references"] = load_references(model, manifest["parts"], manifest_path.parent, device, args.reference_cache)
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
            with Image.open(manifest_path.parent / frame_meta["depth"]) as source_depth:
                depth = torch.from_numpy(np.array(source_depth).astype(np.int32))[None].to(device)
            camera = json.loads((manifest_path.parent / frame_meta["camera"]).read_text())
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
            scores = [{"part_key": part["key"], "part_name": corrections.get(part["key"], part["name"]),
                       **{key: value[i] for key, value in cpu_values.items()}}
                      for i, part in enumerate(manifest["parts"])]
            original_hypothesis = next(s["score"] for s in scores if s["part_key"] == candidate["part_key"])
            delta = abs(original_hypothesis - candidate["score"])
            if delta > 0.001:
                raise ValueError(f"El score de la clase original cambió {delta:.6f}; revisar referencias antes de publicar.")
            rgb.crop((int(x), int(y), int(x + w), int(y + h))).save(folder / "primary_region.png")
            timing["total_s"] = time.perf_counter() - image_start
            report["frames"].append({"name": frame["name"], "status": "scored", "bbox": candidate["bbox"],
                                     "original_prediction": candidate["part_key"], "original_score": candidate["score"],
                                     "original_recognized": frame["recognized"], "original_score_delta": delta,
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
