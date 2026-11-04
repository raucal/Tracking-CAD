"""Resident single-job SAM-6D engine; RGB-only and full RGB-D scores remain distinct."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from settings import ROOT, PIPELINE, ISM, NATIVE, PREPARED, CONFIG, path, display_names
sys.path.insert(0, str(ISM))
from model.utils import Detections, mask_to_rle  # noqa: E402
from run_inference_pddl import visualize  # noqa: E402
from segmentor_runtime import configure, move_segmentor
from run_suite_pddl import choose_device, hash_files, load_references, measured, process_frame  # noqa: E402

WEIGHTS = {"vit_h": "sam_vit_h_4b8939.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_b": "sam_vit_b_01ec64.pth"}


def environment():
    return {"cuda_available": torch.cuda.is_available(), "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "sam_weights": {name: (ISM / "checkpoints/segment-anything" / filename).is_file() for name, filename in WEIGHTS.items()},
            "dino_weights": (ISM / "checkpoints/dinov2/dinov2_vitl14_pretrain.pth").is_file(),
            "fastsam_weights": (ISM / "checkpoints/FastSAM/FastSAM-x.pt").is_file(),
            "pose_available": (ISM.parent / "Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth").is_file() and bool(list((NATIVE / "pointnet2").glob("*.so"))), "render_backend": "Blender 3.3.1 Cycles CPU; inference CUDA/CPU"}


def rgb_inference(model, rgb, part, device, min_score):
    """2D ablation: semantic and appearance only, never invent a depth map."""
    start = time.perf_counter()
    timings = {}
    with measured(timings, "sam_s", device):
        detections = Detections(model.segmentor_model.generate_masks(np.array(rgb)))
    results, masks = [], []
    if len(detections):
        detections.remove_very_small_detections(model.post_processing_config.mask_post_processing)
    proposals = len(detections)
    if len(detections):
        with measured(timings, "dino_matching_s", device):
            query, patches = model.descriptor_model(np.array(rgb), detections)
            selected, objects, semantic, views = model.compute_semantic_score(query)
            detections.filter(selected)
            patches = patches[selected]
            if len(detections):
                appearance, _ = model.compute_appearance_score(views, objects, patches)
                score = (semantic + appearance) / 2
                detections.add_attribute("scores", score)
                detections.add_attribute("object_ids", objects)
                detections.add_attribute("semantic_scores", semantic)
                detections.add_attribute("appearance_scores", appearance)
                detections.filter(torch.isfinite(score) & (score >= min_score))
                if len(detections):
                    detections.apply_nms_per_object_id(model.post_processing_config.nms_thresh)
                    detections.to_numpy()
                    for index in np.argsort(-detections.scores):
                        x1, y1, x2, y2 = detections.boxes[index].tolist()
                        mask = detections.masks[index] > 0.5
                        masks.append(mask)
                        results.append({"category_id": part["category_id"], "part_key": part["key"], "part_name": part["name"],
                                        "score": float(detections.scores[index]),
                                        "semantic_score": float(detections.semantic_scores[index]),
                                        "appearance_score": float(detections.appearance_scores[index]),
                                        "geometry_score": None, "visible_ratio": None,
                                        "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": mask_to_rle(mask)})
    timings["processing_s"] = time.perf_counter() - start
    return results, masks, timings, proposals


class Engine:
    def __init__(self, data_root):
        self.data_root = data_root
        self.model = None
        self.model_key = None
        self.reference_token = None
        self.reference_record = None
        self.manifest_path = PREPARED / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.names = display_names()
        self.reference_cache = path("ui_reference_cache")

    def prepare_cad(self, folder, inputs, update):
        raw = (folder / inputs["cad_filename"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if inputs["options"]["cad_units"] == "mm":
            for part in self.manifest["parts"]:
                if part["source_sha256"] == digest:
                    part = dict(part)
                    part["name"] = self.names.get(part["key"], part["name"])
                    update("Reutilizando las 42 vistas del CAD conocido")
                    return part, self.manifest_path.parent, "existing_blender_templates"
        key = hashlib.sha256(raw + inputs["options"]["cad_units"].encode() + b"blender_cycles_v2").hexdigest()
        base = self.data_root / "cache/cad"
        cache = base / key
        cache.mkdir(parents=True, exist_ok=True)
        mesh = cache / "object_mm.obj"
        if not mesh.is_file():
            mesh.write_bytes((folder / "object_mm.obj").read_bytes())
        templates = cache / "templates"
        metadata = templates / "render_metadata.json"
        ready = metadata.is_file() and all((templates / f"{prefix}_{i}.png").is_file() for i in range(42) for prefix in ("rgb", "mask")) and all((templates / f"xyz_{i}.npy").is_file() for i in range(42))
        if not ready:
            update("Generando las 42 vistas del CAD nuevo; este paso se guarda en caché")
            command = [sys.executable, str(Path(__file__).with_name("render_reference.py")), "--mesh", str(mesh),
                       "--output", str(templates), "--poses", str(ISM / "utils/poses/predefined_poses/cam_poses_level0.npy")]
            with (folder / "render.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=330)
        part = {"key": "uploaded_cad", "name": inputs["cad_name"], "category_id": 1,
                "sam_template_dir": str(Path(key) / "templates"), "sam_ply": str(Path(key) / "object_mm.obj")}
        return part, base, "blender_cycles_v2"

    def prepare_model(self, config, device, update, timing):
        key = (config['segmentor'],config['backbone'],str(device))
        with measured(timing, "model_load_s", device):
            reused_model = self.model_key == key
            if not reused_model:
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                update("Cargando SAM y DINOv2 en " + str(device))
                with initialize_config_dir(version_base=None, config_dir=str(ISM / "configs")):
                    cfg = compose(config_name="run_inference.yaml", overrides=["model=ISM_fastsam"] if config["segmentor"] == "fastsam" else [])
                configure(cfg, config["segmentor"], config["backbone"], config["stability"])
                self.model = instantiate(cfg.model)
                self.model.eval()
                self.model.descriptor_model.model.to(device)
                self.model.descriptor_model.model.device = device
                move_segmentor(self.model, device, config["segmentor"])
                self.model_key = key
        if not reused_model:
            self.reference_token=None
            self.reference_record=None
        return reused_model

    def warmup(self, segmentor='sam', backbone='vit_h', update=lambda message:None):
        from .inputs import DEFAULTS
        config={**DEFAULTS,'segmentor':segmentor,'backbone':backbone}
        if not environment()['cuda_available']:
            raise RuntimeError('No hay CUDA disponible para precargar los modelos.')
        timing={};os.chdir(ISM)
        self.prepare_model(config,choose_device('cuda'),update,timing)
        torch.cuda.synchronize()
        return timing

    def reference_key(self,part,base):
        paths=[ISM/'checkpoints/dinov2/dinov2_vitl14_pretrain.pth',ISM/'model/dinov2.py',ISM/'utils/bbox_utils.py',base/part['sam_ply']]
        folder=base/part['sam_template_dir']
        paths.extend(folder/f'{prefix}_{i}.png' for i in range(42) for prefix in ('rgb','mask'))
        return (self.model_key,part['key'],tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths))

    @torch.inference_mode()
    def run(self, folder, inputs, update, save_outputs=True):
        started = time.perf_counter()
        config = inputs["options"]
        if not save_outputs and config["pose"]:raise ValueError("La detección en segundo plano no puede registrar poses.")
        device = choose_device(config["device"])
        torch.set_num_threads(16)
        key = (config["segmentor"], config["backbone"], str(device))
        weights = environment()
        selected = ISM / "checkpoints" / ("segment-anything/" + WEIGHTS[config["backbone"]] if config["segmentor"] == "sam" else "FastSAM/FastSAM-x.pt")
        required = [selected, ISM / "checkpoints/dinov2/dinov2_vitl14_pretrain.pth"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            temporary = any(path.is_symlink() and str(path.resolve()).startswith("/dev/shm/") for path in missing)
            detail = " Los enlaces apuntan a /dev/shm, que es temporal y se vacía al reiniciar el equipo." if temporary else ""
            raise ValueError("Faltan checkpoints: " + ", ".join(path.name for path in missing) + "." + detail + " Consulta la sección de recuperación de pesos del README de la interfaz.")
        timing = {}
        os.chdir(ISM)
        reused_model = self.prepare_model(config,device,update,timing)
        model = self.model
        if config["segmentor"] == "sam":
            model.segmentor_model.stability_score_thresh = config["stability"]
            model.segmentor_model.pred_iou_thresh = config["pred_iou"]
            model.segmentor_model.points_per_batch = config["points_per_batch"]
        else:
            model.segmentor_model.conf_threshold = config["fastsam_conf"]
            model.segmentor_model.iou_threshold = config["fastsam_iou"]
        model.post_processing_config.nms_thresh = config["nms"]
        model.matching_config.aggregation_function = config["aggregation"]
        with measured(timing, "cad_prepare_s", device):
            if inputs.get('auto_assembly'):
                from .assembly_identification import assembly_catalog
                manifest, parts = assembly_catalog(inputs['auto_assembly'])
                base = manifest.parent
                parts = [dict(p, name=self.names.get(p['key'], p['name'])) for p in parts]
                part = parts[0]
                renderer = 'existing_blender_templates'
            else:
                part, base, renderer = self.prepare_cad(folder, inputs, update)
                parts = [part]
        with measured(timing, "reference_features_s", device):
            update("Preparando las referencias DINOv2 del CAD")
            reference_token=tuple(self.reference_key(p,base) for p in parts)
            if reference_token==self.reference_token and self.reference_record is not None:
                references={**self.reference_record,'parts':[{**record,'cache_hit':True,'memory_hit':True} for record in self.reference_record['parts']]}
            else:
                references = load_references(model, parts, base, device, self.reference_cache)
                self.reference_token=reference_token;self.reference_record=references
        update("Detectando y segmentando el objeto en la imagen")
        with Image.open(folder / "input_rgb.png") as source:
            rgb = source.convert("RGB")
        if config["mode"] == "rgbd":
            frame = {"name": "input", "frame_id": 0, "rgb": "input_rgb.png", "depth": "depth.png", "camera": "camera.json"}
            case, rgb, detections, masks, accepted = process_frame(model, frame, parts, folder, device, config["min_score"], config["confidence"])
            inference_timing = case["timing"]
            proposals = case["proposal_counts"]["valid_depth_and_size"]
            formula = "(semantic + appearance + geometry * visible_ratio) / (2 + visible_ratio)"
        else:
            detections, masks, inference_timing, proposals = rgb_inference(model, rgb, part, device, config["min_score"])
            accepted = [i for i, d in enumerate(detections) if d["score"] >= config["confidence"]]
            formula = "(semantic + appearance) / 2; sin geometría ni profundidad"
        identification = None
        if inputs.get('auto_assembly'):
            # process_frame sorts by score. Keep one CAD for registration and retain
            # the complete set of candidates for inspection in result.json.
            candidates = [{k: v for k, v in d.items() if k != 'segmentation'} for d in detections]
            winner = detections[accepted[0]]['part_key'] if accepted else None
            identification = {'assembly': inputs['auto_assembly'], 'part_key': winner,
                              'candidates': candidates, 'comparisons': case['comparisons']}
            if winner is not None:
                part = next(p for p in parts if p['key'] == winner)
                keep = [i for i, d in enumerate(detections) if d['part_key'] == winner]
                detections, masks = [detections[i] for i in keep], [masks[i] for i in keep]
                accepted = [i for i, d in enumerate(detections) if d['score'] >= config['confidence']]
        if save_outputs:
            update("Exportando máscaras y resultados")
            visualize(rgb, [detections[i] for i in accepted], [masks[i] for i in accepted], folder / "overlay.png")
            for i, mask in enumerate(masks):
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"mask_{i:03d}.png")
            (folder / "detections.json").write_text(json.dumps(detections, ensure_ascii=False, indent=2, allow_nan=False))
        pose_result = None
        model_from_input = np.eye(4)
        if "T_local_from_sam_mm" in part:
            model_from_input = np.linalg.inv(np.array(part["T_local_from_sam_mm"]))
        else:
            model_from_input[:3, 3] = -np.array(inputs["input_center_mm"])
        if inputs.get('auto_assembly') and 'T_local_from_sam_mm' not in part:
            raise ValueError('El CAD preparado necesita T_local_from_sam_mm para identificar su pose automáticamente.')
        if config["pose"]:
            if device.type != "cuda":
                raise ValueError('La pose 6D requiere CUDA.')
            if config['pose_backend'] == 'foundationpose':
                if accepted:
                    from .backends.foundationpose import run as foundationpose_run
                    update('FoundationPose: registrando el CAD' + (' y siguiendo la grabación' if inputs.get('recording_path') else ''))
                    with measured(timing, 'pose_pipeline_s', device):
                        pose_result = foundationpose_run(folder, base / part['sam_ply'], accepted[0],
                                                         model_from_input, inputs.get('recording_path'))
                    for pose in pose_result.get('poses', []):
                        pose['T_camera_from_input_cad_mm'] = (np.array(pose['T_camera_from_model_mm']) @ model_from_input).tolist()
                    (folder / 'pose.json').write_text(json.dumps(pose_result, indent=2, allow_nan=False))
                else:
                    pose_result = {'status': 'no_accepted_detection', 'backend': 'foundationpose', 'poses': []}
            elif not weights["pose_available"]:
                raise ValueError("PEM requiere CUDA, su checkpoint y la extensión PointNet2 compilada.")
            elif accepted:
                update("Estimando posición y orientación 6D con PEM")
                pose_input = folder / "pose_detections.json"
                pose_input.write_text(json.dumps([detections[i] for i in accepted]))
                pose_output = folder / "pose.json"
                transform_path=folder/'model_from_input_mm.json'
                transform_path.write_text(json.dumps(model_from_input.tolist()))
                command = [sys.executable, str(PIPELINE / "workers/pose_worker.py"), "--model-from-input",str(transform_path),"--rgb", str(folder / "input_rgb.png"),
                    "--depth", str(folder / "depth.png"), "--camera", str(folder / "camera.json"),
                    "--cad", str(base / part["sam_ply"]), "--templates", str(base / part["sam_template_dir"]),
                    "--detections", str(pose_input), "--output", str(pose_output), "--min-score", str(max(0, config["confidence"] - 1e-7))]
                with measured(timing, "pose_pipeline_s", device), (folder / "pose.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                pose_result = json.loads(pose_output.read_text())
                for pose in pose_result.get("poses", []):
                    pose["T_camera_from_input_cad_mm"] = (np.array(pose["T_camera_from_model_mm"]) @ model_from_input).tolist()
                pose_output.write_text(json.dumps(pose_result, indent=2, allow_nan=False))
            else:
                pose_result = {"status": "no_accepted_detection", "poses": []}
        result = {"status": "complete", "mode": config["mode"], "backend": config['pose_backend'] if config['pose'] else 'sam6d', "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "cad_name": part['name'] if identification else inputs["cad_name"], "cad_key": part["key"] if not identification or identification['part_key'] else None, "display_name": part["name"] if not identification or identification['part_key'] else 'Sin identificación fiable',
                  "recognized": bool(accepted), "accepted_detections": len(accepted), "valid_proposals": proposals,
                  "best_score": detections[0]["score"] if detections else None,
                  "formula": formula, "score_is_probability": False, "pose_6d_estimated": bool(pose_result and pose_result.get("poses")), "pose": pose_result,
                  "timing": {**timing, "inference_s": inference_timing["processing_s"], "total_s": time.perf_counter() - started},
                  "inference_stages": inference_timing, "model_reused": reused_model,
                  "reference_cache_hit": references["parts"][0]["cache_hit"], "renderer": renderer,
                  "options": config, "detections": [{k: v for k, v in d.items() if k != "segmentation"} for d in detections]}
        if identification is not None:
            result['identification'] = identification
            result['tracking_model'] = {'mesh': str((base / part['sam_ply']).resolve()),
                                        'model_from_input': model_from_input.tolist()}
        if save_outputs:(folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:result["_masks"]=masks
        return result
