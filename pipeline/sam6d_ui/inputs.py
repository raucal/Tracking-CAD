"""Validate local UI input before scheduling any GPU work."""
import base64
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

DEFAULTS = {"mode": "rgb", "device": "cuda", "backbone": "vit_h", "cad_units": "mm",
            "segmentor": "sam", "pose": False, "pose_backend": "pem", "fastsam_conf": 0.05, "fastsam_iou": 0.9,
            "min_score": 0.30, "confidence": 0.50, "stability": 0.97, "pred_iou": 0.88,
            "points_per_batch": 64, "nms": 0.25, "aggregation": "avg_5"}


def options(value):
    if not isinstance(value, dict):
        raise ValueError('Las opciones deben ser un objeto JSON.')
    result = {**DEFAULTS, **{k: v for k, v in value.items() if k in DEFAULTS}}
    for key, allowed in {"mode": ("rgb", "rgbd"), "device": ("auto", "cpu", "cuda"),
                         "segmentor": ("sam", "fastsam"),
                         "pose_backend": ("pem", "foundationpose"),
                         "backbone": ("vit_h", "vit_l", "vit_b"), "cad_units": ("mm", "cm", "m"),
                         "aggregation": ("avg_5", "mean", "median", "max")}.items():
        if result[key] not in allowed:
            raise ValueError(f"Opción {key} no válida.")
    for key in ("min_score", "confidence", "stability", "pred_iou", "nms", "fastsam_conf", "fastsam_iou"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} debe estar entre 0 y 1.")
    if result["min_score"] > result["confidence"]:
        raise ValueError("El umbral de conservación no puede superar al de aceptación.")
    if not isinstance(result["pose"], bool):
        raise ValueError("La opción pose debe ser true o false.")
    if result['pose_backend'] == 'foundationpose':
        result['pose'] = True
    if result["pose"] and (result["mode"] != "rgbd" or result["device"] == "cpu"):
        raise ValueError("La pose 6D requiere modo RGB-D y GPU CUDA.")
    result["points_per_batch"] = int(result["points_per_batch"])
    if not 1 <= result["points_per_batch"] <= 128:
        raise ValueError("Los puntos por lote deben estar entre 1 y 128.")
    return result


def uploaded(item, extensions, maximum):
    if not isinstance(item, dict) or not item.get("data"):
        raise ValueError("Falta seleccionar un archivo.")
    name = str(item.get("name", "")).replace("\\", "/").split("/")[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in extensions:
        raise ValueError(f"Formato no admitido: {suffix}. Se admite: {', '.join(extensions)}.")
    try:
        content = base64.b64decode(item["data"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Archivo codificado incorrectamente.") from error
    if not content or len(content) > maximum:
        raise ValueError("Archivo vacío o demasiado grande.")
    return name, content


def read_rgb(content):
    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > 4_000_000:
            raise ValueError("La imagen supera 4 megapíxeles. Redimensiona RGB y profundidad de forma coherente.")
        return image.convert("RGB")


def read_depth(content, suffix, size):
    if suffix == ".npy":
        depth = np.load(BytesIO(content), allow_pickle=False)
    else:
        with Image.open(BytesIO(content)) as source:
            depth = np.array(source)
    if depth.ndim != 2 or depth.shape != (size[1], size[0]):
        raise ValueError("La profundidad debe ser una matriz alineada y del mismo tamaño que el RGB.")
    if depth.dtype != np.uint16:
        raise ValueError("Se requiere profundidad raw uint16 (.npy o PNG de 16 bits), no una imagen coloreada.")
    if not np.any(depth > 0):
        raise ValueError("La profundidad no contiene valores válidos.")
    return depth


def read_camera(content, depth, size):
    camera = json.loads(content)
    if "cam_K" not in camera:
        from prepare import camera_from_metadata
        return camera_from_metadata(camera, depth, size)
    matrix = np.array(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    scale = float(camera["depth_scale"])
    if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("Intrínsecos de cámara no válidos.")
    if not np.allclose(matrix[2], [0, 0, 1]) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Matriz de cámara o escala de profundidad no válida.")
    if np.any(np.abs(camera.get("dist_coeffs", [])) > 1e-10):
        raise ValueError("La imagen RGB-D debe estar rectificada, sin distorsión residual.")
    return {"cam_K": matrix.reshape(-1).tolist(), "depth_scale": scale}


def cad_mesh(content, suffix, units):
    mesh = trimesh.load(BytesIO(content), file_type=suffix.lstrip("."), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("El CAD no contiene una malla triangular válida.")
    if len(mesh.faces) > 250_000:
        raise ValueError("El CAD supera 250.000 caras. Exporta una malla simplificada.")
    if not np.isfinite(mesh.vertices).all() or np.max(mesh.extents) <= 0:
        raise ValueError("El CAD contiene coordenadas o dimensiones no válidas.")
    mesh.apply_scale({"mm": 1, "cm": 10, "m": 1000}[units])
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    mesh.metadata['input_center_mm'] = center.tolist()
    return mesh


def prepare_upload(request, folder):
    config = options(request.get("options", {}))
    rgb_name, rgb_bytes = uploaded(request.get("rgb"), (".jpg", ".jpeg", ".png"), 20 * 1024**2)
    cad_name, cad_bytes = uploaded(request.get("cad"), (".stl", ".obj", ".ply"), 50 * 1024**2)
    rgb = read_rgb(rgb_bytes)
    mesh = cad_mesh(cad_bytes, Path(cad_name).suffix.lower(), config["cad_units"])
    depth = camera = None
    if config["mode"] == "rgbd":
        depth_name, content = uploaded(request.get("depth"), (".npy", ".png"), 20 * 1024**2)
        depth = read_depth(content, Path(depth_name).suffix.lower(), rgb.size)
        _, content = uploaded(request.get("camera"), (".json",), 1024**2)
        camera = read_camera(content, depth, rgb.size)
    folder.mkdir(parents=True)
    rgb.save(folder / "input_rgb.png")
    (folder / ("input_cad" + Path(cad_name).suffix.lower())).write_bytes(cad_bytes)
    mesh.export(folder / "object_mm.obj", include_color=False)
    if depth is not None:
        Image.fromarray(depth).save(folder / "depth.png")
        (folder / "camera.json").write_text(json.dumps(camera))
    inputs = {"rgb_name": rgb_name, "cad_name": cad_name, "cad_filename": "input_cad" + Path(cad_name).suffix.lower(),
              "input_center_mm": mesh.metadata['input_center_mm'],
              "options": config, "rgb_size": list(rgb.size), "cad_bounds_mm": mesh.bounds.tolist()}
    (folder / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
    return inputs
