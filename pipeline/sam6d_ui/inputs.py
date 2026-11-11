"""Validate local UI input before scheduling any GPU work."""
import base64
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from settings import profile
DEFAULTS = profile('interface')['inference']


def options(value):
    if not isinstance(value, dict):
        raise ValueError('Options must be a JSON object.')
    result = {**DEFAULTS, **{k: v for k, v in value.items() if k in DEFAULTS}}
    for key, allowed in {"mode": ("rgb", "rgbd"), "device": ("auto", "cpu", "cuda"),
                         "segmentor": ("sam", "fastsam"),
                         "pose_backend": ("pem", "foundationpose"),
                         "backbone": ("vit_h", "vit_l", "vit_b"), "cad_units": ("mm", "cm", "m"),
                         "aggregation": ("avg_5", "mean", "median", "max")}.items():
        if result[key] not in allowed:
            raise ValueError(f"Invalid option {key}.")
    for key in ("min_score", "confidence", "stability", "pred_iou", "nms", "fastsam_conf", "fastsam_iou"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    if result["min_score"] > result["confidence"]:
        raise ValueError("The retention threshold cannot exceed the acceptance threshold.")
    if not isinstance(result["pose"], bool):
        raise ValueError("The pose option must be true or false.")
    if result['pose_backend'] == 'foundationpose':
        result['pose'] = True
    if result["pose"] and (result["mode"] != "rgbd" or result["device"] == "cpu"):
        raise ValueError("6D pose requires RGB-D mode and a CUDA GPU.")
    result["points_per_batch"] = int(result["points_per_batch"])
    if not 1 <= result["points_per_batch"] <= 128:
        raise ValueError("Points per batch must be between 1 and 128.")
    return result


def uploaded(item, extensions, maximum):
    if not isinstance(item, dict) or not item.get("data"):
        raise ValueError("No file selected.")
    name = str(item.get("name", "")).replace("\\", "/").split("/")[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in extensions:
        raise ValueError(f"Formato no admitido: {suffix}. Se admite: {', '.join(extensions)}.")
    try:
        content = base64.b64decode(item["data"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Incorrectly encoded file.") from error
    if not content or len(content) > maximum:
        raise ValueError("File is empty or too large.")
    return name, content


def read_rgb(content):
    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > 4_000_000:
            raise ValueError("Image exceeds 4 megapixels. Resize RGB and depth consistently.")
        return image.convert("RGB")


def read_depth(content, suffix, size):
    if suffix == ".npy":
        depth = np.load(BytesIO(content), allow_pickle=False)
    else:
        with Image.open(BytesIO(content)) as source:
            depth = np.array(source)
    if depth.ndim != 2 or depth.shape != (size[1], size[0]):
        raise ValueError("Depth must be an aligned array with the same size as RGB.")
    if depth.dtype != np.uint16:
        raise ValueError("Raw uint16 depth is required (.npy or 16-bit PNG), not a colourized image.")
    if not np.any(depth > 0):
        raise ValueError("Depth contains no valid values.")
    return depth


def read_camera(content, depth, size):
    camera = json.loads(content)
    if "cam_K" not in camera:
        from tools.prepare import camera_from_metadata
        return camera_from_metadata(camera, depth, size)
    matrix = np.array(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    scale = float(camera["depth_scale"])
    if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics.")
    if not np.allclose(matrix[2], [0, 0, 1]) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid camera matrix or depth scale.")
    if np.any(np.abs(camera.get("dist_coeffs", [])) > 1e-10):
        raise ValueError("RGB-D must be rectified, without residual distortion.")
    return {"cam_K": matrix.reshape(-1).tolist(), "depth_scale": scale}


def cad_mesh(content, suffix, units):
    mesh = trimesh.load(BytesIO(content), file_type=suffix.lstrip("."), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("The CAD does not contain a valid triangular mesh.")
    if len(mesh.faces) > 250_000:
        raise ValueError("CAD exceeds 250,000 faces. Export a simplified mesh.")
    if not np.isfinite(mesh.vertices).all() or np.max(mesh.extents) <= 0:
        raise ValueError("CAD contains invalid coordinates or dimensions.")
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
