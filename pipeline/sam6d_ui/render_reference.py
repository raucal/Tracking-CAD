"""Render single-CAD references with official poses; Blender by default, EGL optional."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
if os.path.isfile("/usr/share/glvnd/egl_vendor.d/10_nvidia.json"):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

from settings import path as configured_path
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pyrender
import trimesh
import cv2


def render_blender(mesh_path, output, poses_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    poses = np.load(poses_path).copy()
    poses[:, :3, 1:3] *= -1
    poses[:, :3, 3] *= 0.002
    output.mkdir(parents=True, exist_ok=True)
    (output / "render_metadata.json").unlink(missing_ok=True)
    spec = output / "render_spec.json"
    spec.write_text(json.dumps({"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist(),
                               "scale": 1 / (2 * np.linalg.norm(mesh.vertices, axis=1).max()), "poses": poses.tolist()}))
    binary = configured_path("blender")
    if not binary.is_file():
        raise ValueError("Falta Blender 3.3.1 en .runtime/blender; consulta la instalación de la interfaz.")
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run([str(binary), "-b", "-t", "8", "--python", str(Path(__file__).with_name("blender_reference.py")),
                    "--", "--spec", str(spec), "--output", str(output)], env=env, check=True, timeout=300)
    for i in range(42):
        raw = output / f"rgba_{i}.png"
        with Image.open(raw) as source:
            rgba = np.array(source.convert("RGBA"))
        mask = rgba[:, :, 3] > 127
        if not mask.any():
            raise ValueError(f"Vista Blender vacía: {i}")
        color = rgba[:, :, :3].copy()
        color[~mask] = 64
        Image.fromarray(color).save(output / f"rgb_{i}.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"mask_{i}.png")
        exr = output / f"position_{i:04d}.exr"
        position = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)
        if position is None:
            raise ValueError(f"No se ha generado el pase métrico XYZ: {i}")
        xyz = position[:, :, :3][:, :, ::-1].copy() / (1 / (2 * np.linalg.norm(mesh.vertices, axis=1).max()))
        xyz[~mask] = 0
        tolerance = max(0.5, float(mesh.extents.max()) * 0.01)
        if not np.isfinite(xyz).all() or np.any(xyz[mask] < mesh.bounds[0] - tolerance) or np.any(xyz[mask] > mesh.bounds[1] + tolerance):
            raise ValueError(f"XYZ fuera de las dimensiones físicas del CAD: {i}")
        np.save(output / f"xyz_{i}.npy", xyz.astype(np.float32))
        exr.unlink()
        raw.unlink()
    spec.unlink()
    (output / "render_metadata.json").write_text(json.dumps({"views": 42, "size": 256, "renderer": "blender_cycles_v2", "xyz_unit": "mm",
        "poses": "official_cam_poses_level0", "mesh_unit": "mm", "cad_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest()}))


def render(mesh_path, output, poses_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    mesh.apply_scale(1 / (2 * radius))
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.4, 0.4, 0.4, 1], metallicFactor=0, roughnessFactor=0.8, doubleSided=True)
    scene = pyrender.Scene(bg_color=[0.25, 0.25, 0.25, 1], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
    camera = scene.add(pyrender.PerspectiveCamera(yfov=0.7, znear=0.01, zfar=10))
    light = scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0))
    renderer = pyrender.OffscreenRenderer(256, 256)
    output.mkdir(parents=True, exist_ok=True)
    (output / "render_metadata.json").unlink(missing_ok=True)
    try:
        poses = np.load(poses_path)
        for index, pose in enumerate(poses):
            pose = pose.copy()
            pose[:3, 1:3] *= -1
            pose[:3, 3] *= 0.002
            scene.set_pose(camera, pose)
            light_pose = pose.copy()
            light_pose[:3, 3] *= 2.5
            scene.set_pose(light, light_pose)
            color, depth = renderer.render(scene)
            mask = depth > 0
            if not mask.any():
                raise ValueError(f"La vista {index} no contiene geometría visible.")
            Image.fromarray(color[:, :, :3]).save(output / f"rgb_{index}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"mask_{index}.png")
            if index % 7 == 0:
                print(f"Render CAD: {index + 1}/{len(poses)}", flush=True)
    finally:
        renderer.delete()
    (output / "render_metadata.json").write_text(json.dumps({"views": 42, "size": 256,
        "renderer": "pyrender_egl_v2", "poses": "official_cam_poses_level0", "mesh_unit": "mm",
        "cad_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "note": "References for ISM only. No XYZ/PEM templates are generated."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--backend", choices=("blender", "egl"), default="blender")
    args = parser.parse_args()
    (render_blender if args.backend == "blender" else render)(args.mesh, args.output, args.poses)
