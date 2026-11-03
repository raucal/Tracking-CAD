"""Prepare RGB-D captures and CADs for detection and pose estimation, without models.

All output paths in manifest.json are relative to that manifest. Originals are
read-only. This step does not produce detections or a physically validated plan.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from settings import PREPARED, dataset_path
HERE = dataset_path().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def translation(x=0.0, y=0.0, z=0.0):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def camera_from_metadata(metadata, depth, rgb_size):
    """BOP/SAM-6D depth_scale is millimetres per raw integer, not metres."""
    if metadata.get("aligned_to") != "color":
        raise ValueError("La profundidad debe estar alineada al color.")
    height, width = int(metadata["height"]), int(metadata["width"])
    if depth.shape != (height, width) or tuple(rgb_size) != (width, height):
        raise ValueError("RGB, profundidad y calibración tienen resoluciones distintas.")
    if depth.dtype != np.uint16 or metadata.get("dtype") != "uint16":
        raise ValueError("Se esperaba profundidad raw uint16; no se convierte silenciosamente.")
    if metadata.get("invalid_depth_value") != 0:
        raise ValueError("SAM-6D espera cero como profundidad inválida.")
    scale = float(metadata["depth_scale_m_per_unit"]) * 1000.0
    if not np.isfinite(scale) or scale <= 0 or not np.any(depth > 0):
        raise ValueError("Escala o profundidad no válida.")
    intr = metadata["intrinsics"]
    if np.any(np.abs(intr.get("coefficients", [])) > 1e-10):
        raise ValueError("Hay distorsión no nula: rectificar RGB-D antes de usar SAM-6D.")
    fx, fy, cx, cy = [float(intr[k]) for k in ("fx", "fy", "ppx", "ppy")]
    if not np.all(np.isfinite([fx, fy, cx, cy])) or min(fx, fy) <= 0:
        raise ValueError("Intrínsecos de cámara no válidos.")
    return {"cam_K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], "depth_scale": scale}




def repair_topology(mesh):
    """Split singular coincident vertices, accepting only unchanged triangles.

    Manifold can disambiguate a closed STL with four faces sharing an edge.
    We refuse any operation that moves, removes, or adds geometric triangles.
    """
    import manifold3d
    import trimesh

    solid = manifold3d.Manifold(manifold3d.Mesh(
        np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32)))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f"No se puede reparar la topología sin cambiar geometría: {solid.status()}")
    result = solid.to_mesh()
    repaired = trimesh.Trimesh(result.vert_properties[:, :3], result.tri_verts, process=False)
    def signature(m):
        return sorted(tuple(sorted(tuple(float(x) for x in v) for v in triangle))
                      for triangle in np.asarray(m.triangles, dtype=np.float32))
    if not repaired.is_watertight or not repaired.is_winding_consistent or signature(repaired) != signature(mesh):
        raise ValueError("La reparación cambia triángulos o no deja una malla cerrada.")
    if not np.isclose(repaired.volume, mesh.volume, rtol=1e-6):
        raise ValueError("La reparación cambia el volumen del CAD.")
    return repaired


def prepare(config_path, output, repair=False):
    import trimesh

    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    cfg = json.loads(config_path.read_text())
    def resolve(key):
        return (config_path.parent / cfg[key]).resolve()
    cad_dir, photos = resolve("cad_dir"), resolve("photos_dir")
    # Never overwrite source data or share a destination between source inputs.
    for source in (cad_dir, photos):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("La salida debe estar separada de los datos originales.")
    catalog = yaml.safe_load(resolve("catalog").read_text())
    by_key = {p["cad_stem"]: dict(p, name=name, fixed=False)
              for name, p in catalog["parts"].items()}
    by_key.update({p["cad_stem"]: dict(p, name=name, fixed=True, optional=False)
                   for name, p in catalog["fixed_parts"].items()})
    stls = {p.stem: p for p in cad_dir.glob("*.stl")}
    if set(stls) != set(by_key):
        raise ValueError("El inventario CAD y catálogo no coincide.")
    output.mkdir(parents=True, exist_ok=True)
    # An existing output must belong to this tool, avoiding stale optional meshes.
    marker = output / ".cad_tracking_prepared"
    if not marker.exists() and any(output.iterdir()):
        raise ValueError("El directorio de salida no está vacío ni pertenece a esta herramienta.")
    marker.touch()
    def rel(p):
        return os.path.relpath(Path(p).resolve(), output)
    manifest = {"schema_version": 1, "assembly_id": cfg["assembly_id"],
                "status": "data_prepared_not_inferred", "parts": [], "frames": [],
                "warnings": [], "provenance": {key: sha256(resolve(key)) for key in
                ("catalog",)}}
    for category_id, key in enumerate(sorted(stls), 1):
        mesh = trimesh.load_mesh(stls[key], process=True)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"Malla vacía o no triangular: {key}")
        if not np.all(np.isfinite(mesh.vertices)) or mesh.volume <= 0:
            raise ValueError(f"Geometría no válida: {key}")
        original_watertight = bool(mesh.is_watertight)
        if not original_watertight and repair:
            mesh = repair_topology(mesh)
        info = by_key[key]
        center = mesh.bounds.mean(axis=0)
        centered = mesh.copy()
        centered.apply_translation(-center)
        model_dir = output / "sam6d" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        ply = model_dir / f"obj_{category_id:06d}.ply"
        obj = model_dir / f"obj_{category_id:06d}.obj"
        centered.export(ply)
        centered.export(obj, include_color=False)
        edges_count = np.bincount(mesh.edges_unique_inverse)
        part = {"category_id": category_id, "key": key, "name": cfg.get("display_names", {}).get(key, info["name"]),
                "fixed": info["fixed"], "optional": info.get("optional", False),
                "source_stl": rel(stls[key]), "source_sha256": sha256(stls[key]),
                "sam_ply": rel(ply), "sam_obj": rel(obj),
                "sam_template_dir": f"sam6d/templates/{key}/templates",
                "T_local_from_sam_mm": translation(*center).tolist(),
                "bounds_local_mm": mesh.bounds.tolist(),
                "topology_repair": "split_coincident_vertices_same_triangles" if not original_watertight and repair else None,
                "source_watertight": original_watertight,
                "mesh_quality": {"watertight": bool(mesh.is_watertight),
                    "winding_consistent": bool(mesh.is_winding_consistent),
                    "nonmanifold_edges": int(np.sum(edges_count > 2)),
                    "boundary_edges": int(np.sum(edges_count == 1)),
                    "faces": len(mesh.faces), "volume_mm3": float(mesh.volume)}}
        manifest["parts"].append(part)
        if not mesh.is_watertight:
            manifest["warnings"].append(f"Malla CAD no cerrada; revisar antes de usarla en simulación: {key}")

    seen_rgb = set()
    names = {p["name"] for p in manifest["parts"]}
    for frame_id, path in enumerate(sorted(photos.glob("*_depth.json"))):
        metadata = json.loads(path.read_text())
        rgb = photos / metadata["color_image"]
        depth_file = photos / metadata["depth_image"]
        if rgb.parent.resolve() != photos or depth_file.parent.resolve() != photos:
            raise ValueError("Los nombres RGB-D deben estar dentro del directorio de capturas.")
        depth = np.load(depth_file, allow_pickle=False)
        with Image.open(rgb) as im:
            camera = camera_from_metadata(metadata, depth, im.size)
            converted_rgb = im.convert("RGB")
        name = rgb.stem
        if name in seen_rgb:
            raise ValueError(f"Captura duplicada: {name}")
        seen_rgb.add(name)
        frame_dir = output / "sam6d" / "frames" / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        converted_rgb.save(frame_dir / "rgb.png")
        Image.fromarray(depth).save(frame_dir / "depth.png")
        write_json(frame_dir / "camera.json", camera)
        alias = name.removeprefix("mounted_")
        alias = cfg["photo_aliases"].get(alias, alias)
        # Filename hints aid selection only: they are never segmentation labels.
        hint = alias if alias in names else None
        manifest["frames"].append({"frame_id": frame_id, "name": name,
            "rgb": rel(frame_dir / "rgb.png"), "depth": rel(frame_dir / "depth.png"),
            "camera": rel(frame_dir / "camera.json"), "source_rgb": rel(rgb),
            "source_depth": rel(depth_file), "source_metadata": rel(path),
            "source_rgb_sha256": sha256(rgb), "prepared_rgb_sha256": sha256(frame_dir / "rgb.png"),
            "filename_hint_only": hint,
            "valid_depth_fraction": float(np.mean(depth > 0)),
            "depth_percentiles_mm": (np.percentile(depth[depth > 0], [1, 50, 99]) * camera["depth_scale"]).tolist()})
    unpaired = sorted(p.name for p in photos.glob("*.jpg") if p.stem not in seen_rgb)
    manifest["unpaired_rgb"] = unpaired
    if unpaired:
        manifest["warnings"].append("RGB sin profundidad/calibración: " + ", ".join(unpaired))
    write_json(output / "manifest.json", manifest)
    print(f"Preparadas {len(manifest['parts'])} piezas y {len(manifest['frames'])} capturas RGB-D: {output}")
    for warning in manifest["warnings"]:
        print("AVISO:", warning)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=dataset_path())
    parser.add_argument("--output", type=Path, default=PREPARED)
    parser.add_argument("--repair-topology", action="store_true", help="Repair singular topology without changing triangles; requires manifold3d.")
    args = parser.parse_args()
    prepare(args.config, args.output, repair=args.repair_topology)


if __name__ == "__main__":
    main()
