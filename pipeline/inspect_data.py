"""Audit the converted files and draw a standalone RGB-D/CAD overview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from prepare import HERE, sha256, write_json


def audit(manifest_path):
    base = manifest_path.parent
    data = json.loads(manifest_path.read_text())
    for frame in data["frames"]:
        original = np.load(base / frame["source_depth"], allow_pickle=False)
        with Image.open(base / frame["depth"]) as source:
            converted = np.array(source)
        if not np.array_equal(original, converted):
            raise ValueError(f"La conversión altera la profundidad de {frame['name']}")
        with Image.open(base / frame["source_rgb"]) as source:
            original_rgb = np.array(source.convert("RGB"))
        with Image.open(base / frame["rgb"]) as source:
            if not np.array_equal(original_rgb, np.array(source)):
                raise ValueError(f"La conversión altera el RGB de {frame['name']}")
    T = np.array(data["T_asap_from_assembly_mm"])
    if not np.allclose(T @ data["T_assembly_mm_from_asap"], np.eye(4)):
        raise ValueError("Transformaciones de unidades incompatibles.")
    dense_counts = {}
    for variant, folder in data["asap_assemblies"].items():
        expected = [p for p in data["parts"] if variant == "with_options" or not p["optional"]]
        if {p.stem for p in (base / folder).glob("*.obj")} != {p["key"] for p in expected}:
            raise ValueError(f"Piezas incorrectas en {variant}")
        for part in expected:
            if sha256(base / part["source_stl"]) != part["source_sha256"]:
                raise ValueError(f"El original ha cambiado: {part['key']}")
            sam = trimesh.load_mesh(base / part["sam_ply"], process=False)
            placed = trimesh.load_mesh(base / folder / f"{part['key']}.obj", process=False)
            expected_bounds = np.array(part["bounds_local_mm"]) + np.array(part["T_assembly_from_local_mm"])[:3, 3]
            expected_bounds = expected_bounds @ T[:3, :3].T + T[:3, 3]
            if not np.allclose(placed.bounds, expected_bounds, atol=1e-5):
                raise ValueError(f"Pieza mal colocada: {part['key']}")
            if not np.allclose(sam.bounds.mean(axis=0), 0, atol=1e-5):
                raise ValueError(f"CAD de SAM no centrado: {part['key']}")
            if not placed.is_watertight:
                raise ValueError(f"Malla de ASAP no cerrada: {part['key']}")
        dense_dir = base / "asap_dense" / variant
        if (dense_dir / "pddl_preprocessing.json").is_file():
            meshes = [trimesh.load_mesh(dense_dir / f"{p['key']}.obj", process=False) for p in expected]
            if not all(m.is_watertight for m in meshes):
                raise ValueError("Malla densa no cerrada.")
            dense_counts[variant] = {p["key"]: len(m.faces) for p, m in zip(expected, meshes)}
    return {"status": "data_checks_passed", "parts": len(data["parts"]),
            "rgbd_frames": len(data["frames"]), "unpaired_rgb": data["unpaired_rgb"],
            "depth_and_rgb_preserved": True, "cad_placements_checked": True,
            "all_asap_meshes_watertight": True, "dense_mesh_faces": dense_counts,
            "topology_repairs": [p["key"] for p in data["parts"] if p["topology_repair"]],
            "scope": "data_only_no_model_or_physics_validation"}


def preview(manifest_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    frame = next(f for f in data["frames"] if f["name"] == "mounted_front_housing")
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    ax = fig.add_subplot(221)
    with Image.open(base / frame["rgb"]) as source:
        ax.imshow(source)
    ax.set_title("Captura real RGB (sin segmentación)")
    ax.axis("off")
    ax = fig.add_subplot(222)
    with Image.open(base / frame["depth"]) as source:
        depth = np.array(source, dtype=float)
    camera = json.loads((base / frame["camera"]).read_text())
    depth *= camera["depth_scale"]
    im = ax.imshow(np.ma.masked_equal(depth, 0), cmap="viridis", vmin=350, vmax=650)
    fig.colorbar(im, ax=ax, label="Profundidad (mm)")
    ax.set_title("Profundidad alineada al color")
    ax.axis("off")
    for panel, exploded in ((223, False), (224, True)):
        ax = fig.add_subplot(panel, projection="3d")
        lows, highs = [], []
        for index, part in enumerate(data["parts"]):
            mesh = trimesh.load_mesh(base / part["sam_ply"], process=False)
            T = np.array(part["T_assembly_from_local_mm"]) @ part["T_local_from_sam_mm"]
            mesh.apply_transform(T)
            if exploded:
                if part["optional"]:
                    mesh.apply_translation([0, 0, 120 + 70 * ("03b" in part["key"])])
                elif not part["fixed"]:
                    mesh.apply_translation([0, part["T_assembly_from_local_mm"][1][3] * 2.5, 0])
            collection = Poly3DCollection(mesh.triangles, alpha=0.55,
                                         facecolor=plt.cm.tab10(index / 10), linewidth=0)
            ax.add_collection3d(collection)
            lows.append(mesh.bounds[0])
            highs.append(mesh.bounds[1])
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
        ax.set_box_aspect(high - low, zoom=0.85)
        ax.tick_params(labelsize=8)
        ax.view_init(elev=21, azim=55)
        ax.set_title("Vista separada ilustrativa; no es un plan" if exploded else "CAD en posición final de montaje")
    fig.suptitle("PDDL · Entradas preparadas para SAM-6D y ASAP", fontsize=16)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "prepared/manifest.json")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    report = audit(path)
    write_json(path.parent / "data_audit.json", report)
    if args.preview:
        preview(path, path.parent / "datos_preparados.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
