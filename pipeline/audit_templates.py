"""Check rendered CAD templates and export a contact sheet for the experiment."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "prepared/manifest.json")
    args = parser.parse_args()
    base = args.manifest.resolve().parent
    data = json.loads(args.manifest.read_text())
    random = np.random.default_rng(0)
    selected = [0, 7, 21, 34]
    sheet = Image.new("RGB", (800, 90 + 210 * len(data["parts"])), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15)
    draw.text((15, 12), "Referencias CAD usadas por SAM-6D / DINOv2", font=font, fill="black")
    draw.text((15, 36), "Selección de 4 de las 42 vistas por pieza · Render CPU: 256 px, 16 muestras", font=font, fill="black")
    report = {"scope": "rendered_geometry_not_detection_accuracy", "parts": [], "views_checked": 0}
    for row, part in enumerate(data["parts"]):
        folder = base / part["sam_template_dir"]
        metadata = json.loads((folder / "render_metadata.json").read_text())
        if metadata["views"] != 42 or metadata["xyz_unit"] != "mm":
            raise ValueError("Unidades o vistas no válidas.")
        mesh = trimesh.load_mesh(base / part["sam_ply"], process=False)
        points = []
        for index in range(42):
            xyz = np.load(folder / f"xyz_{index}.npy", allow_pickle=False)
            with Image.open(folder / f"mask_{index}.png") as source:
                mask = np.array(source) > 0
            if xyz.dtype != np.float32 or xyz.shape != (*mask.shape, 3) or not np.isfinite(xyz).all() or not np.any(mask):
                raise ValueError(f"XYZ/máscara no válidos: {part['key']}, {index}")
            if not np.all(xyz[~mask] == 0):
                raise ValueError("Hay coordenadas de objeto en el fondo.")
            valid = xyz[mask]
            points.extend(valid[random.choice(len(valid), size=min(8, len(valid)), replace=False)])
            report["views_checked"] += 1
        _, distance, _ = trimesh.proximity.closest_point(mesh, np.array(points))
        report["parts"].append({"key": part["key"], "xyz_unit": "mm", "views": 42,
            "sampled_distance_to_cad_mm": {"median": float(np.median(distance)), "p95": float(np.percentile(distance, 95)), "max": float(np.max(distance))}})
        top = 90 + 210 * row
        draw.text((15, top), part["key"], font=font, fill="black")
        for column, index in enumerate(selected):
            with Image.open(folder / f"rgb_{index}.png") as source:
                thumbnail = source.convert("RGB").resize((176, 176))
            sheet.paste(thumbnail, (15 + column * 195, top + 25))
    sheet.save(base / "plantillas_cad.png")
    (base / "template_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
