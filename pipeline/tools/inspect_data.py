"""Audit the converted files and draw a standalone RGB-D/CAD overview."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from tools.prepare import HERE, sha256, write_json
from settings import PREPARED


def audit(manifest_path):
    base = manifest_path.parent
    data = json.loads(manifest_path.read_text())
    rgb_warnings=[]
    for frame in data["frames"]:
        original = np.load(base / frame["source_depth"], allow_pickle=False)
        with Image.open(base / frame["depth"]) as source:
            converted = np.array(source)
        if not np.array_equal(original, converted):
            raise ValueError(f"Conversion changes depth for {frame['name']}")
        with Image.open(base / frame["source_rgb"]) as source:
            original_rgb = np.array(source.convert("RGB"))
        with Image.open(base / frame["rgb"]) as source:
            converted_rgb=np.array(source)
        rgb_hashes=frame.get('source_rgb_sha256') and frame.get('prepared_rgb_sha256')
        if rgb_hashes:
            if sha256(base/frame['source_rgb'])!=frame['source_rgb_sha256'] or sha256(base/frame['rgb'])!=frame['prepared_rgb_sha256']:
                raise ValueError('RGB bytes have changed: '+frame['name'])
        elif not np.array_equal(original_rgb,converted_rgb):
            # Legacy PNGs were prepared with another JPEG decoder. Preserve them;
            # they are the exact images used by the saved benchmarks.
            delta=np.abs(original_rgb.astype(float)-converted_rgb.astype(float))
            rgb_warnings.append({'frame':frame['name'],'max_pixel_difference':float(delta.max()),
                                 'mean_pixel_difference':float(delta.mean()),'reason':'legacy input without RGB byte hashes; decoder-dependent equality cannot establish source preservation'})
    for part in data['parts']:
        if sha256(base/part['source_stl']) != part['source_sha256']:
            raise ValueError('The original has changed: '+part['key'])
        model=trimesh.load_mesh(base/part['sam_ply'],process=False)
        if not np.allclose(model.bounds.mean(axis=0),0,atol=1e-5):
            raise ValueError('CAD no centrado: '+part['key'])
        restored=model.copy();restored.apply_transform(np.asarray(part['T_local_from_sam_mm']))
        if not np.allclose(restored.bounds,part['bounds_local_mm'],atol=1e-5):
            raise ValueError('The transformation changes the original CAD frame: '+part['key'])
    return {'status':'data_checks_passed_with_rgb_warnings' if rgb_warnings else 'data_checks_passed','parts':len(data['parts']),
            'rgbd_frames':len(data['frames']),'unpaired_rgb':data['unpaired_rgb'],
            'depth_preserved':True,'depth_and_rgb_preserved':not rgb_warnings,'rgb_warnings':rgb_warnings,'cad_frames_checked':True,
            'scope':'vision_inputs_only'}


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
    ax.set_title("Real RGB capture (without segmentation)")
    ax.axis("off")
    ax = fig.add_subplot(222)
    with Image.open(base / frame["depth"]) as source:
        depth = np.array(source, dtype=float)
    camera = json.loads((base / frame["camera"]).read_text())
    depth *= camera["depth_scale"]
    im = ax.imshow(np.ma.masked_equal(depth, 0), cmap="viridis", vmin=350, vmax=650)
    fig.colorbar(im, ax=ax, label="Depth (mm)")
    ax.set_title("Depth aligned to colour")
    ax.axis("off")
    ax=fig.add_subplot(212)
    names=[p['key'] for p in data['parts']]
    ax.barh(names,[p['mesh_quality']['faces'] for p in data['parts']])
    ax.set_xlabel('CAD triangles');ax.set_title('Models prepared for vision')
    fig.suptitle('CAD Tracking · Entradas RGB-D y modelos CAD',fontsize=16)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PREPARED / "manifest.json")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path, help="Save audit JSON only when requested")
    args = parser.parse_args()
    path = args.manifest.resolve()
    report = audit(path)
    if args.output: write_json(args.output, report)
    if args.preview:
        preview(path, path.parent / "datos_preparados.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
