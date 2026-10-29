"""Explicit real-GPU integration check; not part of automatic unittest discovery."""
import base64
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[3]
ENDPOINT = "http://127.0.0.1:8770"


def api(path, payload=None):
    request = Request(ENDPOINT + path, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def wait(job):
    start = time.monotonic()
    while time.monotonic() - start < 180:
        result = api("/api/jobs/" + job["id"])
        if result["status"] == "error":
            raise RuntimeError(result["message"])
        if result["status"] == "complete":
            return result
        time.sleep(0.5)
    raise TimeoutError("El trabajo no finalizó en tres minutos.")


def main():
    status = api("/api/status")
    assert status["environment"]["cuda_available"]
    labels = {p["key"]: p["name"] for p in status["catalog"]}
    assert labels["03a_vertical_option_a"] == "sensor_module"
    assert labels["03b_vertical_option_b"] == "communications_module"
    try:
        api("/api/jobs", {"options": {"mode": "rgbd"}})
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("La petición incompleta no debe aceptarse.")
    full = wait(api("/api/jobs", {"example": "control_board", "example_cad": "05_control_board",
                                  "options": {"mode": "rgbd", "device": "cuda"}}))
    assert full["result"]["recognized"] and full["result"]["best_score"] > 0.5
    assert full["result"]["mode"] == "rgbd" and full["result"]["device"] == "cuda"
    print("RGB-D:", full["id"], full["result"]["best_score"], flush=True)
    manifest_path = ROOT / "SAM6D/pipeline/prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frame = next(f for f in manifest["frames"] if f["name"] == "control_board")
    part = next(p for p in manifest["parts"] if p["key"] == "05_control_board")
    mesh = trimesh.load_mesh(ROOT / "PDDL/Assembly CAD/Assembly parts" / (part["key"] + ".stl"))
    # Different file format forces the new-CAD render path rather than the known-STL cache.
    ply = mesh.export(file_type="ply")
    request = {"rgb": {"name": "capture.png", "data": base64.b64encode((manifest_path.parent / frame["rgb"]).read_bytes()).decode()},
               "cad": {"name": "new_control_board.ply", "data": base64.b64encode(ply).decode()},
               "options": {"mode": "rgb", "device": "cuda", "cad_units": "mm"}}
    rgb = wait(api("/api/jobs", request))
    assert rgb["result"]["mode"] == "rgb" and rgb["result"]["renderer"] == "blender_cycles_v2"
    assert all(d["geometry_score"] is None for d in rgb["result"]["detections"])
    assert rgb["result"]["best_score"] is not None
    print("RGB, CAD nuevo:", rgb["id"], rgb["result"]["best_score"], flush=True)
    for job in (full, rgb):
        folder = ROOT / "SAM6D/pipeline/prepared/sam6d_ui/jobs" / job["id"]
        detections = json.loads((folder / "detections.json").read_text())
        for i, d in enumerate(detections):
            counts = d["segmentation"]["counts"]
            shape = d["segmentation"]["size"]
            mask = np.repeat(np.arange(len(counts)) % 2, counts).reshape(shape, order="F").astype(bool)
            with Image.open(folder / f"mask_{i:03d}.png") as image:
                assert np.array_equal(np.array(image) > 0, mask)
    proof = {"status": "passed", "rgbd_job": full["id"], "rgbd_best_score": full["result"]["best_score"],
             "rgb_new_cad_job": rgb["id"], "rgb_best_score": rgb["result"]["best_score"],
             "device": "cuda", "new_cad_rendered": True, "masks_verified": True}
    (ROOT / "SAM6D/pipeline/prepared/sam6d_ui/integration_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
