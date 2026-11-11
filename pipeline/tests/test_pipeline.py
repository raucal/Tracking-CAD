"""Check physical units, frame transforms, mesh topology and command handoffs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare", PIPELINE / "tools/prepare.py")
sam = load_module("sam_pddl", ROOT / "models/SAM-6D/pddl_pipeline.py")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, source_dataset):
    output = tmp_path_factory.mktemp("converted data with spaces")
    data = prepare.prepare(source_dataset, output, repair=False)
    return output, data


def test_realsense_scale_is_millimetres_per_raw_integer(source_dataset):
    source = source_dataset.parent / "photos/control_board_depth.json"
    metadata = json.loads(source.read_text())
    depth = np.load(source.with_suffix(".npy"), allow_pickle=False)
    camera = prepare.camera_from_metadata(metadata, depth, (64, 48))
    assert camera["depth_scale"] == pytest.approx(1.0)
    assert 500 * camera["depth_scale"] / 1000 == pytest.approx(0.5)
    assert camera["cam_K"][0] == metadata["intrinsics"]["fx"]
    wrong = copy.deepcopy(metadata)
    wrong["intrinsics"]["coefficients"][0] = 0.1
    with pytest.raises(ValueError, match="distortion"):
        prepare.camera_from_metadata(wrong, depth, (64, 48))
    with pytest.raises(ValueError, match="resolutions"):
        prepare.camera_from_metadata(metadata, depth, (1280, 720))
    wrong = dict(metadata, aligned_to="depth")
    with pytest.raises(ValueError, match="aligned"):
        prepare.camera_from_metadata(wrong, depth, (64, 48))


def test_real_dataset_inventory_and_lossless_depth(prepared):
    output, data = prepared
    assert len(data["parts"]) == 10
    assert len(data["frames"]) == 22
    assert data["unpaired_rgb"] == ["front_housing.jpg"]
    assert not any("asap" in key.lower() for key in data)
    assert not (output/"asap").exists()
    for frame in data["frames"]:
        original = np.load(output / frame["source_depth"], allow_pickle=False)
        with Image.open(output / frame["depth"]) as image:
            np.testing.assert_array_equal(np.array(image), original)
    by_name = {f["name"]: f for f in data["frames"]}
    assert by_name["termal_fan"]["filename_hint_only"] == "thermal_core"
    assert by_name["unlabelled_capture"]["filename_hint_only"] is None




def test_base_repair_preserves_original_triangles_and_volume(prepared):
    output, data = prepared
    base = next(p for p in data["parts"] if p["fixed"])
    pytest.importorskip("manifold3d")
    original = trimesh.load_mesh(output / base["source_stl"])
    repaired = prepare.repair_topology(original)
    assert not original.is_watertight
    assert repaired.is_watertight
    assert len(repaired.faces) == len(original.faces)
    assert repaired.volume == pytest.approx(original.volume, rel=1e-9)
    np.testing.assert_array_equal(repaired.bounds, original.bounds)
    assert prepare.sha256(output / base["source_stl"]) == base["source_sha256"]




def test_pose_transform_restores_the_original_local_cad_frame(prepared):
    _, data = prepared
    part = next(p for p in data["parts"] if p["key"] == "01_front_housing")
    T_camera_sam = np.array([[0, -1, 0, 20], [1, 0, 0, 30], [0, 0, 1, 500], [0, 0, 0, 1]], dtype=float)
    T_camera_local = T_camera_sam @ np.linalg.inv(part["T_local_from_sam_mm"])
    sam_origin_in_local = np.array(part["T_local_from_sam_mm"])[:, 3]
    np.testing.assert_allclose(T_camera_local @ sam_origin_in_local, T_camera_sam[:, 3])


def test_render_depth_unprojection_has_metric_scale_and_camera_axis_convention():
    path = ROOT / "models/SAM-6D/SAM-6D/Render/render_pddl_templates.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "xyz_from_depth_mm")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    depth = np.full((3, 3), 2.0)
    K = np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]])
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2
    xyz = namespace["xyz_from_depth_mm"](depth, K, camera_pose, 0.01)
    np.testing.assert_allclose(xyz[1, 1], [0, 0, 0])
    np.testing.assert_allclose(xyz[1, 2], [2, 0, 0])
    np.testing.assert_allclose(xyz[0, 1], [0, 2, 0])






def test_sam_needs_metric_render_completion_markers(prepared):
    output, data = prepared
    assert sam.template_problems(output, data["parts"]) == [p["key"] for p in data["parts"]]




@pytest.mark.parametrize("script,arguments", [
    ("models/SAM-6D/pddl_pipeline.py", ["segment", "--frame", "control_board"]),
    ("models/SAM-6D/pddl_pipeline.py", ["render", "--part", "05_control_board"]),
])
def test_launchers_accept_paths_with_spaces_without_running_models(prepared, script, arguments):
    output, _ = prepared
    completed = subprocess.run([sys.executable, str(ROOT / script), *arguments,
                                "--manifest", str(output / "manifest.json"), "--dry-run"],
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert str(output) in completed.stdout
    assert not (output / "sam6d/results").exists()


def test_vision_audit_without_planning_inputs(prepared):
    from tools.inspect_data import audit
    output,_=prepared
    report=audit(output/'manifest.json')
    assert report['cad_frames_checked']
    assert report['depth_and_rgb_preserved']
    assert report['parts']==10 and report['rgbd_frames']==22


def test_vision_audit_rejects_modified_prepared_rgb(prepared,tmp_path):
    import shutil
    from tools.inspect_data import audit
    output,data=prepared
    copied=tmp_path/'modified'
    shutil.copytree(output,copied)
    manifest=copy.deepcopy(data)
    for frame in manifest['frames']:
        for key in ('source_rgb','source_depth','source_metadata'):
            frame[key]=str((output/frame[key]).resolve())
    (copied/'manifest.json').write_text(json.dumps(manifest))
    rgb=copied/manifest['frames'][0]['rgb']
    Image.new('RGB',(640,480),'red').save(rgb)
    with pytest.raises(ValueError,match='RGB bytes'):
        audit(copied/'manifest.json')
