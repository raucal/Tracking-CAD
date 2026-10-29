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

PROJECTS = Path(__file__).resolve().parents[3]
PIPELINE = PROJECTS / "SAM6D/pipeline"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare", PIPELINE / "prepare.py")
asap = load_module("asap_pddl", PROJECTS / "ASAP/pddl_pipeline.py")
sam = load_module("sam_pddl", PROJECTS / "SAM6D/SAM-6D/pddl_pipeline.py")
sim_string = load_module("asap_sim_string", PROJECTS / "ASAP/plan_sequence/sim_string.py")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    pytest.importorskip("manifold3d")
    output = tmp_path_factory.mktemp("converted data with spaces")
    data = prepare.prepare(PIPELINE / "config.json", output, repair=True)
    return output, data


def test_realsense_scale_is_millimetres_per_raw_integer():
    source = PROJECTS / "PDDL/DINOPDDL/realsense_photos/control_board_depth.json"
    metadata = json.loads(source.read_text())
    depth = np.load(source.with_suffix(".npy"), allow_pickle=False)
    camera = prepare.camera_from_metadata(metadata, depth, (640, 480))
    assert camera["depth_scale"] == pytest.approx(1.0)
    assert 500 * camera["depth_scale"] / 1000 == pytest.approx(0.5)
    assert camera["cam_K"][0] == metadata["intrinsics"]["fx"]
    wrong = copy.deepcopy(metadata)
    wrong["intrinsics"]["coefficients"][0] = 0.1
    with pytest.raises(ValueError, match="distorsión"):
        prepare.camera_from_metadata(wrong, depth, (640, 480))
    with pytest.raises(ValueError, match="resoluciones"):
        prepare.camera_from_metadata(metadata, depth, (1280, 720))
    wrong = dict(metadata, aligned_to="depth")
    with pytest.raises(ValueError, match="alineada"):
        prepare.camera_from_metadata(wrong, depth, (640, 480))


def test_real_dataset_inventory_and_lossless_depth(prepared):
    output, data = prepared
    assert len(data["parts"]) == 10
    assert len(data["frames"]) == 22
    assert data["unpaired_rgb"] == ["front_housing.jpg"]
    assert data["asap_meshes_ready"]
    for frame in data["frames"]:
        original = np.load(output / frame["source_depth"], allow_pickle=False)
        with Image.open(output / frame["depth"]) as image:
            np.testing.assert_array_equal(np.array(image), original)
    by_name = {f["name"]: f for f in data["frames"]}
    assert by_name["termal_fan"]["filename_hint_only"] == "thermal_core"
    assert by_name["prueba 1"]["filename_hint_only"] is None


def test_assembly_uses_local_placements_and_one_global_scale(prepared):
    output, data = prepared
    T = np.array(data["T_asap_from_assembly_mm"])
    np.testing.assert_allclose(T @ data["T_assembly_mm_from_asap"], np.eye(4), atol=1e-12)
    bounds = []
    for part in data["parts"]:
        sam_mesh = trimesh.load_mesh(output / part["sam_ply"], process=False)
        np.testing.assert_allclose(sam_mesh.bounds.mean(axis=0), 0, atol=1e-5)
        mesh = trimesh.load_mesh(output / "asap/with_options" / f"{part['key']}.obj", process=False)
        expected = np.array(part["bounds_local_mm"]) + np.array(part["T_assembly_from_local_mm"])[:3, 3]
        np.testing.assert_allclose(mesh.bounds, expected @ T[:3, :3].T + T[:3, 3], atol=1e-6)
        assert mesh.is_watertight
        bounds.append(mesh.bounds)
    bounds = np.array(bounds)
    assert np.max(bounds[:, 1].max(0) - bounds[:, 0].min(0)) == pytest.approx(10)
    assert bounds[:, 0, 2].min() == pytest.approx(0)
    assert len(list((output / "asap/without_options").glob("*.obj"))) == 8


def test_base_repair_preserves_original_triangles_and_volume(prepared):
    output, data = prepared
    base = next(p for p in data["parts"] if p["fixed"])
    assert base["topology_repair"] == "split_coincident_vertices_same_triangles"
    original = trimesh.load_mesh(output / base["source_stl"])
    repaired = prepare.repair_topology(original)
    assert not original.is_watertight
    assert repaired.is_watertight
    assert len(repaired.faces) == len(original.faces)
    assert repaired.volume == pytest.approx(original.volume, rel=1e-9)
    np.testing.assert_array_equal(repaired.bounds, original.bounds)
    assert prepare.sha256(output / base["source_stl"]) == base["source_sha256"]


def test_placement_drift_is_rejected():
    config = json.loads((PIPELINE / "config.json").read_text())
    config["assembled_y_mm"]["01_front_housing"] = 0
    with pytest.raises(ValueError, match="desactualizada"):
        prepare.check_placements(config, PIPELINE / config["placement_source"])


def test_pose_transform_restores_the_original_local_cad_frame(prepared):
    _, data = prepared
    part = next(p for p in data["parts"] if p["key"] == "01_front_housing")
    T_camera_sam = np.array([[0, -1, 0, 20], [1, 0, 0, 30], [0, 0, 1, 500], [0, 0, 0, 1]], dtype=float)
    T_camera_local = T_camera_sam @ np.linalg.inv(part["T_local_from_sam_mm"])
    sam_origin_in_local = np.array(part["T_local_from_sam_mm"])[:, 3]
    np.testing.assert_allclose(T_camera_local @ sam_origin_in_local, T_camera_sam[:, 3])


def test_render_depth_unprojection_has_metric_scale_and_camera_axis_convention():
    path = PROJECTS / "SAM6D/SAM-6D/SAM-6D/Render/render_pddl_templates.py"
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


def test_failed_physics_never_becomes_an_assembly_plan():
    result = asap.summarize({"success": False, "sequence": None}, [])
    assert not result["physics_validated"]
    assert result["assembly_order"] is None


def test_assembly_reverses_disassembly_and_includes_fixed_base(prepared):
    _, data = prepared
    parts = data["parts"]
    removals = [p["key"] for p in parts if not p["fixed"]]
    result = asap.summarize({"success": True, "sequence": removals}, parts)
    assert result["assembly_order"] == ["00_rear_mounting_plate", *reversed(removals)]
    assert not result["robot_arm_validated"]
    with pytest.raises(ValueError):
        asap.summarize({"success": True, "sequence": removals[:-1]}, parts)


def test_sam_needs_metric_render_completion_markers(prepared):
    output, data = prepared
    assert sam.template_problems(output, data["parts"]) == [p["key"] for p in data["parts"]]


def test_asap_xml_supports_cad_names_and_retains_numeric_colors(prepared):
    output, data = prepared
    xml = sim_string.get_contact_sim_string(str(output / "asap/with_options"))
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    names = {body.attrib["name"] for body in root.iter("body")}
    assert names == {f"part{p['key']}" for p in data["parts"]}
    np.testing.assert_allclose(sim_string._get_color("0"), np.array([210, 87, 89, 255]) / 255)
    np.testing.assert_array_equal(sim_string._get_color("05_control_board"), sim_string._get_color("05_control_board"))


@pytest.mark.parametrize("script,arguments", [
    ("SAM6D/SAM-6D/pddl_pipeline.py", ["segment", "--frame", "control_board"]),
    ("SAM6D/SAM-6D/pddl_pipeline.py", ["render", "--part", "05_control_board"]),
    ("ASAP/pddl_pipeline.py", ["plan", "--variant", "without_options"]),
])
def test_launchers_accept_paths_with_spaces_without_running_models(prepared, script, arguments):
    output, _ = prepared
    completed = subprocess.run([sys.executable, str(PROJECTS / script), *arguments,
                                "--manifest", str(output / "manifest.json"), "--dry-run"],
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert str(output) in completed.stdout
    assert not (output / "sam6d/results").exists()
