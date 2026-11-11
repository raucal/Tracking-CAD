"""Validate uploaded files and RGB-D units before any model execution."""
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam6d_ui.inputs import cad_mesh, options, prepare_upload, read_camera, read_depth, uploaded  # noqa: E402


class UiInputs(unittest.TestCase):
    def payload(self):
        image = BytesIO()
        Image.new("RGB", (16, 12), "red").save(image, format="PNG")
        cad = trimesh.creation.box().export(file_type="stl")
        return {"rgb": {"name": "capture.png", "data": base64.b64encode(image.getvalue()).decode()},
                "cad": {"name": "object.stl", "data": base64.b64encode(cad).decode()}}

    def test_rgb_does_not_create_fake_depth(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "job"
            result = prepare_upload(self.payload(), path)
            self.assertEqual(result["options"]["mode"], "rgb")
            self.assertFalse((path / "depth.png").exists())
            self.assertFalse((path / "camera.json").exists())

    def test_rgbd_requires_depth_and_camera(self):
        request = self.payload()
        request["options"] = {"mode": "rgbd"}
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            prepare_upload(request, Path(directory) / "job")

    def test_depth_is_lossless_uint16_and_aligned(self):
        array = np.full((12, 16), 500, dtype=np.uint16)
        image = BytesIO()
        Image.fromarray(array).save(image, format="PNG")
        np.testing.assert_array_equal(read_depth(image.getvalue(), ".png", (16, 12)), array)
        with self.assertRaisesRegex(ValueError, "same size"):
            read_depth(image.getvalue(), ".png", (8, 12))
        image = BytesIO()
        Image.new("RGB", (16, 12)).save(image, format="PNG")
        with self.assertRaises(ValueError):
            read_depth(image.getvalue(), ".png", (16, 12))

    def test_bop_camera_preserves_mm_scale(self):
        camera = read_camera(json.dumps({"cam_K": [100, 0, 8, 0, 100, 6, 0, 0, 1], "depth_scale": 1.0}),
                             np.ones((12, 16), dtype=np.uint16), (16, 12))
        self.assertEqual(camera["depth_scale"], 1.0)
        with self.assertRaises(ValueError):
            read_camera(json.dumps({"cam_K": [0] * 9, "depth_scale": 0}), np.ones((12, 16)), (16, 12))

    def test_cad_units_are_converted_explicitly(self):
        source = trimesh.creation.box(extents=[1, 2, 3]).export(file_type="stl")
        mesh = cad_mesh(source, ".stl", "m")
        np.testing.assert_allclose(mesh.extents, [1000, 2000, 3000])
        np.testing.assert_allclose(mesh.bounds.mean(axis=0), 0)

    def test_bad_files_and_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            uploaded({"name": "object.step", "data": "YWJj"}, (".stl",), 1024)
        with self.assertRaises(ValueError):
            uploaded({"name": "object.stl", "data": "not base64"}, (".stl",), 1024)
        for bad in ({"confidence": 0.2, "min_score": 0.5}, {"stability": float("nan")}, {"mode": "fake_depth"}):
            with self.assertRaises(ValueError):
                options(bad)

    def test_pose_requires_rgbd_and_cuda(self):
        for bad in ({"pose": True, "mode": "rgb"}, {"pose": True, "mode": "rgbd", "device": "cpu"}, {"pose": "false"}):
            with self.assertRaises(ValueError):
                options(bad)
        valid = options({"segmentor": "fastsam", "mode": "rgbd", "pose": True})
        self.assertEqual(valid["segmentor"], "fastsam")
        self.assertTrue(valid["pose"])

    def test_center_offset_preserves_input_cad_coordinate_frame(self):
        mesh = trimesh.creation.box(extents=[1, 2, 3])
        mesh.apply_translation([4, 5, 6])
        centered = cad_mesh(mesh.export(file_type="stl"), ".stl", "m")
        np.testing.assert_allclose(centered.metadata['input_center_mm'], [4000, 5000, 6000])
        np.testing.assert_allclose(centered.bounds.mean(axis=0), 0)


if __name__ == "__main__":
    unittest.main()
