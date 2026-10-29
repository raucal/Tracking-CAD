"""Exercise upstream tensor methods without importing models or downloading weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import unittest

try:
    import torch
except ImportError:
    torch = None
ISM = Path(__file__).resolve().parents[3] / "SAM6D/SAM-6D/SAM-6D/Instance_Segmentation_Model"


def method(path, class_name, name, globals_dict):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    code = compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec")
    namespace = dict(globals_dict)
    exec(code, namespace)
    return namespace[name]


@unittest.skipUnless(torch is not None, "PyTorch no está instalado en este intérprete")
class TensorShapes(unittest.TestCase):
    def test_dino_mask_cropping_keeps_detection_axis_and_input_shape(self):
        function = method(ISM / "model/dinov2.py", "CustomDINOv2", "process_masks_proposals", {})
        holder = SimpleNamespace(rgb_proposal_processor=lambda masks, boxes: torch.nn.functional.interpolate(masks, size=(14, 14)))
        for count in (1, 3):
            with self.subTest(detections=count):
                masks = torch.ones(count, 8, 9)
                result = function(holder, masks, torch.zeros(count, 4))
                self.assertEqual(result.shape, (count, 14, 14))
                self.assertEqual(masks.shape, (count, 8, 9))

    def test_geometric_translation_keeps_detection_axis_and_units(self):
        depth_module = ast.parse((ISM / "utils/trimesh_utils.py").read_text())
        depth_function = next(n for n in depth_module.body if isinstance(n, ast.FunctionDef) and n.name == "depth_image_to_pointcloud_translate_torch")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[depth_function], type_ignores=[]), "trimesh_utils.py", "exec"), namespace)
        function = method(ISM / "model/detector.py", "Instance_Segmentation_Model", "Calculate_the_query_translation", namespace)
        for count in (1, 3):
            with self.subTest(detections=count):
                masks = torch.ones(count, 8, 9)
                K = torch.tensor([[100, 0, 4], [0, 100, 3.5], [0, 0, 1]], dtype=torch.float32)
                result = function(None, masks, torch.full((8, 9), 500), K, torch.tensor([1.0]))
                self.assertEqual(result.shape, (count, 3))
                self.assertEqual(masks.shape, (count, 8, 9))
                torch.testing.assert_close(result[:, 2], torch.full((count,), 0.5))


if __name__ == "__main__":
    unittest.main()
