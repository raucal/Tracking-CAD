"""Check benchmark equivalence, missing detections and explicit device handling."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ISM = Path(__file__).resolve().parents[3] / "SAM6D/SAM-6D/SAM-6D/Instance_Segmentation_Model"
sys.path.insert(0, str(ISM))
from model.detector import Instance_Segmentation_Model  # noqa: E402
from model.loss import PairwiseSimilarity  # noqa: E402
from run_suite_pddl import assign, choose_device, compare_all, process_frame  # noqa: E402


class SamSuite(unittest.TestCase):
    def test_per_cad_comparison_matches_original_assignment(self):
        torch.manual_seed(25)
        for count in (1, 7):
            for method in ("avg_5", "mean", "median", "max"):
                with self.subTest(proposals=count, aggregation=method):
                    refs = torch.rand(10, 42, 64)
                    query = torch.rand(count, 64)
                    holder = SimpleNamespace(ref_data={"descriptors": refs},
                                             matching_config=SimpleNamespace(metric=PairwiseSimilarity(),
                                                                             aggregation_function=method,
                                                                             confidence_thresh=0.2))
                    holder.best_template_pose = lambda scores, objects: scores[
                        torch.arange(len(objects)), objects].argmax(dim=-1)
                    expected = Instance_Segmentation_Model.compute_semantic_score(holder, query)
                    semantic, views, times = compare_all(holder, query, torch.device("cpu"))
                    actual = assign(semantic, views, 0.2)
                    for a, b in zip(expected, actual):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(semantic.shape, (count, 10))
                    self.assertEqual(len(times), 10)
                    self.assertTrue(all(value > 0 for value in times))

    def test_semantic_threshold_and_no_selected_proposal(self):
        semantic = torch.tensor([[0.2, 0.1], [0.8, 0.9]])
        views = torch.tensor([[4, 5], [6, 7]])
        selected, objects, _, best = assign(semantic, views, 0.2)
        self.assertEqual(selected.tolist(), [1])
        self.assertEqual(objects.tolist(), [1])
        self.assertEqual(best.tolist(), [7])
        selected, objects, _, best = assign(semantic[:1], views[:1], 0.2)
        self.assertEqual(len(selected), 0)
        self.assertEqual(len(objects), 0)
        self.assertEqual(len(best), 0)

    def test_explicit_cuda_never_falls_back_silently(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                choose_device("cuda")
            self.assertEqual(choose_device("auto").type, "cpu")
            self.assertEqual(choose_device("cpu").type, "cpu")

    def test_empty_sam_result_still_records_times_and_all_candidates(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            Image.new("RGB", (16, 12)).save(base / "rgb.png")
            Image.fromarray(np.full((12, 16), 500, dtype=np.uint16)).save(base / "depth.png")
            (base / "camera.json").write_text(json.dumps({"cam_K": [100, 0, 8, 0, 100, 6, 0, 0, 1], "depth_scale": 1.0}))
            proposals = {"boxes": torch.empty((0, 4)), "masks": torch.empty((0, 12, 16))}
            model = SimpleNamespace(segmentor_model=SimpleNamespace(generate_masks=lambda rgb: proposals))
            frame = {"name": "case with spaces", "frame_id": 1, "rgb": "rgb.png", "depth": "depth.png", "camera": "camera.json"}
            parts = [{"key": str(i), "name": str(i), "category_id": i + 1} for i in range(10)]
            case, _, detections, masks, accepted = process_frame(model, frame, parts, base, torch.device("cpu"), 0.3, 0.5)
            self.assertFalse(case["recognized"])
            self.assertIsNone(case["candidate"])
            self.assertEqual(len(case["comparisons"]), 10)
            self.assertEqual((detections, masks, accepted), ([], [], []))
            self.assertGreater(case["timing"]["processing_s"], 0)
            self.assertTrue(all(t >= 0 for t in case["timing"].values()))
            self.assertEqual(case["timing"]["dino_s"], 0)
            self.assertTrue(all(c["semantic_comparison_s"] == 0 and c["semantic_score"] is None for c in case["comparisons"]))
            json.dumps(case, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
