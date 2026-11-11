"""All-CAD diagnostics preserve low hypothesis scores and never mutate the saved region."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.score_all_cads import Detections, score_hypotheses  # noqa: E402


class AllCadScores(unittest.TestCase):
    def fixture(self):
        detections = Detections({"boxes": torch.tensor([[0, 0, 8, 9]]), "masks": torch.ones(1, 9, 8)})
        model = SimpleNamespace(ref_data={"descriptors": torch.zeros(3, 42, 4)}, visible_thred=0.5)
        model.compute_appearance_score = lambda views, objects, patches: (torch.tensor([0.2, 0.4, 0.8]), patches)
        model.project_template_to_image = lambda views, objects, batch, masks: torch.zeros(len(objects), 8, 2)
        model.compute_geometric_score = lambda uv, proposals, patches, reference, threshold: (
            torch.tensor([0.1, 0.3, 0.7]), torch.tensor([0.0, 0.5, 1.0]))
        return model, detections

    def test_all_classes_scored_even_below_detection_cutoff(self):
        model, detections = self.fixture()
        semantic = torch.tensor([[0.1, 0.2, 0.9]])
        values = score_hypotheses(model, detections, torch.ones(1, 4, 4), {}, semantic, torch.zeros(1, 3, dtype=torch.long))
        expected = (semantic[0] + torch.tensor([0.2, 0.4, 0.8]) + torch.tensor([0.1, 0.3, 0.7]) * torch.tensor([0, 0.5, 1])) / torch.tensor([2, 2.5, 3])
        torch.testing.assert_close(values["score"], expected)
        self.assertEqual(len(values["score"]), 3)
        self.assertLess(float(values["score"][0]), 0.3)
        self.assertGreater(float(values["score"][0]), 0)
        self.assertEqual(detections.masks.shape, (1, 9, 8))
        self.assertEqual(detections.boxes.shape, (1, 4))

    def test_scalar_zero_from_upstream_geometry_remains_exportable(self):
        model, detections = self.fixture()
        model.compute_geometric_score = lambda *args: (0.0, torch.ones(3))
        values = score_hypotheses(model, detections, torch.ones(1, 4, 4), {},
                                 torch.tensor([[.1, .2, .9]]), torch.zeros(1, 3, dtype=torch.long))
        self.assertEqual(values['geometry'].cpu().tolist(), [0., 0., 0.])
        self.assertTrue(torch.isfinite(values['score']).all())

    def test_nonfinite_score_is_error_not_zero(self):
        model, detections = self.fixture()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_hypotheses(model, detections, torch.ones(1, 4, 4), {},
                             torch.tensor([[float('nan'), 0.2, 0.3]]), torch.zeros(1, 3, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
