"""The report must display final SAM-6D scores and preserve label provenance."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suites.sam6d_suite_report import comparison_coverage, final_confidences, score_style  # noqa: E402


class SamReport(unittest.TestCase):
    def test_orange_threshold_is_strict_and_winner_has_priority(self):
        self.assertEqual(score_style(0.4), "score-low")
        self.assertEqual(score_style(0.40001), "score-alternative")
        self.assertEqual(score_style(0.8, selected=True), "score-winner")
        self.assertEqual(score_style(0.35, selected=True), "score-winner")

    def test_missing_score_is_never_highlighted(self):
        self.assertEqual(score_style(None, selected=True), "score-empty")

    def setUp(self):
        self.run = {"catalog": [{"key": "cad_a", "name": "pieza_a", "category_id": 1},
                                {"key": "cad_b", "name": "pieza_b", "category_id": 2},
                                {"key": "cad_c", "name": "pieza_c", "category_id": 3}],
                    "frames": [{"name": "foto_con_nombre_cad_a"}], "settings": {"confidence": 0.5}}
        self.detections = [
            {"part_key": "cad_a", "part_name": "pieza_a", "category_id": 1, "score": 0.61, "semantic_score": 0.99},
            {"part_key": "cad_a", "part_name": "pieza_a", "category_id": 1, "score": 0.64, "semantic_score": 0.40},
            {"part_key": "cad_b", "part_name": "pieza_b", "category_id": 2, "score": 0.80, "semantic_score": 0.50},
        ]

    def read(self, run=None, detections=None):
        run = run or self.run
        with TemporaryDirectory() as directory:
            source = Path(directory)
            folder = source / "cases" / run["frames"][0]["name"]
            folder.mkdir(parents=True)
            (folder / "detection_ism.json").write_text(json.dumps(self.detections if detections is None else detections))
            return final_confidences(run, source)

    def test_uses_final_score_not_dino_similarity(self):
        _, rows = self.read()
        self.assertEqual(rows[0]["confianza_final_sam6d"], 0.64)
        self.assertEqual(rows[0]["indice_deteccion"], 1)
        self.assertEqual(rows[1]["confianza_final_sam6d"], 0.80)
        self.assertEqual(rows[0]["origen_score"], "detection_ism.json:score")

    def test_missing_class_score_is_unknown_not_zero(self):
        _, rows = self.read()
        self.assertIsNone(rows[2]["confianza_final_sam6d"])
        self.assertEqual(rows[2]["detecciones_conservadas"], 0)
        self.assertFalse(rows[2]["supera_umbral"])

    def test_photo_filename_does_not_change_predicted_label_or_score(self):
        renamed = copy.deepcopy(self.run)
        renamed["frames"][0]["name"] = "foto_con_nombre_cad_b"
        _, old = self.read()
        _, new = self.read(renamed)
        for a, b in zip(old, new):
            self.assertEqual(a["cad_predicho"], b["cad_predicho"])
            self.assertEqual(a["nombre_catalogo"], b["nombre_catalogo"])
            self.assertEqual(a["confianza_final_sam6d"], b["confianza_final_sam6d"])

    def test_rejects_inconsistent_label_mapping(self):
        wrong = copy.deepcopy(self.detections)
        wrong[0]["part_name"] = "nombre_tomado_de_la_foto"
        with self.assertRaisesRegex(ValueError, "etiqueta"):
            self.read(detections=wrong)

    def coverage_run(self):
        run = copy.deepcopy(self.run)
        run["algorithm"] = {"views_per_cad": 42}
        run["frames"][0].update(proposal_counts={"valid_depth_and_size": 2},
                                proposal_semantic_scores=[[0.3, 0.4, 0.2], [0.2, 0.5, 0.1]],
                                comparisons=[{"part_key": p["key"], "compared_proposals": 2, "semantic_score": 0.3}
                                             for p in run["catalog"]])
        return run

    def test_comparison_coverage_independent_of_retained_detections(self):
        result = comparison_coverage(self.coverage_run())
        self.assertTrue(result["complete"])
        self.assertEqual(result["verified_image_cad_pairs"], 3)
        self.assertTrue(result["frames"][0]["all_cads_compared"])

    def test_missing_comparison_is_not_marked_as_verified(self):
        run = self.coverage_run()
        run["frames"][0]["comparisons"].pop()
        result = comparison_coverage(run)
        self.assertFalse(result["complete"])
        self.assertEqual(result["verified_image_cad_pairs"], 2)

    def test_no_valid_masks_does_not_claim_comparison_was_run(self):
        run = self.coverage_run()
        run["frames"][0]["proposal_counts"]["valid_depth_and_size"] = 0
        run["frames"][0]["proposal_semantic_scores"] = []
        result = comparison_coverage(run)
        self.assertFalse(result["complete"])
        self.assertEqual(result["verified_image_cad_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
