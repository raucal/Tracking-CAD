"""Per-view audits must expose independent winners and preserve original geometry inputs."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.score_all_cads import Detections, score_view_details
from suites.ism_report_details import reference_gallery, view_audit_section


class IsmViewDetails(unittest.TestCase):
    def test_independent_winners_and_measured_geometry_on_same_mask(self):
        model = SimpleNamespace(ref_data={'descriptors': torch.zeros(1, 3, 2)}, visible_thred=.5)
        model.matching_config = SimpleNamespace(metric=lambda q, r: torch.tensor([[[.9, .3, .5]]]))
        model.compute_appearance_score = lambda v, o, p: (torch.tensor([[.1, .8, .4][int(v[0])]]), p)
        model.project_template_to_image = lambda v, o, b, m: torch.tensor([[[2, 3], [8, 9]]])
        model.Calculate_the_query_translation = lambda *args: torch.tensor([[.1, .2, 1.]])
        detections = Detections({'boxes': torch.tensor([[0, 0, 10, 10]]), 'masks':torch.ones(1, 10, 10)})
        batch = {'depth':torch.ones(1, 10, 10), 'cam_intrinsic':torch.eye(3)[None], 'depth_scale':1}
        result = score_view_details(model, torch.zeros(1, 2), torch.ones(1, 3, 2), detections, batch, torch.tensor([[0]]))[0]
        self.assertEqual(result['semantic_winner'], 0)
        self.assertEqual(result['appearance_winner'], 1)
        self.assertEqual(len(result['semantic_by_view']), 3)
        self.assertEqual(result['geometry_details']['intersection_px2'], 36)
        self.assertEqual(result['geometry_details']['union_px2'], 100)
        self.assertEqual(detections.masks.shape, (1, 10, 10))
        result.update(part_key='cad', semantic=.7, appearance=.1, geometry=.36, visible_ratio=.5, score=.392)
        page = view_audit_section({'status':'scored', 'scores':[result]})
        for value in ['Ganadora semántica', 'Ganadora apariencia', 'vista ISM 0', '36.000 px²', '100.000 px²']:
            self.assertIn(value, page)

    def test_gallery_exports_every_available_view_and_escapes_labels(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'templates';folder.mkdir();target=root/'report'
            for view in range(42): Image.new('RGB',(2,2)).save(folder/f'rgb_{view}.png')
            part={'key':'cad', 'name':'<CAD>', 'sam_template_dir':'templates'}
            page=reference_gallery([part], {'cad':part}, root, target)
            self.assertEqual(len(list((target/'referencias_cad/cad').glob('*.png'))),42)
            self.assertIn('42 vistas',page);self.assertIn('Vista 41',page);self.assertIn('&lt;CAD&gt;',page)
            self.assertNotIn('<CAD>',page)

    def test_missing_measurements_are_explicit(self):
        self.assertIn('mediciones no disponibles', view_audit_section(None))
        self.assertIn('mediciones no disponibles', view_audit_section({'status':'no_saved_region'}))


if __name__ == '__main__': unittest.main()
