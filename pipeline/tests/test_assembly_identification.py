"""Full-catalog classification and preflight checks without camera or GPU."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam6d_ui import assembly_identification as catalog
from sam6d_ui import engine
from sam6d_ui.inputs import DEFAULTS


class CatalogTests(unittest.TestCase):
    def test_incomplete_or_wrong_catalog_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            dataset = root/'dataset.json'
            dataset.write_text(json.dumps({'assembly_registry': '.'}))
            stack.enter_context(patch.object(catalog, 'dataset_path', return_value=dataset))
            stack.enter_context(patch.object(catalog, 'PREPARED', root))
            stack.enter_context(patch.object(catalog, 'load_registry', return_value={
                'default':'demo', 'assemblies':{'demo':{'assembly_id':'demo','parts':{'one':{}}}}}))
            with self.assertRaisesRegex(ValueError, 'Faltan las plantillas'):
                catalog.assembly_catalog('demo')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'assembly_id':'other','parts':[]}))
            with self.assertRaisesRegex(ValueError, 'no corresponde'):
                catalog.assembly_catalog('demo')
            manifest.write_text(json.dumps({'assembly_id':'demo','parts':[]}))
            with self.assertRaisesRegex(ValueError, 'todas las piezas'):
                catalog.assembly_catalog('demo')
            part={'key':'one', 'source_stl':'one.stl', 'sam_ply':'one.ply',
                  'sam_template_dir':'templates','T_local_from_sam_mm':np.eye(4).tolist()}
            manifest.write_text(json.dumps({'assembly_id':'demo','parts':[part]}))
            with self.assertRaisesRegex(ValueError, 'Faltan mallas o plantillas'):
                catalog.assembly_catalog('demo')
            for filename in ('one.stl','one.ply'):
                (root/filename).touch()
            (root/'templates').mkdir()
            for prefix in ('rgb','mask'):
                for i in range(42):
                    (root/'templates'/f'{prefix}_{i}.png').touch()
            self.assertEqual(catalog.assembly_catalog('demo'),(manifest,[part]))
            with self.assertRaisesRegex(ValueError, 'desconocido'):
                catalog.assembly_catalog('../../unknown')


class EngineIdentificationTests(unittest.TestCase):
    def run_identification(self, scores):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        for filename in ('FastSAM/FastSAM-x.pt','dinov2/dinov2_vitl14_pretrain.pth'):
            target = root/'checkpoints'/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        Image.new('RGB',(8,8)).save(root/'input_rgb.png')
        parts = [dict(key=key,name=key,category_id=i+1,sam_ply=key+'.ply',
                      T_local_from_sam_mm=[[1,0,0,i*10],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                 for i,key in enumerate(('first','second'))]
        detections = [dict(part_key=p['key'],score=score) for p,score in zip(parts,scores)]
        detections.sort(key=lambda d:-d['score'])
        accepted = [i for i,d in enumerate(detections) if d['score']>=.5]
        model = SimpleNamespace(segmentor_model=SimpleNamespace(),
                                post_processing_config=SimpleNamespace(), matching_config=SimpleNamespace())
        service = object.__new__(engine.Engine)
        service.names = {}; service.model = model; service.reference_cache=root/'cache'
        service.reference_token=service.reference_record=None
        service.prepare_model=Mock(return_value=True)
        service.reference_key=lambda part,base:part['key']
        config = {**DEFAULTS,'segmentor':'fastsam','device':'cpu','pose':False,'pose_backend':'pem',
                  'mode':'rgbd','confidence':.5}
        process = Mock(return_value=({'timing':{'processing_s':.01},
                                     'proposal_counts':{'valid_depth_and_size':2},'comparisons':[]},
                                    Image.new('RGB',(8,8)),detections,
                                    [np.ones((8,8),bool) for _ in detections],accepted))
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine,'ISM',root))
            stack.enter_context(patch.object(engine,'environment',return_value={}))
            stack.enter_context(patch.object(engine,'process_frame',process))
            load = stack.enter_context(patch.object(engine,'load_references',return_value={'parts':[{'cache_hit':True}]*2}))
            stack.enter_context(patch.object(catalog,'assembly_catalog',return_value=(root/'manifest.json',parts)))
            result = service.run(root,{'auto_assembly':'demo','cad_name':'placeholder',
                                       'input_center_mm':[999,999,999],'options':config},lambda message:None,save_outputs=False)
        self.assertEqual([p['key'] for p in process.call_args.args[2]],['first','second'])
        self.assertEqual([p['key'] for p in load.call_args.args[1]],['first','second'])
        return result,root

    def test_winning_second_cad_controls_mesh_mask_and_transform(self):
        result,root = self.run_identification([.6,.9])
        self.assertEqual(result['cad_key'],'second')
        self.assertEqual(result['display_name'],'second')
        self.assertEqual(result['tracking_model']['mesh'],str(root/'second.ply'))
        self.assertEqual(result['tracking_model']['model_from_input'][0][3],-10)
        self.assertEqual(len(result['_masks']),1)
        self.assertEqual(result['detections'],[{'part_key':'second','score':.9}])
        self.assertEqual(len(result['identification']['candidates']),2)

    def test_below_threshold_never_claims_a_cad(self):
        result,_ = self.run_identification([.4,.3])
        self.assertFalse(result['recognized'])
        self.assertIsNone(result['cad_key'])
        self.assertIsNone(result['identification']['part_key'])
        self.assertFalse(result['pose_6d_estimated'])


class SubmissionTests(unittest.TestCase):
    def test_automatic_request_uses_catalog_without_requiring_a_user_cad(self):
        import trimesh
        from sam6d_ui.server import App
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'part.stl'
            source.write_bytes(trimesh.creation.box().export(file_type='stl'))
            app=App(root/'results', root, root/'captures')
            frame_folder=root/'capture';frame_folder.mkdir()
            Image.new('RGB',(10,10)).save(frame_folder/'rgb.png')
            Image.fromarray(np.full((10,10),1000,np.uint16)).save(frame_folder/'depth.png')
            (frame_folder/'camera.json').write_text(json.dumps({'cam_K':[100,0,5,0,100,5,0,0,1], 'depth_scale':1}))
            app.camera.capture=Mock(return_value={'id':'capture'})
            app.camera.capture_folder=Mock(return_value=frame_folder)
            app._run=Mock()
            try:
                with patch.object(catalog,'assembly_catalog',return_value=(root/'manifest.json',[{'source_stl':'part.stl'}])):
                    queued=app.submit({'live_camera':True,'auto_assembly':'demo','options':{'mode':'rgbd','pose':False,'pose_backend':'pem','cad_units':'mm'}})
                inputs=app.job(queued['id'])['inputs']
                self.assertEqual(inputs['auto_assembly'],'demo')
                self.assertEqual(inputs['assembly_manifest'],str(root/'manifest.json'))
                self.assertEqual(inputs['server_paths']['cad'],str(source))
                self.assertEqual(inputs['capture_id'],'capture')
            finally:
                app.pool.shutdown(wait=True)

    def test_incompatible_modes_fail_before_capturing(self):
        from sam6d_ui.server import App
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            app=App(root/'results',root,root/'captures')
            app.camera.capture=Mock()
            try:
                with patch.object(catalog,'assembly_catalog',return_value=(root/'manifest.json',[{'source_stl':'part.stl'}])):
                    for extra in ({'live_camera':False},{'manual_bbox':[1,1,4,4]},
                                  {'recording_id':'video'}, {'options':{'mode':'rgb','pose':False,'pose_backend':'pem'}},
                                  {'options':{'mode':'rgbd','cad_units':'cm','pose':False,'pose_backend':'pem'}}):
                        with self.subTest(extra=extra), self.assertRaises(ValueError):
                            app.submit({'live_camera':True,'auto_assembly':'demo',**extra})
                app.camera.capture.assert_not_called()
            finally:
                app.pool.shutdown(wait=True)


class TrackingModelTests(unittest.TestCase):
    def test_switch_reuses_networks_and_updates_transform_and_geometry(self):
        import importlib
        import trimesh
        with patch.object(sys,'path',[str(Path(__file__).resolve().parents[1]/'sam6d_ui/backends'),*sys.path]):
            worker=importlib.import_module('resident_worker')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);mesh=root/'new.ply'
            mesh.write_bytes(trimesh.creation.box(extents=[100,200,300]).export(file_type='ply'))
            models=(object(),object(),object())
            runtime={'models':models,'mesh_key':'old','estimator':SimpleNamespace(mesh_path=None)}
            factory=Mock(return_value=SimpleNamespace(mesh_path=None))
            metadata={'mesh':str(mesh),'model_from_input':np.eye(4).tolist()}
            with patch.dict(sys.modules,{'estimater':SimpleNamespace(FoundationPose=factory)}), patch.object(worker,'configure_tracking') as configure:
                worker.select_tracking_model(runtime,metadata,root)
                configure.assert_called_once_with(runtime)
                self.assertEqual(factory.call_args.kwargs['scorer'],models[0])
                self.assertEqual(factory.call_args.kwargs['refiner'],models[1])
                self.assertEqual(factory.call_args.kwargs['glctx'],models[2])
                np.testing.assert_allclose(runtime['mesh'].extents,[.1,.2,.3])
                metadata['model_from_input'][0][3]=42
                worker.select_tracking_model(runtime,metadata,root)
                self.assertEqual(factory.call_count,1)
                self.assertEqual(runtime['model_from_input'][0,3],42)


if __name__ == '__main__':
    unittest.main()
