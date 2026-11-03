"""Metric-depth and temporal loss heuristics, without GPU or camera."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam6d_ui.backends.tracking_quality import PoseHealth
from settings import profile

class TrackingQualityTests(unittest.TestCase):
    def setUp(self):
        self.config=profile('interface')['tracking']['quality']
        self.mesh=trimesh.creation.box(extents=[.2,.2,.02])
        self.K=np.array([[200,0,100],[0,200,100],[0,0,1]],float)
        self.pose=np.eye(4);self.pose[2,3]=.5
        self.depth=np.full((200,200),.49)
        self.health=PoseHealth(self.mesh,self.config)

    def test_aligned_surface_is_healthy(self):
        quality=self.health.evaluate(self.pose,self.depth,self.K,1.)
        self.assertTrue(quality['ok'],quality)
        self.assertGreater(quality['depth_support'],.9)
        self.assertGreaterEqual(quality['depth_samples'],20)

    def test_mathematically_valid_pose_with_wrong_depth_is_rejected(self):
        quality=self.health.evaluate(self.pose,np.full((200,200),.8),self.K,1.)
        self.assertFalse(quality['ok']);self.assertIn('depth_mismatch',quality['reasons'])

    def test_insufficient_depth_and_outside_image_are_rejected(self):
        result=self.health.evaluate(self.pose,np.zeros((200,200)),self.K,1.)
        self.assertIn('insufficient_object_depth',result['reasons'])
        outside=self.pose.copy();outside[0,3]=2
        result=self.health.evaluate(outside,self.depth,self.K,1.)
        self.assertFalse(result['ok']);self.assertIn('outside_image',result['reasons'])

    def test_partial_occlusion_is_tolerated_but_full_occlusion_is_lost(self):
        partial=self.depth.copy();partial[:,:90]=.3
        result=self.health.evaluate(self.pose,partial,self.K,1.)
        self.assertTrue(result['ok'],result)
        result=self.health.evaluate(self.pose,np.full((200,200),.3),self.K,1.1)
        self.assertFalse(result['ok']);self.assertIn('occluded',result['reasons'])

    def test_translation_jump_does_not_replace_last_accepted_pose(self):
        self.assertTrue(self.health.evaluate(self.pose,self.depth,self.K,1.)['ok'])
        jumped=self.pose.copy();jumped[0,3]=.04
        result=self.health.evaluate(jumped,self.depth,self.K,1.01)
        self.assertIn('translation_jump',result['reasons'])
        recovered=self.pose.copy();recovered[0,3]=.001
        self.assertTrue(self.health.evaluate(recovered,self.depth,self.K,1.02)['ok'])

    def test_rotation_jump_and_motion_gap(self):
        self.health.evaluate(self.pose,self.depth,self.K,1.)
        rotated=self.pose.copy();a=np.deg2rad(50)
        rotated[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
        result=self.health.evaluate(rotated,self.depth,self.K,1.01)
        self.assertIn('rotation_jump',result['reasons'])
        result=self.health.evaluate(rotated,self.depth,self.K,2.)
        self.assertFalse(result['motion_checked']);self.assertTrue(result['ok'],result)

    def test_motion_uses_object_center_not_remote_cad_origin(self):
        mesh=self.mesh.copy();mesh.apply_translation([1,0,0]);health=PoseHealth(mesh,self.config)
        pose=self.pose.copy();pose[0,3]=-1
        self.assertTrue(health.evaluate(pose,self.depth,self.K,1.)['ok'])
        rotated=pose.copy();a=np.deg2rad(10)
        rotated[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
        rotated[:3,3]=np.array([0,0,.5])-rotated[:3,:3]@health.center
        result=health.evaluate(rotated,self.depth,self.K,1.1)
        self.assertTrue(result['ok'],result);self.assertAlmostEqual(result['translation_jump_mm'],0,places=5)

    def test_reset_accepts_new_location_without_stale_motion_anchor(self):
        self.health.evaluate(self.pose,self.depth,self.K,1.)
        self.health.reset();pose=self.pose.copy();pose[0,3]=.04
        result=self.health.evaluate(pose,self.depth,self.K,1.01)
        self.assertTrue(result['ok']);self.assertFalse(result['motion_checked'])

    def test_custom_profile_inherits_quality_defaults_and_rejects_bad_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'interface.json'
            path.write_text(json.dumps({'tracking':{'quality':{'depth_tolerance_mm':30}}}))
            quality=profile('interface',path)['tracking']['quality']
            self.assertEqual(quality['depth_tolerance_mm'],30);self.assertEqual(quality['sample_points'],1024)
            for changes in ({'quality_bad_frames':0},{'recovery_retry_s':0},{'max_detection_age_s':float('nan')},{'quality':{'min_depth_support':2}},{'quality':{'sample_points':True}}):
                path.write_text(json.dumps({'tracking':changes}))
                with self.assertRaises(ValueError):profile('interface',path)

class TrackingStartDefaultsTests(unittest.TestCase):
    def call(self,folder,request,manual=False):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import threading,time
        from sam6d_ui.server import App
        inputs={'capture_id':'capture','cad_filename':'input_cad.stl','options':{'segmentor':'sam','confidence':.5}}
        if manual: inputs['manual_bbox']=[1,1,8,8]
        (folder/'inputs.json').write_text(json.dumps(inputs))
        (folder/'capture.json').write_text(json.dumps({'timestamp':time.time()}))
        start=Mock(return_value={'running':True})
        app=SimpleNamespace(submission_lock=threading.Lock(),lock=threading.Lock(),jobs={},
            job=lambda _:dict(status='complete',result={'backend':'foundationpose','pose_6d_estimated':True}),
            job_folder=lambda _:folder,camera=SimpleNamespace(capture_folder=lambda _:folder),
            tracker=SimpleNamespace(start=start))
        App.start_tracking(app,{'job_id':'job',**request})
        return start.call_args.kwargs

    def test_api_uses_configured_backup_when_request_omits_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs=self.call(Path(folder),{})
            self.assertEqual(kwargs['redetect_interval_s'],10)
            self.assertIsNotNone(kwargs['redetect'])

    def test_api_zero_backup_keeps_automatic_detector(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs=self.call(Path(folder),{'redetect_interval_s':0})
            self.assertEqual(kwargs['redetect_interval_s'],0)
            self.assertIsNotNone(kwargs['redetect'])

    def test_api_manual_initialization_stays_sam_free(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs=self.call(Path(folder),{},manual=True)
            self.assertEqual(kwargs['redetect_interval_s'],0)
            self.assertIsNone(kwargs['redetect'])


if __name__=='__main__':unittest.main()
