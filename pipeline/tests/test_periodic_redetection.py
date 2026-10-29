"""SAM verification must not pause tracking or overwrite a pose with old results."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam6d_ui.camera.tracking import LivePoseTracker,validate_interval
from sam6d_ui.camera.redetection import Redetector
from sam6d_ui.backends.geometry import projected_bbox,bbox_agreement

class Camera:
    def __init__(self):self.number=0
    def frame(self):
        self.number+=1
        return dict(number=self.number,timestamp=time.time(),rgb=np.full((10,10,3),self.number%200,np.uint8),depth=np.full((10,10),500,np.uint16),camera={'cam_K':[100,0,5,0,100,5,0,0,1],'depth_scale':1})

class Worker:
    def __init__(self):self.registrations=0;self.fail=False;self.tracks=0;self.box=[0,0,10,10]
    def start_tracking(self,folder):pass
    def status(self):return {'pid':42}
    def track(self,scratch):
        time.sleep(.01)
        if self.fail:raise RuntimeError('lost pose')
        self.tracks+=1;(scratch/'preview.jpg').write_bytes(b'jpeg')
        return {'T_camera_from_input_cad_mm':np.eye(4).tolist(),'bbox_xyxy':self.box}
    def relocalize(self,scratch):
        self.registrations+=1;self.fail=False;self.box=[0,0,10,10]
        self.mask=np.load(scratch/'mask.npy');self.depth=np.load(scratch/'frame.npz')['depth']
        return {'seconds':.01}

class PeriodicTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.camera=Camera();self.worker=Worker();self.seed=self.root/'seed';self.seed.mkdir()
        (self.seed/'camera.json').write_text(json.dumps(self.camera.frame()['camera']))
        self.tracker=LivePoseTracker(self.camera,self.root/'sessions',resident_factory=lambda:self.worker);self.addCleanup(self.tracker.stop)
        self.detected=[]
    def until(self,check,timeout=6):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if check():return
            time.sleep(.01)
        self.fail(str(self.tracker.status()))
    def detector(self,frame,scratch,cancelled):
        self.detected.append(frame['number']);np.save(scratch/'mask.npy',np.ones((10,10),bool))
        return {'score':.8,'segmentor':'sam','bbox_xyxy':[0,0,10,10]}
    def test_interval_validation(self):
        for value in (True,'3',None,-1,.5,float('nan'),float('inf'),3601):
            with self.assertRaises(ValueError):validate_interval(value)
        self.assertEqual(validate_interval(3),3);self.assertEqual(validate_interval(0),0)
    def test_healthy_tracking_advances_while_sam_runs_without_reinitializing(self):
        entered=threading.Event();release=threading.Event()
        def detector(frame,scratch,cancelled):
            entered.set();release.wait(2);return self.detector(frame,scratch,cancelled)
        self.addCleanup(release.set)
        self.tracker.start(self.seed,1,detector);self.assertTrue(entered.wait(3))
        before=self.worker.tracks;time.sleep(.2)
        self.assertGreater(self.worker.tracks,before+3);self.assertEqual(self.tracker.status()['state'],'tracking');self.assertIsNotNone(self.tracker.latest_preview())
        release.set();self.until(lambda:self.tracker.status()['sam_checks']>=1)
        self.assertEqual(self.worker.registrations,0)
        self.assertTrue(self.tracker.status()['last_sam_agreement']['consistent'])
    def test_two_disagreements_recover_once_then_consistent_checks_keep_pose(self):
        self.worker.box=[40,40,50,50];self.tracker.start(self.seed,1,self.detector)
        self.until(lambda:self.tracker.status()['sam_checks']==1)
        self.assertEqual(self.worker.registrations,0);self.assertEqual(self.tracker.status()['state'],'tracking')
        self.until(lambda:self.worker.registrations==1)
        self.until(lambda:self.tracker.status()['sam_checks']>=3)
        self.assertEqual(self.worker.registrations,1);self.assertEqual(self.tracker.status()['worker_pid'],42)
    def test_one_miss_keeps_pose_two_misses_suspend_until_fresh_detection(self):
        attempts=[]
        def detector(frame,scratch,cancelled):
            attempts.append(frame['number'])
            return None if len(attempts)<3 else self.detector(frame,scratch,cancelled)
        self.tracker.start(self.seed,1,detector);self.until(lambda:self.tracker.status()['sam_misses']==1)
        self.assertEqual(self.tracker.status()['state'],'tracking');self.assertIsNotNone(self.tracker.jpeg)
        self.until(lambda:self.tracker.status()['state']=='recovering')
        self.assertIsNone(self.tracker.status()['pose']);self.assertIsNone(self.tracker.jpeg);self.assertEqual(self.worker.registrations,0)
        self.until(lambda:self.worker.registrations==1);self.until(lambda:self.tracker.status()['state']=='tracking')
        self.assertGreater(attempts[2],attempts[1])
    def test_tracker_failure_triggers_fresh_sam_recovery(self):
        self.worker.fail=True;self.tracker.start(self.seed,1,self.detector)
        self.until(lambda:self.worker.registrations==1);self.until(lambda:self.tracker.status()['state']=='tracking')
    def test_detection_from_before_tracking_error_cannot_restore_pose(self):
        entered=threading.Event();release=threading.Event();calls=[]
        def detector(frame,scratch,cancelled):
            calls.append(frame['number'])
            if len(calls)==1:entered.set();release.wait(2)
            return self.detector(frame,scratch,cancelled)
        self.addCleanup(release.set);self.tracker.start(self.seed,1,detector);self.assertTrue(entered.wait(3))
        self.worker.fail=True;self.until(lambda:self.tracker.status()['state']=='recovering');release.set()
        self.until(lambda:self.worker.registrations==1)
        self.assertGreaterEqual(self.tracker.status()['sam_discarded'],1);self.assertGreaterEqual(len(calls),2)
    def test_stop_during_detection_never_registers_late_result(self):
        entered=threading.Event()
        def detector(frame,scratch,cancelled):
            entered.set();cancelled.wait(3);return self.detector(frame,scratch,cancelled)
        self.tracker.start(self.seed,1,detector);self.assertTrue(entered.wait(3))
        self.tracker.stop();self.assertEqual(self.worker.registrations,0);self.assertFalse(self.tracker.busy())
    def test_disabling_sam_discards_inflight_result_and_keeps_tracking(self):
        entered=threading.Event()
        def detector(frame,scratch,cancelled):
            entered.set();cancelled.wait(3);return self.detector(frame,scratch,cancelled)
        self.tracker.start(self.seed,1,detector);self.assertTrue(entered.wait(3))
        self.tracker.configure(0);self.until(lambda:not self.tracker.status()['sam_running'])
        before=self.worker.tracks;time.sleep(.1)
        self.assertGreater(self.worker.tracks,before);self.assertEqual(self.worker.registrations,0);self.assertEqual(self.tracker.status()['sam_checks'],0)
    def test_slow_detection_skips_missed_ticks_without_queueing(self):
        times=[];active=0;maximum=0
        def detector(frame,scratch,cancelled):
            nonlocal active,maximum
            times.append(time.monotonic());active+=1;maximum=max(maximum,active);cancelled.wait(1.1);active-=1
            return self.detector(frame,scratch,cancelled)
        self.tracker.start(self.seed,1,detector);self.until(lambda:len(times)>=2)
        self.assertGreaterEqual(times[1]-times[0],1.9);self.assertEqual(maximum,1);self.assertGreater(self.worker.tracks,50)
    def test_stale_result_is_discarded_without_interrupting_valid_tracking(self):
        def detector(frame,scratch,cancelled):frame['timestamp']-=3;return self.detector(frame,scratch,cancelled)
        self.tracker.start(self.seed,1,detector);self.until(lambda:self.tracker.status()['sam_discarded']>=1)
        self.assertEqual(self.worker.registrations,0);self.assertEqual(self.tracker.status()['state'],'tracking')
    def test_consistency_uses_pose_of_same_snapshot_not_current_pose(self):
        original=self.worker.track
        def track(scratch):
            result=original(scratch)
            with np.load(scratch/'frame.npz') as data:x=int(data['rgb'][0,0,0])
            result['bbox_xyxy']=[x,0,x+10,10];return result
        self.worker.track=track
        def detector(frame,scratch,cancelled):
            cancelled.wait(.2);result=self.detector(frame,scratch,cancelled);x=int(frame['rgb'][0,0,0]);result['bbox_xyxy']=[x,0,x+10,10];return result
        self.tracker.start(self.seed,1,detector);self.until(lambda:self.tracker.status()['sam_checks']>=1)
        self.assertTrue(self.tracker.status()['last_sam_agreement']['consistent']);self.assertEqual(self.worker.registrations,0)

class DetectorTests(unittest.TestCase):
    def test_current_mask_is_returned_without_exports_and_never_reused_on_miss(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seed=root/'seed';seed.mkdir();scratch=root/'scratch';scratch.mkdir()
            for name in ('input_cad.stl','object_mm.obj'):(seed/name).write_text('test fixture')
            inputs={'cad_filename':'input_cad.stl','manual_bbox':[1,1,5,5],'recording_path':'old','options':{'segmentor':'sam','confidence':.5,'pose':True}}
            (seed/'inputs.json').write_text(json.dumps(inputs))
            class Engine:
                found=True
                def run(self,folder,inputs,update,save_outputs=True):
                    assert not save_outputs;assert not inputs['options']['pose'];assert 'manual_bbox' not in inputs;assert 'recording_path' not in inputs
                    return {'detections':[{'score':.8}] if self.found else [],'_masks':[np.ones((10,10),bool)]}
            engine=Engine();detector=Redetector(lambda:engine,seed);frame=Camera().frame()
            result=detector(frame,scratch,threading.Event());self.assertEqual(result['score'],.8);self.assertEqual(result['bbox_xyxy'],[0,0,10,10])
            self.assertTrue((scratch/'mask.npy').is_file());self.assertFalse((scratch/'detection/mask_000.png').exists())
            engine.found=False;self.assertIsNone(detector(frame,scratch,threading.Event()));self.assertFalse((scratch/'mask.npy').exists())

class GeometryTests(unittest.TestCase):
    def test_projection_and_agreement(self):
        pose=np.eye(4);pose[2,3]=1;K=np.array([[100,0,50],[0,100,50],[0,0,1]])
        box=projected_bbox(np.array([[-.1,-.1,0],[.1,.1,0]]),pose,K,(100,100,3))
        np.testing.assert_allclose(box,[40,40,60,60]);self.assertTrue(bbox_agreement(box,[40,40,60,60])['consistent'])
        self.assertFalse(bbox_agreement(box,[0,0,10,10])['consistent']);self.assertIsNone(bbox_agreement(None,box))
        pose[2,3]=-1;self.assertIsNone(projected_bbox(np.array([[-.1,-.1,0],[.1,.1,0]]),pose,K,(100,100,3)))

if __name__=='__main__':unittest.main()
