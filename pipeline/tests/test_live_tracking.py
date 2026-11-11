"""Persistent-worker lifecycle and error propagation without a GPU or camera."""
from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam6d_ui.camera.tracking import LivePoseTracker

class Camera:
    def __init__(self):self.n=0;self.fail=False
    def frame(self):
        if self.fail:raise RuntimeError('TEST: camera disconnected')
        self.n+=1
        return {'number':self.n,'timestamp':time.time(),'rgb':np.zeros((10,10,3),np.uint8),'depth':np.ones((10,10),np.uint16),'camera':{'cam_K':[100,0,5,0,100,5,0,0,1],'depth_scale':1}}

class LiveTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.camera=Camera();self.seed=self.root/'seed';self.seed.mkdir()
        (self.seed/'camera.json').write_text(json.dumps(self.camera.frame()['camera']))
        worker=self.root/'worker.py'
        worker.write_text('''import argparse,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--seed');p.add_argument('--scratch');a=p.parse_args()\nprint('@LIVE@'+json.dumps({'status':'ready'}),flush=True)\nimport sys\nfor line in sys.stdin:\n Path(a.scratch,'preview.jpg').write_bytes(b'preview')\n print('@LIVE@'+json.dumps({'status':'frame','T_camera_from_input_cad_mm':[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}),flush=True)\n''')
        self.tracker=LivePoseTracker(self.camera,self.root/'sessions',worker)
        self.addCleanup(self.tracker.stop)
        self.python=patch('sam6d_ui.camera.tracking.python_path',return_value=Path(sys.executable));self.python.start();self.addCleanup(self.python.stop)
    def until(self,check):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if check():return
            time.sleep(.01)
        self.fail(str(self.tracker.status()))
    def test_reuses_worker_and_latest_frame_then_releases_process(self):
        self.tracker.start(self.seed);self.until(lambda:self.tracker.status()['frames']>=3)
        process=self.tracker.process
        self.assertGreater(self.tracker.status()['fps'],0)
        self.assertEqual(self.tracker.preview(-1)[1],b'preview')
        with self.assertRaises(ValueError):self.tracker.start(self.seed)
        self.tracker.stop();self.assertIsNotNone(process.poll());self.assertFalse(self.tracker.busy())
        metadata=next((self.root/'sessions').glob('*/session.json'))
        self.assertFalse(json.loads(metadata.read_text())['running'])
    def test_camera_failure_stops_worker_and_preserves_error(self):
        self.tracker.start(self.seed);self.until(lambda:self.tracker.status()['frames']>=1)
        self.camera.fail=True;self.until(lambda:not self.tracker.busy())
        self.assertIn('disconnected',self.tracker.status()['error'])
        self.assertEqual(self.tracker.status()['state'],'error')
    def test_changed_calibration_requires_new_initialization(self):
        self.tracker.start(self.seed);self.until(lambda:self.tracker.status()['frames']>=1)
        calibration=json.loads((self.seed/'camera.json').read_text())
        # Change the camera matrix while preserving a valid shared frame.
        original=self.camera.frame
        def changed():
            frame=original();frame['camera']['cam_K'][0]=200;return frame
        self.camera.frame=changed;self.until(lambda:not self.tracker.busy())
        self.assertIn('Calibration',self.tracker.status()['error'])

if __name__=='__main__':unittest.main()
