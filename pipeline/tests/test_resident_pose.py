"""Protocol, recovery, and shared-estimator lifecycle without CUDA."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam6d_ui.backends.resident import ResidentPose
from sam6d_ui.camera.tracking import LivePoseTracker
import numpy as np

class ResidentPoseTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  script=self.root/'worker.py';script.write_text('''import json,sys,os,time
print('ordinary library output',flush=True)
print('@POSE@'+json.dumps({'status':'ready','load_s':.01}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['command']=='slow':time.sleep(1)
 if r['command']=='error':out={'status':'error','error':'bad mask'}
 else:out={'status':'ok','result':{'pid':os.getpid()}}
 print('@POSE@'+json.dumps({'request_id':r['id'],**out}),flush=True)
''')
  self.worker=ResidentPose(sys.executable,script,self.root/'worker.log');self.addCleanup(self.worker.close)
 def test_estimates_share_process_and_error_does_not_unload_weights(self):
  first=self.worker.request('estimate')['pid']
  self.assertEqual(first,self.worker.request('estimate')['pid'])
  with self.assertRaisesRegex(RuntimeError,'bad mask'):self.worker.request('error')
  self.assertEqual(first,self.worker.request('estimate')['pid'])
  self.assertIn('ordinary library output',(self.root/'worker.log').read_text())
 def test_close_releases_worker_and_next_request_recovers(self):
  self.worker.warmup();process=self.worker.process;self.worker.close()
  self.assertIsNotNone(process.poll());self.assertFalse(self.worker.status()['ready'])
  self.worker.warmup();self.assertNotEqual(process.pid,self.worker.status()['pid'])
 def test_timeout_releases_worker(self):
  with self.assertRaisesRegex(RuntimeError,'tiempo'):self.worker.request('slow',timeout=.03)
  self.assertIsNone(self.worker.process)
 def test_tracking_uses_same_worker_without_unloading_on_stop(self):
  root=self.root;seed=root/'seed';seed.mkdir();camera_data={'cam_K':[100,0,5,0,100,5,0,0,1],'depth_scale':1}
  (seed/'camera.json').write_text(json.dumps(camera_data))
  class Camera:
   number=0
   def frame(self):
    self.number+=1;return dict(number=self.number,timestamp=time.time(),rgb=np.zeros((10,10,3),np.uint8),depth=np.ones((10,10),np.uint16),camera=camera_data)
  class Worker:
   starts=0;calls=0
   def start_tracking(self,folder):self.starts+=1
   def track(self,scratch):
    self.calls+=1;(scratch/'preview.jpg').write_bytes(b'jpeg');time.sleep(.01);return {'T_camera_from_input_cad_mm':np.eye(4).tolist()}
   def status(self):return {'pid':42}
  worker=Worker();tracker=LivePoseTracker(Camera(),root/'sessions',resident_factory=lambda:worker);self.addCleanup(tracker.stop)
  tracker.start(seed);deadline=time.monotonic()+3
  while tracker.status()['frames']<2 and time.monotonic()<deadline:time.sleep(.01)
  tracker.stop();self.assertGreaterEqual(worker.calls,2);self.assertEqual(worker.starts,1);self.assertEqual(tracker.status()['worker_pid'],42);self.assertFalse(tracker.busy())

if __name__=='__main__':unittest.main()
