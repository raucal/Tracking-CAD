"""A slow viewer must jump to the latest frame instead of replaying TCP-buffered video."""
import json
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam6d_ui.camera import CameraService
from sam6d_ui.camera.realsense import RealSenseSource
from sam6d_ui.server import App,handler

class Source:
    def __init__(self,options):self.number=0
    def read(self):
        time.sleep(1/30);self.number+=1
        return dict(number=self.number,timestamp=time.time(),device_timestamp_ms=self.number*1000/30,rgb=np.full((10,10,3),self.number%200,np.uint8),depth=np.full((10,10),500,np.uint16),camera={'cam_K':[100,0,5,0,100,5,0,0,1],'depth_scale':1})
    def close(self):pass

class LatestVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name)
        self.app=App(root/'data',root,root/'captures');self.app.camera=CameraService(root/'captures',Source)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.app));threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.app.camera.start({})
        def cleanup():
            self.app.camera.stop();self.server.shutdown();self.server.server_close();self.app.pool.shutdown(wait=True)
        self.addCleanup(cleanup)
    def test_latest_and_unchanged_responses_have_bounded_payload_and_age(self):
        with urlopen(self.base+'/api/camera/frame.jpg') as response:
            self.assertEqual(response.status,200);self.assertEqual(response.headers['Content-Type'],'image/jpeg')
            seq=int(response.headers['X-Frame-Sequence']);self.assertLess(float(response.headers['X-Frame-Age-Ms']),100)
            data=response.read();self.assertEqual(len(data),int(response.headers['Content-Length']))
            Image.open(BytesIO(data)).verify()
        packet=self.app.camera.latest_preview()
        with patch.object(self.app.camera,'latest_preview',return_value=packet):
            seq=packet['sequence']
            with urlopen(self.base+f'/api/camera/frame.jpg?after={seq}',timeout=3) as response:
                self.assertEqual(response.status,204);self.assertEqual(response.read(),b'')
    def test_slow_viewer_skips_intermediate_frames(self):
        sequences=[]
        for _ in range(4):
            with urlopen(self.base+'/api/camera/frame.jpg') as response:
                sequence=int(response.headers['X-Frame-Sequence']);response.read();sequences.append(sequence)
            time.sleep(.15)
        self.assertTrue(all(b-a>=3 for a,b in zip(sequences,sequences[1:])),sequences)
        self.assertLessEqual(self.app.camera.sequence-sequences[-1],6)
    def test_legacy_mjpeg_can_buffer_while_latest_requests_do_not(self):
        stream=urlopen(self.base+'/api/camera/stream',timeout=3)
        try:
            shown=[]
            for _ in range(6):
                self.assertIn(b'--frame',stream.readline());headers={}
                while True:
                    line=stream.readline()
                    if line==b'\r\n':break
                    k,v=line.decode().split(':',1);headers[k.lower()]=v.strip()
                data=stream.read(int(headers['content-length']));stream.read(2)
                shown.append(Image.open(BytesIO(data)).getpixel((0,0))[0]);time.sleep(.15)
            legacy_lag=self.app.camera.sequence-shown[-1]
        finally:stream.close()
        with urlopen(self.base+'/api/camera/frame.jpg') as response:
            seq=int(response.headers['X-Frame-Sequence']);response.read()
        latest_lag=self.app.camera.sequence-seq
        self.assertGreater(legacy_lag,10);self.assertLessEqual(latest_lag,2)
        print(f'Video backlog simulation: legacy={legacy_lag} frames; latest={latest_lag} frames',flush=True)
    def test_stop_and_missing_tracking_do_not_replay_old_frame(self):
        with urlopen(self.base+'/api/tracking/frame.jpg') as response:self.assertEqual(response.status,204)
        self.app.camera.stop()
        with urlopen(self.base+'/api/camera/frame.jpg') as response:self.assertEqual(response.status,204)
    def test_invalid_sequence_is_rejected_and_counter_reset_is_visible(self):
        with self.assertRaises(HTTPError) as error:urlopen(self.base+'/api/camera/frame.jpg?after=bad')
        self.assertEqual(error.exception.code,400)
        with urlopen(self.base+'/api/camera/frame.jpg?after=999999') as response:self.assertEqual(response.status,200)
    def test_sdk_drain_aligns_only_latest_frameset(self):
        source=RealSenseSource.__new__(RealSenseSource);source.maps=None;source.camera={}
        empty=Mock();empty.__bool__=lambda self:False
        first=Mock(name='first');latest=Mock(name='latest');source.pipeline=Mock()
        source.pipeline.wait_for_frames.return_value=first;source.pipeline.poll_for_frames.side_effect=[latest,empty]
        source.align=Mock();frames=source.align.process.return_value
        color=frames.get_color_frame.return_value;depth=frames.get_depth_frame.return_value
        color.get_data.return_value=np.zeros((4,4,3),np.uint8);depth.get_data.return_value=np.ones((4,4),np.uint16)
        color.get_frame_number.return_value=42;color.get_timestamp.return_value=100
        self.assertEqual(source.read()['number'],42);source.align.process.assert_called_once_with(latest)

if __name__=='__main__':unittest.main()
