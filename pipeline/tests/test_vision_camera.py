"""RGB-D contract, persistence and API without physical cameras or models."""
from io import BytesIO
import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import pytest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam6d_ui.camera.service import CameraService
from sam6d_ui.inputs import options
from sam6d_ui.server import App, handler


class Source:
    opened = 0
    def __init__(self, config):
        type(self).opened += 1
        self.config = config
        self.number = 0
        self.closed = False
    def read(self):
        time.sleep(.025)
        self.number += 1
        width, height = self.config['width'], self.config['height']
        return {'rgb': np.full((height, width, 3), 140, dtype=np.uint8),
                'depth': np.full((height, width), 1000, dtype=np.uint16),
                'camera': {'cam_K': [500., 0, width/2, 0, 500., height/2, 0, 0, 1],
                           'depth_scale': .1, 'dist_coeffs': [0.]*5},
                'number': self.number, 'timestamp': time.time(), 'device_timestamp_ms': self.number*25}
    def close(self):
        self.closed = True


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.camera = CameraService(self.temp.name, Source)
        self.addCleanup(self.camera.stop)

    def start(self):
        return self.camera.start({'width': 640, 'height': 480, 'fps': 15})

    def test_capture_before_camera_start_reports_required_action_without_waiting(self):
        with patch.object(self.camera.condition, 'wait_for') as wait:
            with self.assertRaisesRegex(RuntimeError, 'Open camera'):
                self.camera.capture()
            wait.assert_not_called()

    def test_camera_failure_is_preserved_when_capture_is_requested(self):
        self.camera.error = 'Device ocupado'
        with self.assertRaisesRegex(RuntimeError, 'Device ocupado'):
            self.camera.capture()

    def test_one_source_shared_by_preview_and_lossless_capture(self):
        before = Source.opened
        self.start()
        self.start()
        sequence, jpeg = self.camera.preview(-1)
        self.assertGreater(sequence, 0)
        with Image.open(BytesIO(jpeg)) as image:
            self.assertEqual(image.size, (640, 480))
        captured = self.camera.capture()
        folder = self.camera.capture_folder(captured['id'])
        depth = np.asarray(Image.open(folder / 'depth.png'))
        self.assertEqual(depth.dtype, np.uint16)
        self.assertTrue(np.all(depth == 1000))
        self.assertEqual(json.loads((folder / 'camera.json').read_text())['depth_scale'], .1)
        self.assertEqual(Source.opened, before+1)
        self.camera.stop()
        with self.assertRaises(RuntimeError):
            self.camera.capture()

    def test_recording_keeps_rgb_depth_timestamps_and_video(self):
        self.start()
        info = self.camera.start_recording()
        deadline = time.monotonic()+3
        while self.camera.status()['recording']['frames'] < 3 and time.monotonic() < deadline:
            time.sleep(.02)
        ended = self.camera.stop_recording()
        folder = self.camera.recording_folder(info['id'])
        rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(len(rows), ended['frames'])
        self.assertEqual(ended['status'], 'complete')
        self.assertGreater((folder / 'video.mp4').stat().st_size, 0)
        self.assertEqual([row['index'] for row in rows], list(range(len(rows))))
        self.assertTrue(all((folder / row['rgb']).is_file() and (folder / row['depth']).is_file() for row in rows))
        self.assertTrue(all(row['timestamp'] > 0 for row in rows))

    def test_shutdown_finalizes_recording_and_rejects_traversal(self):
        self.start()
        info = self.camera.start_recording()
        time.sleep(.1)
        self.camera.stop()
        saved = json.loads((Path(self.temp.name)/'videos'/info['id']/'recording.json').read_text())
        self.assertEqual(saved['status'], 'complete')
        for value in ('../x', '0'*31, 'g'*32):
            with self.assertRaises(ValueError):
                self.camera.capture_folder(value)

    def test_low_disk_and_invalid_profile_never_start_recording(self):
        with self.assertRaises(ValueError):
            self.camera.start({'width': 2, 'height': 2})
        self.start()
        with patch('sam6d_ui.camera.service.shutil.disk_usage', return_value=type('Disk', (), {'free': 1})()):
            with self.assertRaisesRegex(ValueError, '256'):
                self.camera.start_recording()
        self.assertIsNone(self.camera.recording)

    def test_stale_capture_is_not_reused(self):
        self.start()
        with self.camera.condition:
            self.camera.latest['received'] -= 100
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                self.camera.frame()

    def test_foundationpose_requires_real_rgbd_and_never_silently_becomes_2d(self):
        for value in ({'pose_backend':'foundationpose','mode':'rgb'},
                      {'pose_backend':'foundationpose','mode':'rgbd','device':'cpu'},
                      {'pose_backend':'unknown'}, []):
            with self.assertRaises(ValueError):
                options(value)
        self.assertTrue(options({'pose_backend':'foundationpose','mode':'rgbd'})['pose'])


@pytest.mark.usefixtures("app_dataset")
class CameraApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = App(self.root / 'data', self.root, self.root / 'SAM6D/captures')
        self.app.camera = CameraService(self.app.camera.root, Source)
        self.app._run = lambda *args: None
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        def close():
            self.app.camera.stop()
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(3)
            self.app.pool.shutdown(wait=True)
        self.addCleanup(close)

    def api(self, path, payload=None):
        request = Request(self.base+path, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_live_job_uses_server_camera_and_selected_cad(self):
        self.api('/api/camera/start', {})
        cad = self.root/'object.stl'
        cad.write_bytes(trimesh.creation.box().export(file_type='stl'))
        job = self.api('/api/jobs', {'live_camera': True, 'server_files': {'cad':str(cad)}, 'options':{'mode':'rgbd'}})
        folder = self.app.job_folder(job['id'])
        self.assertEqual(folder.parent.name, 'sam6d')
        inputs = json.loads((folder/'inputs.json').read_text())
        self.assertIn('capture_id', inputs)
        self.assertEqual(inputs['server_paths']['cad'], str(cad))
        self.assertEqual(json.loads((folder/'camera.json').read_text())['depth_scale'], .1)
        self.assertTrue(np.all(np.asarray(Image.open(folder/'depth.png')) == 1000))

    def test_continuous_detection_stops_before_capture_when_disk_is_low(self):
        with patch('sam6d_ui.server.shutil.disk_usage') as usage:
            usage.return_value.free = 128 * 1024**2
            with self.assertRaises(HTTPError) as error:
                self.api('/api/jobs', {'live_camera': True, 'continuous': True})
            self.assertEqual(error.exception.code, 400)
            self.assertIn('256 MiB', error.exception.read().decode())
        self.assertEqual(self.app.jobs, {})
        self.assertFalse((self.app.camera.root/'images').exists())

    def test_default_capture_root_is_independent_of_job_directory(self):
        with patch('sam6d_ui.server.CameraService') as camera:
            app = App(self.root/'isolated-results', self.root)
            try:
                expected = Path(sys.modules['sam6d_ui.server'].__file__).resolve().parents[2]/'captures'
                camera.assert_called_once_with(expected)
            finally:
                app.pool.shutdown(wait=True)

    def test_images_and_videos_use_capture_root_and_existing_download_urls(self):
        self.assertEqual(self.app.camera.root, self.root/'SAM6D/captures')
        self.api('/api/camera/start', {})
        capture = self.api('/api/camera/capture', {})
        folder = self.app.camera.capture_folder(capture['id'])
        self.assertEqual(folder.parent.name, 'images')
        self.assertTrue((folder/'rgb.png').is_file())
        record = self.api('/api/camera/record/start', {})
        deadline = time.monotonic()+3
        while self.app.camera.status()['recording']['frames'] < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        self.api('/api/camera/record/stop', {})
        self.assertEqual(self.app.camera.recording_folder(record['id']).parent.name, 'videos')
        self.assertIn(record['id'], [r['id'] for r in self.api('/api/camera/recordings')['recordings']])
        for category, identity, name in [('captures',capture['id'],'rgb.png'),('recordings',record['id'],'video.mp4')]:
            with urlopen(self.base+'/camera-files/'+category+'/'+identity+'/'+name) as response:
                self.assertGreater(len(response.read()),0)

    def test_media_range_and_path_boundaries(self):
        self.api('/api/camera/start', {})
        capture = self.api('/api/camera/capture', {})
        url = self.base+'/camera-files/captures/'+capture['id']+'/rgb.png'
        with urlopen(Request(url, headers={'Range':'bytes=0-7'})) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'\x89PNG\r\n\x1a\n')
        for path in ('/camera-files/../outside.json', '/files/'+'a'*32+'/../../outside.json'):
            with self.assertRaises(HTTPError) as error:
                self.api(path)
            self.assertEqual(error.exception.code, 404)

    def test_unavailable_foundationpose_rejects_before_scheduling(self):
        self.app.foundationpose_status = {'available':False,'reason':'Missing GPU dependency'}
        with self.assertRaises(HTTPError) as error:
            self.api('/api/jobs', {'options':{'pose_backend':'foundationpose','mode':'rgbd'}})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.app.jobs, {})


if __name__ == '__main__':
    unittest.main()
