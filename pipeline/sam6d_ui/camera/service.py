"""Shared video, atomic captures and timestamped RGB-D recording."""
from io import BytesIO
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from .realsense import RealSenseSource
from settings import profile


def identifier(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid capture or recording ID.')
    return value


class CameraService:
    def __init__(self, root, source_factory=RealSenseSource):
        self.config = profile('interface')['camera']
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_factory = source_factory
        self.condition = threading.Condition(threading.RLock())
        self.operations = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.latest = None
        self.jpeg = None
        self.sequence = 0
        self.error = None
        self.frame_warning = None
        self.recording = None
        self.last_recording = None
        self.options = {}

    def status(self):
        with self.condition:
            return {'running': bool(self.thread and self.thread.is_alive() and not self.stop_event.is_set()),
                    'frame_number': self.sequence, 'frame_age_s': time.monotonic() - self.latest['received'] if self.latest else None,
                    'error': self.error, 'frame_warning': self.frame_warning, 'options': self.options,
                    'camera': self.latest['camera'] if self.latest else None,
                    'recording': self._recording_info(self.recording) if self.recording else None,
                    'last_recording': self.last_recording}

    def start(self, options):
        if not isinstance(options, dict):
            raise ValueError('Invalid camera configuration.')
        width, height, fps = (options.get('width', self.config['width']), options.get('height', self.config['height']), options.get('fps', self.config['fps']))
        if (width, height) not in ((640, 480), (1280, 720)) or fps not in (15, 30):
            raise ValueError('Utiliza 640×480 o 1280×720 y 15 o 30 FPS.')
        with self.operations:
            if self.thread and self.thread.is_alive():
                return self.status()
            self.options = {'serial': options.get('serial'), 'width': width, 'height': height, 'fps': fps}
            self.stop_event.clear()
            with self.condition:
                self.latest = self.jpeg = None
                self.error = None
                self.frame_warning = None
                self.sequence = 0
            # Opening here lets the browser report a device-busy error.
            source = self.source_factory(self.options)
            self.thread = threading.Thread(target=self._acquire, args=(source,), name='RealSense-RGBD', daemon=True)
            self.thread.start()
            self.frame()
            return self.status()

    def _acquire(self, source):
        timeouts = 0
        try:
            while not self.stop_event.is_set():
                if source is None:
                    try:
                        source = self.source_factory(dict(self.options))
                    except Exception as warning:
                        with self.condition:
                            self.frame_warning = 'Reconnecting camera: '+str(warning)
                            self.condition.notify_all()
                        self.stop_event.wait(.5)
                        continue
                    if self.stop_event.is_set():break
                try:
                    frame = source.read()
                except (TimeoutError, ConnectionError) as warning:
                    timeouts += 1
                    reconnect = isinstance(warning, ConnectionError) or timeouts >= 3
                    with self.condition:
                        self.frame_warning = ('Reconnecting camera: ' if reconnect else '')+str(warning)
                        self.condition.notify_all()
                    if reconnect:
                        try:source.close()
                        except Exception:pass  # A stopped/disconnected SDK pipeline may reject stop().
                        source = None
                        timeouts = 0
                    self.stop_event.wait(.05)
                    continue
                timeouts = 0
                frame['received'] = time.monotonic()
                output = BytesIO()
                preview = Image.fromarray(frame['rgb'])
                preview.thumbnail((self.config['preview_width'], self.config['preview_height']))
                preview.save(output, format='JPEG', quality=self.config['jpeg_quality'])
                with self.condition:
                    # Pin automatic recovery to the originally opened device.
                    if not self.options.get('serial') and frame['camera'].get('serial'):
                        self.options['serial'] = frame['camera']['serial']
                    self.frame_warning = None
                    self.latest = frame
                    self.sequence += 1
                    self.jpeg = output.getvalue()
                    self.condition.notify_all()
                with self.operations:
                    if self.recording is not None:
                        self._write_frame(frame)
        except Exception as error:
            with self.condition:
                self.error = str(error)
                self.condition.notify_all()
        finally:
            try:
                if source is not None:source.close()
            finally:
                with self.operations:
                    if self.recording:
                        self.stop_recording(error=self.error or None)
                self.stop_event.set()
                with self.condition:
                    self.condition.notify_all()

    def stop(self):
        self.stop_event.set()
        thread = self.thread
        if thread and thread is not threading.current_thread():
            thread.join(5)
            if thread.is_alive():
                raise RuntimeError('The camera is still closing; wait before reopening it.')
        return self.status()

    def frame(self, timeout=3):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            if not self.thread or not self.thread.is_alive() or self.stop_event.is_set():
                raise RuntimeError('The camera is inactive. Click Open camera before analysing a frame.')
            if not self.condition.wait_for(lambda: self.latest is not None or self.error or self.stop_event.is_set(), timeout):
                raise RuntimeError('No frame has arrived from RealSense.')
            if self.error or not self.thread or not self.thread.is_alive() or self.stop_event.is_set():
                raise RuntimeError(self.error or 'The camera is inactive.')
            if self.latest is None or time.monotonic() - self.latest['received'] > 2:
                raise RuntimeError('The latest frame is stale; check the camera.')
            return {**self.latest, 'rgb': self.latest['rgb'].copy(), 'depth': self.latest['depth'].copy()}

    def latest_preview(self):
        with self.condition:
            if not self.jpeg or not self.latest or self.error or self.stop_event.is_set() or not self.thread or not self.thread.is_alive():return None
            age=max(0,time.monotonic()-self.latest['received'])
            if age>1:return None
            return {'sequence':self.sequence,'jpeg':self.jpeg,'age_s':age,'timestamp':self.latest['timestamp']}

    def preview(self, previous, timeout=3):
        with self.condition:
            self.condition.wait_for(lambda: self.jpeg is not None and self.sequence != previous or self.stop_event.is_set() or self.error, timeout)
            if self.stop_event.is_set() or self.error or self.sequence == previous or self.jpeg is None:
                return None
            return self.sequence, self.jpeg

    def capture(self):
        frame = self.frame()
        if not np.any(frame['depth'] > 0):
            raise RuntimeError('Depth contains no valid measurements.')
        capture_id = uuid.uuid4().hex
        folder = self.root / 'images' / capture_id
        folder.mkdir(parents=True)
        Image.fromarray(frame['rgb']).save(folder / 'rgb.png')
        Image.fromarray(frame['depth']).save(folder / 'depth.png')
        (folder / 'camera.json').write_text(json.dumps(frame['camera'], indent=2))
        metadata = {'id': capture_id, 'timestamp': frame['timestamp'], 'frame_number': frame['number'],
                    'device_timestamp_ms': frame['device_timestamp_ms'], 'source': 'intel_realsense',
                    'rgb': 'rgb.png', 'depth': 'depth.png', 'camera': 'camera.json'}
        (folder / 'capture.json').write_text(json.dumps(metadata, indent=2))
        return metadata

    def capture_folder(self, capture_id):
        folder = self.root / 'images' / identifier(capture_id)
        if not (folder / 'capture.json').is_file():
            raise ValueError('Capture not found.')
        return folder

    def recording_folder(self, recording_id):
        folder = self.root / 'videos' / identifier(recording_id)
        metadata = folder / 'recording.json'
        if not metadata.is_file():
            raise ValueError('Recording not found.')
        data = json.loads(metadata.read_text())
        if data.get('status') not in ('complete', 'interrupted') or not data.get('frames'):
            raise ValueError('Stop a recording containing frames before analysing it.')
        return folder

    @staticmethod
    def _recording_info(recording):
        return {key: value for key, value in recording.items() if key not in ('writer', 'index', 'folder')}

    def start_recording(self):
        frame = self.frame()
        with self.operations:
            if self.recording:
                raise ValueError('A recording is already in progress.')
            if shutil.disk_usage(self.root).free < 256 * 1024**2:
                raise ValueError('At least 256 MiB must be free to start a recording.')
            record_id = uuid.uuid4().hex
            folder = self.root / 'videos' / record_id
            folder.mkdir(parents=True)
            width, height = frame['rgb'].shape[1], frame['rgb'].shape[0]
            writer = cv2.VideoWriter(str(folder / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     self.options['fps'], (width, height))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError('Could not open the MP4 video encoder.')
            try:
                index = (folder / 'frames.jsonl').open('w')
                (folder / 'camera.json').write_text(json.dumps(frame['camera'], indent=2))
            except BaseException:
                writer.release()
                raise
            self.recording = {'id': record_id, 'status': 'recording', 'frames': 0,
                              'started_at': time.time(), 'fps': self.options['fps'],
                              'width': width, 'height': height, 'rgb_video': 'video.mp4',
                              'frame_index': 'frames.jsonl', 'camera': 'camera.json',
                              'writer': writer, 'index': index, 'folder': folder}
            self._save_recording()
            return self._recording_info(self.recording)

    def _save_recording(self):
        target = self.recording['folder'] / 'recording.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(self._recording_info(self.recording), indent=2, allow_nan=False))
        temp.replace(target)

    def _write_frame(self, frame):
        record = self.recording
        if shutil.disk_usage(self.root).free < 192 * 1024**2 or time.time() - record['started_at'] > 600:
            self.stop_recording(error='Recording stopped because of free space or the ten-minute limit.')
            return
        number = record['frames']
        folder = record['folder'] / 'frames' / f'{number:06d}'
        folder.mkdir(parents=True)
        Image.fromarray(frame['rgb']).save(folder / 'rgb.png')
        Image.fromarray(frame['depth']).save(folder / 'depth.png')
        record['writer'].write(cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR))
        row = {'index': number, 'timestamp': frame['timestamp'], 'device_timestamp_ms': frame['device_timestamp_ms'],
               'frame_number': frame['number'], 'rgb': f'frames/{number:06d}/rgb.png',
               'depth': f'frames/{number:06d}/depth.png'}
        record['index'].write(json.dumps(row) + '\n')
        record['index'].flush()
        record['frames'] += 1
        # Persist the index even if the process is interrupted.
        if number % 15 == 0:
            self._save_recording()

    def stop_recording(self, error=None):
        with self.operations:
            if not self.recording:
                raise ValueError('No active recording.')
            record = self.recording
            record['writer'].release()
            record['index'].close()
            record.update(status='interrupted' if error else 'complete', error=error,
                          stopped_at=time.time(), duration_s=time.time() - record['started_at'])
            self._save_recording()
            self.last_recording = self._recording_info(record)
            self.recording = None
            return self.last_recording
