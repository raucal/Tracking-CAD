"""Capture/estimate/preview/execute lifecycle for one robot alignment task."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .config import DEFAULT, load
from .geometry import build_plan, from_pose, difference
from .robot import RTDERobot


class AlignmentService:
    routes = {'/api/alignment/connect', '/api/alignment/disconnect', '/api/alignment/plan',
              '/api/alignment/execute', '/api/alignment/stop'}

    def __init__(self, app, settings_file=DEFAULT, robot_factory=RTDERobot):
        self.app = app
        self.settings_file = Path(settings_file)
        self.robot_factory = robot_factory
        self.robot = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='CAD-robot')
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.plan = None
        self.state = {'state': 'disconnected', 'message': 'Robot disconnected. The application does not connect automatically.'}
        self.closed = False

    def status(self):
        with self.lock:
            try:
                cfg, _ = load(self.settings_file)
                config = {'robot_ip': cfg['robot']['ip'], 'offset_m': cfg['alignment']['offset_m'],
                          'motion_enabled': cfg['robot']['motion_enabled'],
                          'calibration_confirmed': cfg['robot']['calibration_confirmed'],
                          'camera_serial': cfg['robot']['camera_serial'],
                          'plane_normal_cad': cfg['alignment']['plane_normal_cad'],
                          'speed_m_s': cfg['motion']['speed_m_s']}
                error = None
            except Exception as exc:config = {};error = str(exc)
            return {**self.state, 'connected': bool(self.robot and self.robot.connected),
                    'settings_file': str(self.settings_file), 'config': config, 'settings_error': error,
                    'plan': self.plan, 'busy': self.state['state'] in ('connecting','estimating','moving','stopping')}

    def _update(self, state, message, **extra):
        with self.lock:self.state = {'state': state, 'message': message, **extra}

    def _audit(self, folder, name, data):
        with (folder / name).open('x') as out:json.dump(data, out, indent=2, allow_nan=False)

    def action(self, route, request):
        if route == '/api/alignment/stop':
            # Cancellation is visible before taking locks or issuing RTDE calls.
            self.cancel.set()
            with self.lock:
                moving = self.state['state'] in ('moving', 'stopping')
                active = self.state['state'] in ('connecting', 'estimating', 'moving', 'stopping')
                self.plan = None
                if active:self.state.update(state='stopping', message='Cancelling the operation and stopping the robot…')
            if self.robot:
                self.robot.cancel.set()
                if moving:self.robot.stop()
            if not active:self._update('stopped', 'Operation cancelled. Calculate a new plan.')
            return self.status()
        with self.lock:
            if self.closed:raise RuntimeError('Alignment service closed.')
            if self.status()['busy']:raise ValueError('Wait for the operation to finish or click Stop robot.')
            if route == '/api/alignment/connect':
                self.plan = None;self.cancel.clear();self._update('connecting', 'Conectando RTDE…')
                self.pool.submit(self._connect)
            elif route == '/api/alignment/disconnect':
                if self.robot:self.robot.disconnect()
                self.plan = None;self._update('disconnected', 'Robot desconectado.')
            elif route == '/api/alignment/plan':
                if not self.robot or not self.robot.connected:raise ValueError('Connect the robot first.')
                cfg, cfg_hash = load(self.settings_file)
                if cfg != self.robot.cfg:raise ValueError('Settings changed. Disconnect and reconnect the robot.')
                self._validate_request(request)
                self.plan = None;self.cancel.clear();self.robot.cancel.clear()
                self._update('estimating', 'Capturando RGB-D y calculando SAM6D + FoundationPose…')
                self.pool.submit(self._make_plan, request, cfg, cfg_hash)
            elif route == '/api/alignment/execute':
                if not self.plan or request.get('plan_id') != self.plan['id']:
                    raise ValueError('Select the current plan before moving the robot.')
                plan = self.plan
                cfg, cfg_hash = load(self.settings_file)
                if cfg_hash != plan['settings_sha256']:raise ValueError('Settings changed; calculate a new plan.')
                self._authorize_motion(cfg, plan)
                self.plan = None  # Single-use even on failure or cancellation.
                self.cancel.clear();self.robot.cancel.clear()
                self._update('moving', 'Moving the TCP to the previewed target…', plan_id=plan['id'])
                self.pool.submit(self._execute, plan)
            else:raise ValueError('Unknown alignment action.')
        return self.status()

    def _connect(self):
        try:
            cfg, _ = load(self.settings_file)
            if self.robot:self.robot.disconnect()
            self.robot = self.robot_factory(cfg)
            self.robot.connect()
            if self.cancel.is_set():
                self.robot.disconnect();raise RuntimeError('Connection cancelled.')
            self._update('ready', 'Robot connected and stationary. Capture and calculate a target.')
        except Exception as exc:self._update('error', str(exc))

    def _validate_request(self, request):
        if not request.get('live_camera') or request.get('recording_id') or request.get('manual_bbox'):
            raise ValueError('Alignment requires a fresh RealSense capture and automatic SAM initialization.')
        if self.app.tracker.busy():raise ValueError('Stop tracking before aligning the robot.')
        if not self.app.camera.status()['running']:raise ValueError('Open the camera before alignment.')
        if not request.get('cad') and not request.get('server_files') and not request.get('auto_assembly'):
            raise ValueError('Select a CAD or assembly.')

    def _authorize_motion(self, cfg, plan):
        if not cfg['robot']['motion_enabled'] or not cfg['robot']['calibration_confirmed']:
            raise ValueError('Configure and verify calibration and enable motion_enabled in alignment_pose_cad/settings.json.')
        if not cfg['robot']['camera_serial']:
            raise ValueError('Set camera_serial to identify the calibrated camera.')
        if cfg['robot']['camera_serial'] != plan['camera_serial']:
            raise ValueError('The capture camera is not the calibrated camera.')
        if time.time() - plan['planned_at'] > cfg['motion']['max_plan_age_s']:
            raise ValueError('The plan expired; capture and calculate again.')
        if time.time() - plan['capture_timestamp'] > cfg['motion']['max_capture_age_s']:
            raise ValueError('The capture expired; calculate a new plan.')
        if self.app.tracker.busy():raise ValueError('Stop tracking before execution.')

    def _make_plan(self, request, cfg, cfg_hash):
        folder = None
        try:
            with self.app.submission_lock:
                with self.app.lock:
                    if any(j['status'] in ('queued','running') for j in self.app.jobs.values()):
                        raise ValueError('Wait for pending analyses to finish.')
                reference = self.robot.snapshot(stationary=True)
                # Wait for a newer camera frame, bracketed by stationary TCP reads.
                with self.app.camera.condition:
                    sequence = self.app.camera.sequence
                    ok = self.app.camera.condition.wait_for(
                        lambda: self.app.camera.sequence > sequence or self.cancel.is_set(), timeout=3)
                if not ok or self.cancel.is_set():raise RuntimeError('Capture cancelled or no new camera frames.')
                captured = self.app.camera.capture()
                source = self.app.camera.capture_folder(captured['id'])
                serial = json.loads((source / 'camera.json').read_text()).get('serial', '')
                self.robot.check_reference(reference)
                if time.time() - captured['timestamp'] > cfg['motion']['max_frame_age_s']:
                    raise RuntimeError('The RGB-D capture is not recent enough.')
                if cfg['robot']['camera_serial'] and serial != cfg['robot']['camera_serial']:
                    raise ValueError('The active camera does not match camera_serial in settings.json.')
                options = {**request.get('options', {}), 'mode': 'rgbd', 'device': 'cuda',
                           'pose': True, 'pose_backend': 'foundationpose', 'segmentor': 'sam'}
                if request.get('auto_assembly'):options['cad_units'] = 'mm'
                prepared = {'options': options}
                for key in ('rgb','depth','camera'):
                    p = source / {'rgb':'rgb.png','depth':'depth.png','camera':'camera.json'}[key]
                    prepared[key] = {'name': p.name, 'data': base64.b64encode(p.read_bytes()).decode()}
                if request.get('auto_assembly'):
                    from sam6d_ui.assembly_identification import assembly_catalog
                    manifest, parts = assembly_catalog(request['auto_assembly'])
                    cad = manifest.parent / parts[0]['source_stl']
                elif request.get('server_files'):
                    cad = self.app.files.resolve(request['server_files'].get('cad', ''))
                else:cad = None
                if cad is not None:
                    # Match the regular upload contract before reading server files.
                    if not cad.is_file() or cad.suffix.lower() not in ('.stl','.obj','.ply') or cad.stat().st_size>50*1024**2:
                        raise ValueError('Select a valid CAD mesh up to 50 MiB.')
                    prepared['cad'] = {'name':cad.name,'data':base64.b64encode(cad.read_bytes()).decode()}
                else:prepared['cad'] = request['cad']
                # Prepare a normal inference job and mark this fresh capture; never
                # accept an arbitrary historical result/pose supplied by a browser.
                from sam6d_ui.inputs import prepare_upload
                job_id = uuid.uuid4().hex
                folder = self.app.jobs_root / 'foundationpose' / job_id
                inputs = prepare_upload(prepared, folder)
                inputs.update(capture_id=captured['id'], backend='foundationpose')
                if request.get('auto_assembly'):inputs['auto_assembly'] = request['auto_assembly']
                (folder/'inputs.json').write_text(json.dumps(inputs, indent=2))
                self._audit(folder, 'robot_capture.json', {'reference':reference,'capture':captured,
                            'camera_serial':serial,'settings_sha256':cfg_hash})
                with self.app.lock:
                    self.app.jobs[job_id] = {'id':job_id,'status':'queued','message':'CAD alignment','inputs':inputs}
                future = self.app.pool.submit(self.app._run, job_id, folder, inputs)
                deadline = time.monotonic() + cfg['motion']['inference_timeout_s']
                while not future.done():
                    if self.cancel.wait(.1):raise RuntimeError('Estimation cancelled; the robot will not move.')
                    self.robot.check_reference(reference)
                    if time.monotonic() > deadline:raise RuntimeError('Inference timed out.')
                future.result()
                self.robot.check_reference(reference)
                job = self.app.job(job_id)
                result = job.get('result', {})
                if job['status'] != 'complete':raise RuntimeError(job.get('message','Inference failed.'))
                pose = result.get('pose') or {}
                if result.get('backend')!='foundationpose' or not result.get('recognized') or not pose.get('poses'):
                    raise ValueError('No accepted detection and FoundationPose pose. No target was created.')
                best = pose['poses'][pose.get('best_index', 0)]
                bounds = np.asarray(inputs['cad_bounds_mm']) + np.asarray(inputs['input_center_mm'])
                if inputs.get('auto_assembly'):
                    part = next(p for p in parts if p['key']==result['cad_key'])
                    bounds = part['bounds_local_mm']
                plan = build_plan(best['T_camera_from_input_cad_mm'], reference['tcp_pose_m_rad'], cfg, bounds, result['cad_key'])
                self.robot.validate_path(plan, reference)
                if self.cancel.is_set():raise RuntimeError('Plan cancelled.')
                plan.update(id=uuid.uuid4().hex, job_id=job_id, capture_id=captured['id'],
                            reference=reference, settings_sha256=cfg_hash, camera_serial=serial,
                            capture_timestamp=captured['timestamp'], planned_at=time.time(),
                            expires_in_s=cfg['motion']['max_plan_age_s'],
                            image_url=f'/files/{job_id}/pose_overlay.png',
                            artifact_folder=str(folder), accuracy_validated=False)
                self._audit(folder, 'robot_alignment_plan.json', plan)
                with self.lock:
                    if self.cancel.is_set():raise RuntimeError('Plan cancelled.')
                    self.plan = plan
                    self._update('planned', 'Target calculated. Review the image and matrix before execution.')
        except Exception as exc:
            self.plan = None
            self._update('error', str(exc))
            if folder is not None and folder.is_dir():
                try:self._audit(folder, 'robot_alignment_error.json', {'error':str(exc),'timestamp':time.time()})
                except OSError:pass

    def _execute(self, plan):
        try:
            with self.app.submission_lock:
                cfg, digest = load(self.settings_file)
                if digest != plan['settings_sha256']:raise ValueError('Settings changed; plan invalidated.')
                self._authorize_motion(cfg, plan)
                if self.cancel.is_set():raise RuntimeError('Movement cancelled.')
                result = self.robot.move(plan, plan['reference'])
                self._audit(Path(plan['artifact_folder']), 'robot_alignment_execution.json', result)
                self._update('completed', 'TCP aligned. Capture again to verify visually.', result=result)
        except Exception as exc:
            self._update('error', str(exc))
            try:
                self._audit(Path(plan['artifact_folder']), 'robot_alignment_execution.json',
                            {'status':'error','error':str(exc),'timestamp':time.time()})
            except OSError:pass

    def close(self):
        self.closed = True
        self.cancel.set()
        if self.robot:
            self.robot.cancel.set()
            if self.state['state'] in ('moving','stopping'):
                try:self.robot.stop()
                except Exception as exc:self._update('error', 'Failed to send stop: '+str(exc))
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.robot:self.robot.disconnect()
