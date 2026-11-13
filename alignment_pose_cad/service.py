"""Capture/estimate/preview/execute lifecycle for one robot alignment task."""
import base64
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .config import DEFAULT, load
from .geometry import build_plan, build_demo_plan, from_pose, difference, rotation_policy
from .robot import RTDERobot
from . import calibration, skill_demos


def alignment_target(request):
    """Normalized target choice: CAD face standoff (default) or a recorded skill demo."""
    target = request.get('alignment_target') or {'mode': 'cad_face'}
    if not isinstance(target, dict) or target.get('mode') not in ('cad_face', 'skill_demo'):
        raise ValueError('Invalid alignment target.')
    if target['mode'] == 'cad_face':
        if set(target) != {'mode'}:raise ValueError('Invalid alignment target.')
        return {'mode': 'cad_face'}
    demo_id = target.get('demo_id')
    if set(target) - {'mode', 'demo_id'} or (demo_id is not None and (not isinstance(demo_id, str) or not demo_id)):
        raise ValueError('Invalid skill demo selection.')
    return {'mode': 'skill_demo', 'demo_id': demo_id}

class AlignmentService:
    routes = {'/api/alignment/connect', '/api/alignment/disconnect', '/api/alignment/plan',
              '/api/alignment/check-path',
              '/api/alignment/execute', '/api/alignment/stop', '/api/alignment/speed',
              '/api/alignment/allow-rotation', '/api/alignment/invalidate', '/api/alignment/execute-skill'}
    BUSY = ('connecting', 'estimating', 'checking', 'moving', 'stopping', 'skill')

    def __init__(self, app, settings_file=DEFAULT, robot_factory=RTDERobot, skill_launcher=None):
        self.app = app
        self.settings_file = Path(settings_file)
        self.robot_factory = robot_factory
        self.robot = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='CAD-robot')
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.plan = None
        # Set only after a completed skill-demo alignment; single use.
        self.skill_ready = None
        self.skill_process = None
        self.skill_launcher = skill_launcher or self._launch_skill
        self.state = {'state': 'disconnected', 'message': 'Robot disconnected. The application does not connect automatically.'}
        self.closed = False

    def status(self):
        with self.lock:
            try:
                cfg, digest = load(self.settings_file)
                hand_eye = calibration.describe(calibration.load(cfg['robot']))
                config = {'robot_ip': cfg['robot']['ip'], 'offset_m': cfg['alignment']['offset_m'],
                          'preview_only': cfg['robot']['preview_only'], 'hand_eye': hand_eye,
                          'motion_enabled': cfg['robot']['motion_enabled'],
                          'execution_mode': cfg['robot']['execution_mode'],
                          'target_selection': 'vision_controls',
                          'calibration_confirmed': cfg['robot']['calibration_confirmed'],
                          'camera_serial': cfg['robot']['camera_serial'],
                          'plane_normal_cad': cfg['alignment']['plane_normal_cad'],
                          'speed_m_s': cfg['motion']['speed_m_s'], 'max_speed_m_s': .25,
                          'max_rotation_deg':cfg['motion']['max_rotation_deg'],
                          'settings_sha256': digest,
                          'orientation_mode': cfg['alignment'].get('orientation_mode', 'plane_frame'),
                          'skill_execution_available': bool(cfg['alignment']['skill_python']),
                          'skill_speed_factor': cfg['alignment']['skill_speed_factor']}
                try:
                    bindings = cfg['alignment']['skill_bindings_file']
                    config['skill_demos'] = skill_demos.catalog(bindings) if bindings else []
                except Exception as exc:
                    config['skill_demos'] = [];config['skill_demos_error'] = str(exc)
                error = None
            except Exception as exc:config = {};error = str(exc)
            camera = getattr(self.app, 'camera', None)
            camera_state = camera.status() if camera else {}
            return {**self.state, 'connected': bool(self.robot and self.robot.connected),
                    'active_camera': camera_state.get('camera') if camera_state.get('running') else None,
                    'needs_reconnect': bool(error is None and self.robot and self.robot.connected and cfg != self.robot.cfg),
                    'settings_file': str(self.settings_file), 'config': config, 'settings_error': error,
                    'plan': self.plan, 'busy': self.state['state'] in self.BUSY,
                    'skill_ready': ({k: v for k, v in self.skill_ready.items() if k != 'demo_files'}
                                    if self.skill_ready else None)}

    def _update(self, state, message, **extra):
        with self.lock:self.state = {'state': state, 'message': message, **extra}

    def _audit(self, folder, name, data):
        with (folder / name).open('x') as out:json.dump(data, out, indent=2, allow_nan=False)

    def action(self, route, request):
        if route == '/api/alignment/invalidate':
            with self.lock:
                if self.closed:raise RuntimeError('Alignment service closed.')
                if self.state['state'] in ('moving','stopping','connecting'):
                    raise ValueError('Wait for the active robot operation before changing targets.')
                if self.state['state'] == 'skill':raise ValueError('Wait for the skill to finish or click Stop robot.')
                self.plan = None;self.skill_ready = None
                if self.state['state'] in ('estimating','checking'):
                    self.cancel.set()
                    if self.robot:self.robot.cancel.set()
                else:
                    self._update('ready' if self.robot and self.robot.connected else 'disconnected',
                                 'CAD selection changed. Detect a new pose before execution.')
                return self.status()
        if route == '/api/alignment/stop':
            # Cancellation is visible before taking locks or issuing RTDE calls.
            self.cancel.set()
            with self.lock:
                moving = self.state['state'] in ('moving', 'stopping')
                active = self.state['state'] in self.BUSY
                process = self.skill_process if self.state['state'] == 'skill' else None
                self.plan = None;self.skill_ready = None
                if active:self.state.update(state='stopping', message='Cancelling the operation and stopping the robot…')
            if process is not None:
                # The DINOBOT runner owns RTDE now; it brakes with stopJ on SIGINT.
                try:process.send_signal(signal.SIGINT)
                except (OSError, ValueError):pass
            elif self.robot:
                self.robot.cancel.set()
                if moving:self.robot.stop()
            if not active:self._update('stopped', 'Operation cancelled. Calculate a new plan.')
            return self.status()
        with self.lock:
            if self.closed:raise RuntimeError('Alignment service closed.')
            if self.status()['busy']:raise ValueError('Wait for the operation to finish or click Stop robot.')
            if route != '/api/alignment/execute-skill':self.skill_ready = None
            if route == '/api/alignment/speed':
                self._set_speed(request)
            elif route == '/api/alignment/allow-rotation':
                self._allow_rotation(request)
            elif route == '/api/alignment/connect':
                self.plan = None;self.cancel.clear();self._update('connecting', 'Connecting RTDE…')
                self.pool.submit(self._connect)
            elif route == '/api/alignment/disconnect':
                if self.robot:self.robot.disconnect()
                self.plan = None;self._update('disconnected', 'Robot disconnected.')
            elif route == '/api/alignment/plan':
                if not self.robot or not self.robot.connected:raise ValueError('Connect the robot first.')
                cfg, cfg_hash = load(self.settings_file)
                if cfg != self.robot.cfg:raise ValueError('Settings changed. Disconnect and reconnect the robot.')
                self.plan = None
                self._validate_request(request)
                self.plan = None;self.cancel.clear();self.robot.cancel.clear()
                self._update('estimating', 'Capturing RGB-D and estimating SAM6D + FoundationPose…')
                self.pool.submit(self._make_plan, request, cfg, cfg_hash)
            elif route == '/api/alignment/check-path':
                if not self.robot or not self.robot.connected:
                    raise ValueError('Connect the robot first.')
                if not self.plan or request.get('plan_id') != self.plan['id']:
                    raise ValueError('Calculate and select the current plan first.')
                cfg, digest = load(self.settings_file)
                if digest != self.plan['settings_sha256'] or cfg != self.robot.cfg:
                    raise ValueError('Settings changed; reconnect and calculate a new plan.')
                if time.time() - self.plan['planned_at'] > cfg['motion']['max_plan_age_s']:
                    raise ValueError('The plan expired; capture and calculate again.')
                self.cancel.clear();self.robot.cancel.clear()
                self.plan['controller_path_checked'] = False
                self._update('checking', 'Checking controller limits and IK without motion…')
                self.pool.submit(self._check_path, self.plan)
            elif route == '/api/alignment/execute':
                if not self.plan or request.get('plan_id') != self.plan['id']:
                    raise ValueError('Select the current plan before moving the robot.')
                plan = self.plan
                if plan.get('selection_signature') is not None and request.get('selection_signature') != plan['selection_signature']:
                    raise ValueError('The CAD selection changed; detect the pose again.')
                if alignment_target(request) != plan.get('alignment_target', {'mode': 'cad_face'}):
                    raise ValueError('The alignment target changed; detect the pose again.')
                cfg, cfg_hash = load(self.settings_file)
                if cfg_hash != plan['settings_sha256']:raise ValueError('Settings changed; calculate a new plan.')
                review = {'confirmed': request.get('confirmed') is True}
                self._authorize_motion(cfg, plan, review)
                if cfg['robot']['execution_mode'] == 'supervised_trial':
                    # Store a copy only after validating this specific operator request.
                    plan['trial_review'] = json.loads(json.dumps(review))
                    self._audit(Path(plan['artifact_folder']), 'trial_authorization_'+uuid.uuid4().hex+'.json',
                                {'plan_id':plan['id'], 'settings_sha256':cfg_hash, 'review':review,
                                 'authorized_at':time.time(), 'accuracy_validated':False,
                                 'calibration_status':plan['hand_eye']['status']})
                self.plan = None  # Single-use even on failure or cancellation.
                self.cancel.clear();self.robot.cancel.clear()
                self._update('moving', 'Moving the TCP to the previewed target…', plan_id=plan['id'])
                self.pool.submit(self._execute, plan)
            elif route == '/api/alignment/execute-skill':
                skill = self._authorize_skill(request)
                self.skill_ready = None  # Single use, even on failure.
                self.cancel.clear()
                self._update('skill', f"Running skill {skill['demo_id']} with DINOBOT from the aligned pose…",
                             skill={k: v for k, v in skill.items() if k != 'demo_files'})
                self.pool.submit(self._run_skill, skill)
            else:raise ValueError('Unknown alignment action.')
        return self.status()

    def _authorize_skill(self, request):
        """The aligned pose, demo, settings and robot must be exactly those of the alignment."""
        skill = self.skill_ready
        if not skill or request.get('plan_id') != skill['plan_id']:
            raise ValueError('Complete a skill-demo alignment first; the skill runs only from that aligned pose.')
        if request.get('confirmed') is not True:
            raise ValueError('Explicit confirmation is required to execute the skill.')
        cfg, digest = load(self.settings_file)
        if digest != skill['settings_sha256']:raise ValueError('Settings changed after alignment; align again.')
        if cfg['robot']['preview_only'] or not cfg['robot']['motion_enabled']:
            raise ValueError('Motion is disabled in alignment_pose_cad/settings.json.')
        python = cfg['alignment']['skill_python']
        if not python or not Path(python).is_file():
            raise ValueError('Set alignment.skill_python to the DINOBOT environment interpreter.')
        if time.time() - skill['aligned_at'] > cfg['motion']['max_capture_age_s']:
            raise ValueError('The alignment is too old; the object may have moved. Detect and align again.')
        if self.app.tracker.busy():raise ValueError('Stop tracking before executing the skill.')
        if not self.robot or not self.robot.connected or cfg != self.robot.cfg:
            raise ValueError('The robot connection used for alignment is no longer active; align again.')
        snapshot = self.robot.snapshot(stationary=True)
        distance, angle = difference(from_pose(skill['aligned_tcp_m_rad']), from_pose(snapshot['tcp_pose_m_rad']))
        if distance > cfg['motion']['position_tolerance_m'] or angle > cfg['motion']['rotation_tolerance_deg']:
            raise ValueError(f'The TCP moved {distance*1000:.1f} mm / {angle:.2f}° away from the aligned pose; align again.')
        dt, dr = difference(from_pose(skill['tcp_offset_m_rad']), from_pose(self.robot.control.getTCPOffset()))
        if dt > .0001 or dr > .1:raise ValueError('The active TCP changed after alignment.')
        skill_demos.verify({'file_hashes': skill['demo_files']})
        return {**skill, 'python': python, 'speed_factor': cfg['alignment']['skill_speed_factor'],
                'robot_ip': cfg['robot']['ip'], 'position_tolerance_m': cfg['motion']['position_tolerance_m'],
                'rotation_tolerance_deg': cfg['motion']['rotation_tolerance_deg']}

    def _launch_skill(self, command, log):
        return subprocess.Popen(command, cwd=str(skill_demos.DEMO_REFERENCE.parent), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

    def _run_skill(self, skill):
        folder = Path(skill['artifact_folder'])
        result_file = folder / f"skill_result_{uuid.uuid4().hex}.json"
        outcome = {'status': 'error', 'demo_id': skill['demo_id']}
        try:
            # UR RTDE accepts one control client: hand it over to the DINOBOT runner.
            if self.robot:self.robot.disconnect()
            command = [skill['python'], str(Path(__file__).with_name('skill_runner.py')),
                       '--dinobot', str(skill_demos.DEMO_REFERENCE.parent), '--demos-root', skill['demos_root'],
                       '--demo', skill['demo_id'], '--robot-ip', skill['robot_ip'],
                       '--expected-start', json.dumps(skill['aligned_tcp_m_rad']),
                       '--tolerance-m', repr(skill['position_tolerance_m']),
                       '--tolerance-deg', repr(skill['rotation_tolerance_deg']),
                       '--speed-factor', repr(float(skill['speed_factor'])), '--result', str(result_file)]
            self._audit(folder, 'skill_execution_request_'+uuid.uuid4().hex+'.json',
                        {'command': command, 'requested_at': time.time(),
                         **{k: v for k, v in skill.items() if k != 'demo_files'}})
            with (folder / 'skill_runner.log').open('a') as log:
                process = self.skill_launcher(command, log)
                with self.lock:self.skill_process = process
                process.wait()
            outcome = json.loads(result_file.read_text()) if result_file.is_file() else {
                'status': 'error', 'error': f'The skill runner exited with code {process.returncode} without a result; see skill_runner.log.'}
        except Exception as exc:
            outcome.update(error=str(exc))
        finally:
            with self.lock:self.skill_process = None
        replay = outcome.get('replay') or {}
        detail = (f" ({replay.get('completed_samples')}/{replay.get('total_samples')} samples)" if replay else '')
        if outcome.get('status') == 'completed':
            self._update('completed', f"Skill {skill['demo_id']} completed{detail}. DINOBOT released the robot; "
                         'connect it again before another alignment.', skill_result=outcome)
        else:
            self._update('error', f"Skill {skill['demo_id']} {outcome.get('status', 'error')}{detail}: "
                         + str(outcome.get('error', 'unknown error'))
                         + ' The robot is disconnected; check it and connect again.', skill_result=outcome)

    def _allow_rotation(self, request):
        # Old clients cannot bypass configured limits with per-plan consent.
        raise ValueError('Manual rotation overrides are disabled. Set motion.max_rotation_deg (up to 180) and detect a new target.')

    def _set_speed(self, request):
        """Change only the requested moveL speed, never controller safety settings."""
        cfg, digest = load(self.settings_file)
        if request.get('settings_sha256') != digest:
            raise ValueError('Settings changed; refresh before applying the speed.')
        speed = request.get('speed_m_s')
        if type(speed) not in (int, float) or not np.isfinite(speed) or not 0 < speed <= .25:
            raise ValueError('Speed must be greater than 0 and at most 250 mm/s.')
        connected = bool(self.robot and self.robot.connected)
        if connected:
            if cfg != self.robot.cfg:raise ValueError('Settings changed; disconnect and reconnect the robot.')
            self.robot.snapshot(stationary=True)
        raw = self.settings_file.read_bytes()
        data = json.loads(raw)
        data['motion']['speed_m_s'] = float(speed)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.settings_file.parent, prefix='.alignment_speed_', suffix='.json', delete=False) as out:
                temporary = Path(out.name)
                json.dump(data, out, indent=2, allow_nan=False)
                out.write('\n')
            updated, _ = load(temporary)
            if load(self.settings_file)[1] != digest:
                raise ValueError('Settings changed while applying speed; refresh and retry.')
            os.chmod(temporary, self.settings_file.stat().st_mode & 0o777)
            os.replace(temporary, self.settings_file)
            temporary = None
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
        if connected:self.robot.cfg = updated
        self.plan = None
        self._update('ready' if connected else 'disconnected',
                     f'Speed set to {speed*1000:g} mm/s. Detect the pose again before execution.')

    def _check_path(self, plan):
        try:
            self._check_trial_tcp(self.robot.cfg)
            details = self.robot.validate_path(plan, plan['reference'])
            if self.cancel.is_set():raise RuntimeError('Check cancelled.')
            _, digest = load(self.settings_file)
            if digest != plan['settings_sha256']:raise ValueError('Settings changed; calculate a new plan.')
            report = {'checked_at':time.time(), 'controller_path_checked':True,
                      'collision_checked':False, 'motion_sent':False,
                      'plan_id':plan['id'], 'samples':len(plan['path_samples_m_rad']),
                      'details':details}
            self._audit(Path(plan['artifact_folder']), 'controller_check_'+uuid.uuid4().hex+'.json', report)
            with self.lock:
                if self.cancel.is_set() or self.plan is not plan:raise RuntimeError('Check cancelled.')
                plan.update(controller_path_checked=True, controller_check=report)
                plan.pop('controller_check_error', None)
                plan.pop('controller_check_details', None)
                self._update('planned', 'Controller IK and limits accepted all samples. No motion sent; collisions and calibration accuracy are not validated.')
        except Exception as exc:
            with self.lock:
                plan['controller_path_checked'] = False
                plan['controller_check_error'] = str(exc)
                plan['controller_check_details'] = getattr(exc,'details',None)
                self._update('error', 'Controller check without motion: '+str(exc))
            self._audit(Path(plan['artifact_folder']), 'controller_rejection_'+uuid.uuid4().hex+'.json',
                        {'error':str(exc),'controller_check_details':getattr(exc,'details',None),
                         'plan_id':plan['id'],'motion_sent':False})

    def _connect(self):
        try:
            cfg, _ = load(self.settings_file)
            if self.robot:self.robot.disconnect()
            self.robot = self.robot_factory(cfg)
            self.robot.connect()
            self._check_trial_tcp(cfg)
            if self.cancel.is_set():
                self.robot.disconnect();raise RuntimeError('Connection cancelled.')
            self._update('ready', 'Robot connected and stationary. Capture and calculate a target.')
        except Exception as exc:
            if self.robot:self.robot.disconnect()
            self._update('error', str(exc))

    def _validate_request(self, request):
        if not request.get('live_camera') or request.get('recording_id') or request.get('manual_bbox'):
            raise ValueError('Alignment requires a fresh RealSense capture and automatic SAM initialization.')
        if self.app.tracker.busy():raise ValueError('Stop tracking before aligning the robot.')
        if not self.app.camera.status()['running']:raise ValueError('Open the camera before alignment.')
        if sum(bool(request.get(k)) for k in ('cad','server_files','auto_assembly','target_cad_key')) != 1:
            raise ValueError('Select exactly one CAD source: file, upload, catalog part or automatic assembly identification.')
        if request.get('auto_assembly') and not isinstance(request['auto_assembly'],str):
            raise ValueError('Select an assembly name for automatic identification.')
        if request.get('selection_signature') is not None and (not isinstance(request['selection_signature'],str) or len(request['selection_signature'])>8192):
            raise ValueError('Invalid CAD selection signature.')
        cfg, _ = load(self.settings_file)
        if request.get('target_cad_key'):self._catalog_cad(request['target_cad_key'])
        target = alignment_target(request)
        if target['mode'] == 'skill_demo':
            if not cfg['alignment']['skill_bindings_file']:
                raise ValueError('Configure alignment.skill_bindings_file to align to skill demos.')
            if request.get('cad'):
                raise ValueError('Skill demo alignment needs a catalog CAD: its saved pose uses that CAD frame.')
            if target['demo_id']:
                demo = skill_demos.load(cfg['alignment']['skill_bindings_file'], target['demo_id'])
                if request.get('target_cad_key') and request['target_cad_key'] != demo['cad_key']:
                    raise ValueError(f"Demo {demo['demo_id']} shows {demo['cad_key']}, not the selected {request['target_cad_key']}.")
        hand_eye = calibration.load(cfg['robot'])
        if hand_eye is not None and not calibration.describe(hand_eye)['preview_ready']:
            raise ValueError('matrix_calibration.yaml must declare camera.serial and tcp_offset_m_rad before calculating a target.')

    def _catalog_cad(self, key, manifest_path=None):
        """Resolve a selected/identified canonical mesh and verify its provenance."""
        manifest_path = Path(manifest_path or self.app.manifest_path)
        manifest = json.loads(manifest_path.read_text())
        part = next((p for p in manifest['parts'] if p['key'] == key), None)
        if part is None:raise ValueError('CAD is missing from the prepared manifest: '+str(key))
        path = (manifest_path.parent / part['source_stl']).resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != part.get('source_sha256'):
            raise ValueError('The CAD differs from its prepared source hash: '+str(key))
        return path, content, digest

    def _check_cad_identity(self, plan):
        if not plan.get('cad_key') or not plan.get('cad_sha256') or not plan.get('cad_source_path'):
            raise ValueError('The alignment target has no verified CAD identity.')
        source = Path(plan['cad_source_path'])
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != plan['cad_sha256']:
            raise ValueError('The selected CAD changed; detect the pose again.')
        if plan.get('cad_manifest_path'):
            manifest = Path(plan['cad_manifest_path'])
            if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=plan['cad_manifest_sha256']:
                raise ValueError('The CAD catalog changed; detect the pose again.')

    def _check_demo(self, demo, cad_key, cad_digest, active_tcp, cfg):
        """The demo must describe the same CAD frame, TCP and robot as this capture."""
        if demo['cad_key'] != cad_key:
            raise ValueError(f"Detected {cad_key}, but demo {demo['demo_id']} shows {demo['cad_key']}.")
        if demo['cad_sha256'] != cad_digest:
            raise ValueError(f"The CAD mesh differs from the one used for the pose of demo {demo['demo_id']}.")
        dt, dr = difference(from_pose(demo['tcp_offset_m_rad']), from_pose(active_tcp))
        if dt > .0001 or dr > .1:
            raise ValueError(f"Demo {demo['demo_id']} was recorded with a different TCP offset.")
        if demo['robot_id'] != cfg['robot']['ip']:
            raise ValueError(f"Demo {demo['demo_id']} was recorded on robot {demo['robot_id']}.")

    def _check_trial_tcp(self, cfg):
        if cfg['robot']['execution_mode'] != 'supervised_trial':return
        hand = calibration.load(cfg['robot'])
        if not hand or hand['parent_frame'] != 'calibration_tcp' or hand.get('tcp_offset_m_rad') is None:
            raise ValueError('Trial requires the explicit full calibration TCP offset.')
        dt, dr = difference(from_pose(hand['tcp_offset_m_rad']), from_pose(self.robot.control.getTCPOffset()))
        if dt > .0001 or dr > .1:
            raise ValueError('Trial active TCP differs from the TCP used for calibration (0.1 mm / 0.1 deg tolerance).')

    def _authorize_motion(self, cfg, plan, review=None):
        if cfg['robot']['preview_only']:
            raise ValueError('Preview-only mode: motion is blocked.')
        if plan.get('alignment_mode') == 'skill_demo':
            skill_demos.verify({'file_hashes': plan['skill_demo_files']})
        if plan.get('T_base_from_tcp_target') is not None:
            distance, angle = difference(from_pose(plan['reference']['tcp_pose_m_rad']), plan['T_base_from_tcp_target'])
            policy = rotation_policy(from_pose(plan['reference']['tcp_pose_m_rad']),plan['T_base_from_tcp_target'],cfg['motion'])
            allowance = policy['effective_limit_deg']
            if distance > cfg['motion']['max_translation_m'] or angle > allowance + 1e-8:
                raise ValueError(f"Target exceeds the motion limits: {distance:.3f} m / {angle:.1f} degrees; allowed {cfg['motion']['max_translation_m']:g} m / {allowance:g} degrees.")
        hand_eye = calibration.load(cfg['robot'])
        trial = cfg['robot']['execution_mode'] == 'supervised_trial'
        if trial:
            if not self.robot or not self.robot.connected:raise ValueError('Connect the robot first.')
            if hand_eye is None or not calibration.describe(hand_eye)['preview_ready']:
                raise ValueError('Trial requires confirmed camera, optical frame and calibration TCP.')
            if cfg != self.robot.cfg:raise ValueError('Settings changed; reconnect before the trial.')
            self._check_trial_tcp(cfg)
            self._check_cad_identity(plan)
            check = plan.get('controller_check') or {}
            if not plan.get('controller_path_checked') or check.get('plan_id') != plan['id']:
                raise ValueError('Complete the controller IK and joint-limit check for this plan before the trial.')
            if not isinstance(review, dict) or review.get('confirmed') is not True:
                raise ValueError('Explicit operator confirmation is required for this supervised trial.')
        elif hand_eye is not None and (hand_eye['status'] != 'validated' or not calibration.describe(hand_eye)['preview_ready']):
            raise ValueError('Hand-eye calibration is not validated for normal motion.')
        if not cfg['robot']['motion_enabled'] or (not trial and not cfg['robot']['calibration_confirmed']):
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
                self._check_trial_tcp(cfg)
                active_tcp = self.robot.control.getTCPOffset()
                hand_eye = calibration.load(cfg['robot'])
                # Wait for a newer camera frame, bracketed by stationary TCP reads.
                with self.app.camera.condition:
                    sequence = self.app.camera.sequence
                    ok = self.app.camera.condition.wait_for(
                        lambda: self.app.camera.sequence > sequence or self.cancel.is_set()
                        or self.app.camera.error or self.app.camera.stop_event.is_set(), timeout=10)
                if self.cancel.is_set():raise RuntimeError('Capture cancelled.')
                camera_status = self.app.camera.status()
                if camera_status['error']:raise RuntimeError('Camera capture failed: '+camera_status['error'])
                if not camera_status['running']:raise RuntimeError('The camera was closed before capture.')
                if not ok:raise RuntimeError('No new camera frames. '+(camera_status.get('frame_warning') or 'Check the camera connection.'))
                captured = self.app.camera.capture()
                source = self.app.camera.capture_folder(captured['id'])
                camera_metadata = json.loads((source / 'camera.json').read_text())
                serial = camera_metadata.get('serial', '')
                self.robot.check_reference(reference)
                if time.time() - captured['timestamp'] > cfg['motion']['max_frame_age_s']:
                    raise RuntimeError('The RGB-D capture is not recent enough.')
                if cfg['robot']['camera_serial'] and serial != cfg['robot']['camera_serial']:
                    raise ValueError('The active camera does not match camera_serial in settings.json.')
                options = {**request.get('options', {}), 'mode': 'rgbd', 'device': 'cuda',
                           'pose': True, 'pose_backend': 'foundationpose', 'segmentor': 'sam'}
                trial = cfg['robot']['execution_mode'] == 'supervised_trial'
                if request.get('auto_assembly') or request.get('target_cad_key'):options['cad_units'] = 'mm'
                prepared = {'options': options}
                for key in ('rgb','depth','camera'):
                    p = source / {'rgb':'rgb.png','depth':'depth.png','camera':'camera.json'}[key]
                    prepared[key] = {'name': p.name, 'data': base64.b64encode(p.read_bytes()).decode()}
                catalog_path = Path(self.app.manifest_path)
                catalog_parts = json.loads(catalog_path.read_text())['parts']
                if request.get('auto_assembly'):
                    from sam6d_ui.assembly_identification import assembly_catalog
                    manifest, parts = assembly_catalog(request['auto_assembly'])
                    catalog_path, catalog_parts = Path(manifest), parts
                    cad = manifest.parent / parts[0]['source_stl']
                elif request.get('target_cad_key'):
                    cad, _, _ = self._catalog_cad(request['target_cad_key'])
                elif request.get('server_files'):
                    cad = self.app.files.resolve(request['server_files'].get('cad', ''))
                else:cad = None
                catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
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
                input_path = (folder / inputs['cad_filename']).resolve()
                input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
                inputs.update(capture_id=captured['id'], backend='foundationpose')
                if request.get('auto_assembly'):inputs['auto_assembly'] = request['auto_assembly']
                (folder/'inputs.json').write_text(json.dumps(inputs, indent=2))
                self._audit(folder, 'robot_capture.json', {'reference':reference,'capture':captured,
                            'camera_serial':serial,'settings_sha256':cfg_hash,
                            'active_tcp_offset_m_rad':active_tcp, 'hand_eye':calibration.describe(hand_eye)})
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
                    target = ('automatic assembly '+request['auto_assembly']) if request.get('auto_assembly') else inputs['cad_name']
                    score = result.get('best_score')
                    detail = f" Best SAM6D score {score:.3f}; required {inputs['options']['confidence']:.3f}." if isinstance(score,(int,float)) else ''
                    raise ValueError('No accepted detection and FoundationPose pose for '+target+'.'+detail+' No target was created.')
                best = pose['poses'][pose.get('best_index', 0)]
                bounds = np.asarray(inputs['cad_bounds_mm']) + np.asarray(inputs['input_center_mm'])
                selected_part = None
                if inputs.get('auto_assembly'):
                    selected_part = next((p for p in catalog_parts if p['key']==result.get('cad_key')), None)
                    if selected_part is None:raise ValueError('Identification returned a CAD outside the selected assembly.')
                    bounds = selected_part['bounds_local_mm']
                    cad_source, _, cad_digest = self._catalog_cad(selected_part['key'], catalog_path)
                else:
                    if inputs['options']['cad_units']=='mm':
                        selected_part = next((p for p in catalog_parts if p.get('source_sha256')==input_digest), None)
                    expected_key = selected_part['key'] if selected_part else 'uploaded_cad'
                    if result.get('cad_key') not in ('uploaded_cad', expected_key):
                        raise ValueError('Single-CAD inference returned a different object identity.')
                    cad_source, cad_digest = (cad.resolve() if cad is not None else input_path), input_digest
                cad_key = selected_part['key'] if selected_part else 'uploaded_cad'
                if request.get('target_cad_key') and cad_key != request['target_cad_key']:
                    raise ValueError('The catalog CAD changed during estimation; detect again.')
                if hashlib.sha256(catalog_path.read_bytes()).hexdigest()!=catalog_digest:
                    raise ValueError('The CAD catalog changed during estimation; detect again.')
                _, current_hash = load(self.settings_file)
                if current_hash != cfg_hash:
                    raise ValueError('Settings or the hand-eye YAML changed during estimation.')
                now_offset = self.robot.control.getTCPOffset()
                dt, dr = difference(from_pose(active_tcp), from_pose(now_offset))
                if dt > .0001 or dr > .1:
                    raise ValueError('The active TCP changed during capture or inference.')
                tcp_camera = calibration.resolve(hand_eye, active_tcp, camera_metadata) if hand_eye else None
                if hand_eye and np.linalg.norm(np.asarray(best['T_camera_from_input_cad_mm'])[:3,3]) * .001 > hand_eye['preview_limits']['max_camera_distance_m']:
                    raise ValueError('Pose exceeds the camera range; check the units.')
                target = alignment_target(request)
                if target['mode'] == 'skill_demo':
                    demo = skill_demos.select(cfg['alignment']['skill_bindings_file'], cad_key, target['demo_id'])
                    self._check_demo(demo, cad_key, cad_digest, active_tcp, cfg)
                    plan = build_demo_plan(best['T_camera_from_input_cad_mm'], reference['tcp_pose_m_rad'], cfg, demo,
                                           tcp_from_camera=tcp_camera, rotation_preview=trial)
                    plan['skill_demo_files'] = demo['file_hashes']
                else:
                    plan = build_plan(best['T_camera_from_input_cad_mm'], reference['tcp_pose_m_rad'], cfg, bounds,
                                      cad_key, tcp_from_camera=tcp_camera,
                                      rotation_preview=trial and {**cfg['alignment'], **cfg['alignment']['cad_overrides'].get(cad_key, {})}.get('orientation_mode') in ('cad_axes', 'cad_axes_inward', 'plane_frame'))
                plan['alignment_target'] = target
                preview = cfg['robot']['preview_only'] or bool(hand_eye and hand_eye['status'] != 'validated' and not trial)
                needs_check = preview or trial
                if not needs_check:
                    self.robot.validate_path(plan, reference)
                if self.cancel.is_set():raise RuntimeError('Plan cancelled.')
                plan.update(id=uuid.uuid4().hex, job_id=job_id, capture_id=captured['id'],
                            preview_only=preview, motion_allowed=False if needs_check else None,
                            controller_path_checked=not needs_check, collision_checked=False,
                            hand_eye=calibration.describe(hand_eye), active_tcp_offset_m_rad=active_tcp,
                            reference=reference, settings_sha256=cfg_hash, camera_serial=serial,
                            capture_timestamp=captured['timestamp'], planned_at=time.time(),
                            expires_in_s=cfg['motion']['max_plan_age_s'],
                            image_url=f'/files/{job_id}/pose_overlay.png',
                            artifact_folder=str(folder), accuracy_validated=False)
                plan.update(execution_mode=cfg['robot']['execution_mode'], cad_sha256=cad_digest,
                            cad_source_path=str(cad_source), cad_name=(selected_part.get('name') or selected_part['key']) if selected_part else inputs['cad_name'],
                            selection_mode='automatic' if inputs.get('auto_assembly') else 'selected',
                            selection_signature=request.get('selection_signature'), assembly=request.get('auto_assembly'),
                            cad_manifest_path=str(catalog_path.resolve()), cad_manifest_sha256=catalog_digest,
                            detection_score=result.get('best_score'))
                self._check_cad_identity(plan)
                self._audit(folder, 'robot_alignment_candidate.json', plan)
                if trial and not preview:
                    # Detect -> review -> execute: controller checks run automatically.
                    try:
                        details = self.robot.validate_path(plan, reference)
                    except Exception as exc:
                        if not plan['rotation_limit_exceeded']:raise
                        details = None
                        plan['controller_check_error'] = str(exc)
                        plan['controller_check_details'] = getattr(exc,'details',None)
                    if self.cancel.is_set():raise RuntimeError('Plan cancelled.')
                    _, current_hash = load(self.settings_file)
                    if current_hash != cfg_hash:raise ValueError('Settings changed during controller checks.')
                    report = {'checked_at':time.time(), 'controller_path_checked':details is not None,
                              'collision_checked':False, 'motion_sent':False,
                              'plan_id':plan['id'], 'samples':len(plan['path_samples_m_rad']),
                              'details':details}
                    self._audit(folder, 'controller_check_'+uuid.uuid4().hex+'.json', report)
                    plan.update(controller_path_checked=details is not None, controller_check=report)
                self._audit(folder, 'robot_alignment_plan.json', plan)
                with self.lock:
                    if self.cancel.is_set():raise RuntimeError('Plan cancelled.')
                    self.plan = plan
                    self._update('planned',
                        (f"Target requires {plan['rotation_deg']:.1f}°; allowed {plan['rotation_limit_deg']:g}°. Check motion.max_rotation_deg and calculate a new target after changing settings."
                         + (' Controller check: '+plan['controller_check_error'] if plan.get('controller_check_error') else ' Controller IK and joint limits accepted the samples.'))
                        if plan['rotation_limit_exceeded'] else 'Target ready. Controller IK and limits checked automatically. Review the pose and select Execute alignment.'
                        if trial else 'Preview calculated using the hand-eye YAML. Motion remains blocked.'
                        if preview else 'Target calculated. Review the image and matrix before execution.')
        except Exception as exc:
            self.plan = None
            self._update('error', str(exc))
            if folder is not None and folder.is_dir():
                try:self._audit(folder, 'robot_alignment_error.json', {'error':str(exc),'timestamp':time.time(),
                                'controller_check_details':getattr(exc,'details',None),'motion_sent':False})
                except OSError:pass

    def _execute(self, plan):
        try:
            with self.app.submission_lock:
                cfg, digest = load(self.settings_file)
                if digest != plan['settings_sha256']:raise ValueError('Settings changed; plan invalidated.')
                self._authorize_motion(cfg, plan, plan.get('trial_review'))
                if self.cancel.is_set():raise RuntimeError('Movement cancelled.')
                # Persist rechecked IK before dispatch; move() repeats controller checks.
                if cfg['robot']['execution_mode'] == 'supervised_trial':
                    details = self.robot.validate_path(plan, plan['reference'])
                    self._audit(Path(plan['artifact_folder']), 'trial_dispatch_check_'+uuid.uuid4().hex+'.json', details)
                def before_dispatch():
                    current, current_hash = load(self.settings_file)
                    if current_hash != plan['settings_sha256']:
                        raise ValueError('Settings or hand-eye changed during controller checks; plan invalidated.')
                    self._authorize_motion(current, plan, plan.get('trial_review'))
                result = self.robot.move(plan, plan['reference'], before_dispatch=before_dispatch)
                result.update(accuracy_validated=False, collision_checked=False,
                              execution_mode=cfg['robot']['execution_mode'])
                self._audit(Path(plan['artifact_folder']), 'robot_alignment_execution.json', result)
                if plan.get('alignment_mode') == 'skill_demo':
                    demo = plan['skill_demo']
                    with self.lock:
                        self.skill_ready = {'plan_id': plan['id'], 'demo_id': demo['demo_id'], 'skills': demo['skills'],
                                            'cad_key': plan['cad_key'], 'aligned_tcp_m_rad': plan['tcp_target_m_rad'],
                                            'aligned_at': time.time(), 'artifact_folder': plan['artifact_folder'],
                                            'settings_sha256': plan['settings_sha256'],
                                            'tcp_offset_m_rad': plan['active_tcp_offset_m_rad'],
                                            'demos_root': str(Path(next(iter(plan['skill_demo_files']))).parents[1]),
                                            'demo_files': plan['skill_demo_files']}
                    self._update('completed', f"TCP aligned to skill demo {demo['demo_id']}. Choose Align again (detect pose) to refine the alignment, review the new target and execute alignment again, or Execute skill when ready.", result=result)
                else:
                    self._update('completed', 'TCP aligned. Choose Align again (detect pose), review the new target and execute alignment again if needed.', result=result)
        except Exception as exc:
            self._update('error', str(exc))
            try:
                self._audit(Path(plan['artifact_folder']), 'robot_alignment_execution.json',
                            {'status':'error','error':str(exc),'timestamp':time.time(),
                             'controller_check_details':getattr(exc,'details',None)})
            except OSError:pass

    def close(self):
        self.closed = True
        self.cancel.set()
        process = self.skill_process
        if process is not None:
            try:process.send_signal(signal.SIGINT)
            except (OSError, ValueError):pass
        if self.robot:
            self.robot.cancel.set()
            if self.state['state'] in ('moving','stopping'):
                try:self.robot.stop()
                except Exception as exc:self._update('error', 'Failed to send stop: '+str(exc))
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.robot:self.robot.disconnect()
