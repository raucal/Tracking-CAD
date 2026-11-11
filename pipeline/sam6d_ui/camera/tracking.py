"""One persistent GPU worker; always use the latest shared RGB-D, without a frame queue."""
import json
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from queue import Queue, Empty
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np

from ..backends.foundationpose import python_path
from ..backends.geometry import bbox_agreement
from settings import profile


def validate_interval(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=0 and not 1<=value<=3600:
        raise ValueError('The SAM interval must be 0 (disabled) or between 1 and 3600 seconds.')
    return float(value)


class LivePoseTracker:
    def __init__(self,camera,root,worker_script=None,resident_factory=None):
        self.camera=camera;self.root=Path(root);self.resident_factory=resident_factory
        self.tracking_config=profile("interface")["tracking"]
        self.worker_script=Path(worker_script or Path(__file__).parents[1]/'backends/live_pose_worker.py')
        self.lock=threading.RLock();self.condition=threading.Condition(self.lock)
        self.stop_event=threading.Event();self.thread=None;self.process=None
        self.redetect=None;self.redetect_interval_s=0;self.config_revision=0;self.detector_cancel=None
        self.jpeg=None;self.cad_jpeg=None;self.sequence=0;self.details={'running':False,'state':'idle','frames':0,'fps':0}

    def status(self):
        with self.lock:return dict(self.details,running=bool(self.thread and self.thread.is_alive() and not self.stop_event.is_set()))

    def busy(self):
        return bool(self.thread and self.thread.is_alive())

    def configure(self,interval):
        interval=validate_interval(interval)
        with self.lock:
            if not self.busy():raise ValueError('Start tracking first.')
            if interval and self.redetect is None:raise ValueError('Tracking has no SAM detector. Restart in automatic mode.')
            if self.detector_cancel is not None:self.detector_cancel.set()
            self.redetect_interval_s=interval;self.config_revision+=1
            self.details['redetect_interval_s']=interval
            return self.status()

    def start(self,folder,redetect_interval_s=0,redetect=None):
        interval=validate_interval(redetect_interval_s)
        if (interval and redetect is None) or (redetect is not None and self.resident_factory is None):raise ValueError('SAM detector and resident worker are required for refresh.')
        with self.lock:
            if self.busy():raise ValueError('Tracking is already active or closing.')
            self.camera.frame()
            self.stop_event.clear();self.jpeg=None;self.cad_jpeg=None;self.sequence=0
            self.redetect=redetect;self.redetect_interval_s=interval;self.config_revision+=1
            session=uuid.uuid4().hex
            self.details={'running':True,'state':'loading','frames':0,'fps':0,'session':session,'seed_job':Path(folder).name,'error':None,'redetect_interval_s':interval,'redetect_attempts':0,'redetections':0,'redetect_error':None,'sam_running':False,'sam_checks':0,'sam_discarded':0,'sam_misses':0,'sam_disagreements':0,'automatic_recovery':redetect is not None,'losses':0,'recovery_attempts':0,'backup_attempts':0,'consecutive_quality_failures':0,'sam_reason':None}
            self.thread=threading.Thread(target=self._run,args=(Path(folder),session),name='FoundationPose-live',daemon=True)
            self.thread.start()
            return self.status()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            process=self.process
            if self.detector_cancel is not None:self.detector_cancel.set()
            if self.busy():self.details['state']='stopping'
        if process and process.poll() is None:process.terminate()
        if self.thread and self.thread is not threading.current_thread():self.thread.join(5)
        with self.condition:self.condition.notify_all()
        return self.status()

    def latest_preview(self, view='camera'):
        with self.lock:
            jpeg = self.cad_jpeg if view == 'cad' else self.jpeg
            if not jpeg or self.stop_event.is_set() or not self.busy() or self.details.get('state')!='tracking':return None
            age=max(0,time.time()-self.details['timestamp'])
            if age>1:return None
            return {'sequence':self.sequence,'jpeg':jpeg,'age_s':age,'timestamp':self.details['timestamp']}

    def preview(self,previous,timeout=3):
        with self.condition:
            self.condition.wait_for(lambda:self.sequence!=previous and self.jpeg is not None or not self.busy() or self.stop_event.is_set(),timeout)
            if not self.jpeg or self.sequence==previous or self.stop_event.is_set():return None
            return self.sequence,self.jpeg

    def _run(self,folder,session):
        if self.resident_factory is not None:
            return self._run_resident(folder,session)
        messages=Queue();reader=None;process=None
        session_dir=self.root/session
        scratch_context=None;log=None
        def receive(timeout):
            deadline=time.monotonic()+timeout
            while not self.stop_event.is_set():
                try:result=messages.get(timeout=min(.2,max(.01,deadline-time.monotonic())))
                except Empty:
                    if time.monotonic()>deadline:raise RuntimeError('The worker did not respond in time.')
                    continue
                if result is None:raise RuntimeError('The tracking worker closed. See worker.log.')
                if result.get('status')=='error':raise RuntimeError(result.get('error','Tracking error.'))
                return result
            return None
        try:
            session_dir.mkdir(parents=True,exist_ok=True)
            scratch_context=tempfile.TemporaryDirectory(prefix='sam6d-live-',dir='/dev/shm' if Path('/dev/shm').is_dir() else None)
            scratch_name=scratch_context.name;log=(session_dir/'worker.log').open('w')
            scratch=Path(scratch_name)
            process=subprocess.Popen([str(python_path()),'-B',str(self.worker_script),'--seed',str(folder),'--scratch',str(scratch)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            with self.lock:self.process=process
            def read():
                try:
                    for line in process.stdout:
                        if line.startswith('@LIVE@'):
                            try:messages.put(json.loads(line[len('@LIVE@'):]))
                            except ValueError:pass
                        elif not log.closed:log.write(line);log.flush()
                finally:messages.put(None)
            reader=threading.Thread(target=read,daemon=True);reader.start()
            if not receive(120):return
            calibration=json.loads((folder/'camera.json').read_text())
            K=np.asarray(calibration['cam_K']).reshape(3,3)
            previous=None;smoothed=None;published=None
            while not self.stop_event.is_set():
                frame=self.camera.frame();identity=frame['number']
                if identity==previous:
                    self.stop_event.wait(.01);continue
                if not np.allclose(K,np.asarray(frame['camera']['cam_K']).reshape(3,3)) or not np.isclose(calibration['depth_scale'],frame['camera']['depth_scale']):
                    raise RuntimeError('Calibration changed. Restart initial detection.')
                previous=identity;started=time.perf_counter()
                np.savez(scratch/'frame.npz',rgb=frame['rgb'],depth=frame['depth'],K=K,scale=calibration['depth_scale'])
                process.stdin.write('{}\n');process.stdin.flush()
                result=receive(30)
                if result is None:break
                finished=time.perf_counter();seconds=finished-started if published is None else finished-published;published=finished
                smoothed=seconds if smoothed is None else .8*smoothed+.2*seconds
                jpeg=(scratch/'preview.jpg').read_bytes()
                with self.condition:
                    self.jpeg=jpeg;self.sequence+=1
                    self.details.update(state='tracking',frames=self.sequence,fps=1/smoothed,frame_number=frame['number'],timestamp=frame['timestamp'],latency_s=time.time()-frame['timestamp'],pose=result['T_camera_from_input_cad_mm'])
                    self.condition.notify_all()
        except Exception as error:
            if not self.stop_event.is_set():
                with self.lock:self.details.update(state='error',error=str(error))
        finally:
            if process:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                if reader:reader.join(3)
                for pipe in (process.stdin,process.stdout):
                    if pipe:pipe.close()
            if log:log.close()
            if scratch_context:scratch_context.cleanup()
            self.stop_event.set()
            with self.condition:
                self.process=None
                if self.details['state']!='error':self.details['state']='stopped'
                self.details['running']=False;self.condition.notify_all()
            if session_dir.is_dir():
                (session_dir/'session.json').write_text(json.dumps(self.status(),indent=2,allow_nan=False))

    def _run_resident(self, folder, session):
        session_dir = self.root / session
        executor = scratch_context = future = job = None
        config = self.tracking_config
        retry_delay = config['recovery_retry_s']
        max_age = config['max_detection_age_s']
        failure_limit = config['quality_bad_frames']
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            worker = self.resident_factory()
            worker.start_tracking(folder)
            seed_result = folder / 'result.json'
            initial = json.loads(seed_result.read_text()) if seed_result.is_file() else {}
            current_cad = initial.get('cad_key')
            with self.lock:
                self.details.update(cad_key=current_cad, display_name=initial.get('display_name'),
                                    identification=initial.get('identification') is not None,
                                    component_changes=0)
            calibration = json.loads((folder/'camera.json').read_text())
            K = np.asarray(calibration['cam_K']).reshape(3, 3)
            previous = published = smoothed = None
            lost = False
            lost_after_frame = -1
            bad_frames = misses = disagreements = 0
            revision = -1
            next_due = float('inf')
            retry_due = 0.0
            force_fresh = False
            scratch_context = tempfile.TemporaryDirectory(prefix='sam6d-live-', dir='/dev/shm' if Path('/dev/shm').is_dir() else None)
            base = Path(scratch_context.name)
            scratch = base/'tracking'; scratch.mkdir()
            sam_scratch = base/'sam'; sam_scratch.mkdir()
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='SAM-verification')

            def check_camera(frame):
                if not np.allclose(K, np.asarray(frame['camera']['cam_K']).reshape(3, 3)) or not np.isclose(calibration['depth_scale'], frame['camera']['depth_scale']):
                    raise RuntimeError('Calibration changed. Restart initial detection.')

            def mark_lost(message, frame_number):
                nonlocal lost, lost_after_frame, force_fresh
                if not lost:
                    lost_after_frame = frame_number
                    force_fresh = True
                    with self.lock: self.details['losses'] += 1
                    if future is not None and job is not None: job['cancelled'].set()
                lost = True
                with self.condition:
                    self.jpeg = None
                    self.details.update(state='recovering', pose=None, redetect_error=message, last_loss_reason=message, fps=0)
                    self.condition.notify_all()

            def detect(snapshot, cancelled):
                np.savez(sam_scratch/'frame.npz', rgb=snapshot['rgb'], depth=snapshot['depth'], K=K, scale=calibration['depth_scale'])
                (sam_scratch/'timestamp.json').write_text(json.dumps(snapshot['timestamp']))
                return self.redetect(snapshot, sam_scratch, cancelled)

            def submit_detection(frame, predicted_box, current_revision, reason):
                nonlocal future, job, next_due, force_fresh
                if future is not None: return
                snapshot = {**frame, 'rgb':frame['rgb'].copy(), 'depth':frame['depth'].copy()}
                cancelled = threading.Event()
                started = time.monotonic()
                with self.lock:
                    if self.stop_event.is_set() or self.config_revision != current_revision or self.redetect is None: return
                    self.detector_cancel = cancelled
                    self.details.update(sam_running=True, sam_reason=reason, redetect_attempts=self.details['redetect_attempts']+1)
                    self.details['recovery_attempts' if reason == 'recovery' else 'backup_attempts'] += 1
                job = {'frame':snapshot, 'predicted':predicted_box, 'revision':current_revision, 'started':started, 'cancelled':cancelled, 'reason':reason}
                future = executor.submit(detect, snapshot, cancelled)
                force_fresh = False
                next_due = started+interval if interval else float('inf')

            while not self.stop_event.is_set():
                with self.lock:
                    interval = self.redetect_interval_s
                    current_revision = self.config_revision
                if revision != current_revision:
                    revision = current_revision
                    next_due = time.monotonic()+interval if interval else float('inf')
                    if lost: force_fresh = True
                frame = self.camera.frame()
                check_camera(frame)

                if future is not None and future.done():
                    elapsed = time.monotonic()-job['started']
                    entry = {'timestamp':time.time(), 'frame_number':job['frame']['number'], 'elapsed_s':elapsed, 'reason':job['reason']}
                    detection = error = None
                    try: detection = future.result()
                    except Exception as exception: error = str(exception)
                    future = None
                    with self.lock:
                        self.details.update(sam_running=False, sam_reason=None, last_redetect_s=elapsed, last_redetect_at=time.time())
                        self.detector_cancel = None
                    discarded = None
                    if current_revision != job['revision']: discarded = 'configuration_changed'
                    elif lost and job['frame']['number'] <= lost_after_frame: discarded = 'before_tracking_loss'
                    elif job['cancelled'].is_set(): discarded = 'cancelled'
                    elif time.time()-job['frame']['timestamp'] > max_age: discarded = 'stale_image'
                    if discarded:
                        entry.update(status='discarded', reason_discarded=discarded)
                        with self.lock: self.details['sam_discarded'] += 1
                    elif error:
                        entry.update(status='detection_error', error=error)
                        with self.lock: self.details['redetect_error'] = 'SAM verification error: '+error
                    else:
                        with self.lock: self.details['sam_checks'] += 1
                        if detection is None:
                            misses += 1; disagreements = 0
                            warning = 'SAM cannot find the part ('+str(misses)+'/2).'
                            with self.lock: self.details.update(sam_misses=misses, sam_disagreements=0, redetect_error=warning)
                            if misses >= 2: mark_lost(warning+' Waiting for a new detection.', frame['number'])
                            entry.update(status='not_detected', misses=misses)
                        else:
                            misses = 0
                            agreement = bbox_agreement(job['predicted'], detection.get('bbox_xyxy'))
                            component_changed = detection.get('cad_key') is not None and detection['cad_key'] != current_cad
                            needs_registration = lost
                            if component_changed:
                                needs_registration = True
                            elif not lost:
                                if agreement is None: entry.update(status='unverified')
                                elif agreement['consistent']:
                                    disagreements = 0; entry.update(status='consistent', **agreement)
                                else:
                                    disagreements += 1; needs_registration = disagreements >= 2
                                    entry.update(status='disagreement', consecutive=disagreements, **agreement)
                            with self.lock:
                                self.details.update(sam_misses=0, sam_disagreements=disagreements, last_detection_score=detection['score'], last_sam_agreement=agreement, redetect_error=None)
                            entry.update(**detection)
                            if needs_registration and not lost:
                                # A backup result can confirm drift, but registration must use
                                # a new snapshot captured after declaring tracking lost.
                                mark_lost('SAM identified another part. Confirming the CAD with a new image…' if component_changed else
                                          'SAM confirmed a drift. Looking for a new detection…', frame['number'])
                                needs_registration = False
                                entry.update(status='tracking_loss_confirmed')
                            if needs_registration:
                                mark_lost('Recovering pose with a recent SAM detection…', frame['number'])
                                with self.lock: self.details['state'] = 'redetecting'
                                try:
                                    check_camera(self.camera.frame())
                                    if self.stop_event.is_set(): break
                                    registered = worker.relocalize(sam_scratch)
                                    if self.stop_event.is_set(): break
                                    if detection.get('cad_key') is not None:
                                        current_cad = detection['cad_key']
                                        with self.lock:
                                            self.details.update(cad_key=current_cad, display_name=detection['display_name'])
                                            if component_changed: self.details['component_changes'] += 1
                                    lost = False
                                    force_fresh = False
                                    previous = published = smoothed = None
                                    bad_frames = misses = disagreements = 0
                                    next_due = time.monotonic()+interval if interval else float('inf')
                                    with self.lock:
                                        self.details.update(redetections=self.details['redetections']+1, redetect_error=None, sam_misses=0, sam_disagreements=0, consecutive_quality_failures=0, last_registration_s=registered.get('seconds'))
                                    entry.update(status='relocalized', registration_s=registered.get('seconds'))
                                except Exception as exception:
                                    if self.stop_event.is_set(): break
                                    mark_lost(str(exception), frame['number'])
                                    force_fresh = False
                                    entry.update(status='registration_error', error=str(exception))
                    with (session_dir/'redetections.jsonl').open('a') as log: log.write(json.dumps(entry, allow_nan=False)+'\n')
                    job = None
                    now = time.monotonic()
                    retry_due = now+retry_delay
                    # Skip missed backup ticks instead of accumulating jobs.
                    if interval and next_due <= now: next_due += (math.floor((now-next_due)/interval)+1)*interval
                    frame = self.camera.frame()
                    check_camera(frame)

                if lost:
                    if self.redetect is None:
                        raise RuntimeError(self.details.get('redetect_error') or 'Pose lost. Locate the part again.')
                    if future is None and (force_fresh or time.monotonic() >= retry_due):
                        submit_detection(frame, None, current_revision, 'recovery')
                    self.stop_event.wait(.01)
                    continue
                if frame['number'] == previous:
                    self.stop_event.wait(.005)
                    continue
                previous = frame['number']
                started = time.perf_counter()
                np.savez(scratch/'frame.npz', rgb=frame['rgb'], depth=frame['depth'], K=K, scale=calibration['depth_scale'])
                (scratch/'timestamp.json').write_text(json.dumps(frame['timestamp']))
                try: result = worker.track(scratch)
                except Exception as exception:
                    mark_lost('Tracking failed: '+str(exception)+'. Waiting for a new SAM detection.', frame['number'])
                    continue
                if self.stop_event.is_set(): break
                quality = result.get('quality', {'ok':True, 'enabled':False})
                if not quality['ok']:
                    bad_frames += 1
                    labels = {'depth_mismatch':'pose inconsistent with depth', 'occluded':'part occluded', 'outside_image':'part outside the image', 'insufficient_object_depth':'insufficient depth on the part', 'no_visible_surface':'CAD has no visible surface', 'translation_jump':'abrupt position change', 'rotation_jump':'abrupt orientation change', 'invalid_pose':'invalid pose'}
                    reason = ', '.join(labels.get(key,key) for key in quality['reasons'])
                    with self.condition:
                        self.jpeg = None
                        self.details.update(state='checking', pose=None, pose_quality=quality, consecutive_quality_failures=bad_frames, redetect_error=reason)
                        self.condition.notify_all()
                    if bad_frames >= failure_limit: mark_lost('Tracking lost: '+reason, frame['number'])
                    continue
                was_checking = bad_frames > 0
                bad_frames = 0
                finished = time.perf_counter()
                seconds = finished-started if published is None else finished-published
                published = finished
                smoothed = seconds if smoothed is None else .8*smoothed+.2*seconds
                jpeg = (scratch/'preview.jpg').read_bytes()
                cad_path = scratch/'cad_preview.jpg'
                cad_jpeg = cad_path.read_bytes() if cad_path.is_file() else None
                with self.condition:
                    self.jpeg = jpeg; self.cad_jpeg = cad_jpeg; self.sequence += 1
                    self.details.update(cad_view_error=result.get('cad_view_error'), cad_render_s=result.get('cad_render_s'))
                    self.details.update(state='tracking', frames=self.details['frames']+1, fps=1/smoothed, frame_number=frame['number'], timestamp=frame['timestamp'], latency_s=time.time()-frame['timestamp'], pose=result['T_camera_from_input_cad_mm'], worker_pid=worker.status()['pid'], pose_quality=quality, consecutive_quality_failures=0)
                    if was_checking: self.details['redetect_error'] = None
                    self.condition.notify_all()
                if self.redetect is not None and interval and future is None and time.monotonic() >= next_due:
                    submit_detection(frame, result.get('bbox_xyxy'), current_revision, 'backup')
        except Exception as exception:
            if not self.stop_event.is_set():
                with self.lock: self.details.update(state='error', error=str(exception), pose=None)
        finally:
            self.stop_event.set()
            with self.lock:
                if self.detector_cancel is not None: self.detector_cancel.set()
            if executor: executor.shutdown(wait=True, cancel_futures=True)
            if scratch_context: scratch_context.cleanup()
            with self.condition:
                self.detector_cancel = None
                self.details.update(sam_running=False, sam_reason=None, running=False)
                if self.details['state'] != 'error': self.details['state'] = 'stopped'
                self.condition.notify_all()
            if session_dir.is_dir(): (session_dir/'session.json').write_text(json.dumps(self.status(), indent=2, allow_nan=False))
