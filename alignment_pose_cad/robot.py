"""UR RTDE adapter. Construction/import never opens hardware or operates a gripper."""
import threading
import time
import numpy as np
from .geometry import from_pose, difference, vector


class RTDERobot:
    def __init__(self, settings, receive_factory=None, control_factory=None):
        self.cfg = settings
        self.receive_factory = receive_factory
        self.control_factory = control_factory
        self.receive = self.control = None
        self.command_lock = threading.RLock()
        self.cancel = threading.Event()
        self.timestamp = None
        self.timestamp_seen = None

    @property
    def connected(self):
        return self.receive is not None and self.control is not None

    def connect(self):
        if self.connected: return self.snapshot(stationary=True)
        if self.receive_factory is None:
            from rtde_receive import RTDEReceiveInterface
            from rtde_control import RTDEControlInterface
            self.receive_factory, self.control_factory = RTDEReceiveInterface, RTDEControlInterface
        cfg = self.cfg['robot']
        self.cancel.clear()
        try:
            self.receive = self.receive_factory(cfg['ip'], cfg['frequency_hz'])
            self.control = self.control_factory(cfg['ip'], cfg['frequency_hz'])
            # Observe the controller's TCP. Never silently change its calibration.
            expected = from_pose(cfg['tcp_offset_m_rad'])
            actual = from_pose(self.control.getTCPOffset())
            dt, dr = difference(expected, actual)
            if dt > .0001 or dr > .1:
                raise ValueError('The active TCP does not match tcp_offset_m_rad in settings.json.')
            return self.snapshot(stationary=True)
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self):
        for obj in (self.control, self.receive):
            if obj is not None:
                try: obj.disconnect()
                except Exception: pass
        self.control = self.receive = None
        self.timestamp = self.timestamp_seen = None

    def snapshot(self, stationary=False):
        if not self.connected:raise RuntimeError('Connect the robot first.')
        r, c = self.receive, self.control
        if not r.isConnected() or not c.isConnected():raise RuntimeError('RTDE connection lost.')
        safety = int(r.getSafetyStatusBits())
        if not safety & 3 or safety & ~3:
            raise RuntimeError(f'Robot safety stop ({safety}); check the teach pendant.')
        if r.getRobotMode() != 7 or not c.isProgramRunning():
            raise RuntimeError('Robot no preparado o programa RTDE detenido.')
        timestamp = float(r.getTimestamp())
        if not np.isfinite(timestamp):raise RuntimeError('Invalid robot telemetry.')
        now = time.monotonic()
        if timestamp != self.timestamp:
            self.timestamp, self.timestamp_seen = timestamp, now
        elif self.timestamp_seen is not None and now - self.timestamp_seen > .5:
            raise RuntimeError('RTDE telemetry is stale.')
        p = vector(r.getActualTCPPose(), 6, 'TCP')
        q = vector(r.getActualQ(), 6, 'juntas')
        speed = vector(r.getActualTCPSpeed(), 6, 'velocidad TCP')
        qd = vector(r.getActualQd(), 6, 'joint velocity')
        if stationary and (np.max(np.abs(speed)) > self.cfg['motion']['max_stationary_tcp_speed']
                           or np.max(np.abs(qd)) > self.cfg['motion']['max_stationary_joint_speed']):
            raise RuntimeError('The robot must be stationary before capturing or executing a plan.')
        return {'tcp_pose_m_rad': p.tolist(), 'q_rad': q.tolist(), 'speed': speed.tolist(),
                'controller_timestamp': timestamp, 'received_at': time.time(), 'safety_bits': safety}

    def check_reference(self, reference, stationary=True):
        now = self.snapshot(stationary=stationary)
        dt, dr = difference(from_pose(reference['tcp_pose_m_rad']), from_pose(now['tcp_pose_m_rad']))
        limits = self.cfg['motion']
        if dt > limits['max_robot_drift_m'] or dr > limits['max_robot_drift_deg']:
            raise RuntimeError('The robot moved after capture. Calculate a new plan.')
        if np.max(np.abs(np.asarray(now['q_rad']) - reference['q_rad'])) > .02:
            raise RuntimeError('The joint configuration changed after capture.')
        return now

    def validate_path(self, plan, reference):
        now = self.check_reference(reference)
        expected = from_pose(self.cfg['robot']['tcp_offset_m_rad'])
        dt, dr = difference(expected, from_pose(self.control.getTCPOffset()))
        if dt > .0001 or dr > .1:raise RuntimeError('The active TCP has changed.')
        previous = np.asarray(now['q_rad'])
        initial = previous.copy()
        for waypoint in plan['path_samples_m_rad']:
            if self.cancel.is_set():raise RuntimeError('Operation stopped.')
            if not self.control.isPoseWithinSafetyLimits(waypoint):
                raise RuntimeError('The controller rejects a trajectory waypoint.')
            if not self.control.getInverseKinematicsHasSolution(waypoint, previous.tolist()):
                raise RuntimeError('The trajectory has no inverse kinematics solution.')
            q = vector(self.control.getInverseKinematics(waypoint, previous.tolist()), 6, 'IK')
            # Do not wrap by 2*pi: a different joint turn is also a physical move.
            limit = self.cfg['motion']['max_joint_delta_rad']
            if max(np.max(np.abs(q - previous)), np.max(np.abs(q - initial))) > limit:
                raise RuntimeError('The trajectory requires an excessive joint jump.')
            if not self.control.isJointsWithinSafetyLimits(q.tolist()):
                raise RuntimeError('The trajectory exceeds controller joint limits.')
            previous = q
        self.check_reference(reference)

    def stop(self):
        self.cancel.set()
        with self.command_lock:
            if self.connected:
                self.control.stopL(self.cfg['motion']['stop_deceleration_m_s2'], True)

    def move(self, plan, reference):
        self.validate_path(plan, reference)
        limits = self.cfg['motion']
        with self.command_lock:
            if self.cancel.is_set():raise RuntimeError('Movement cancelled before dispatch.')
            self.check_reference(reference)
            before = self.control.getAsyncOperationProgress()
            if before >= 0:raise RuntimeError('The controller is already executing another movement.')
            if self.cancel.is_set():raise RuntimeError('Movement cancelled before dispatch.')
            if not self.control.moveL(plan['tcp_target_m_rad'], limits['speed_m_s'], limits['acceleration_m_s2'], True):
                raise RuntimeError('The controller rejected moveL.')
        deadline = time.monotonic() + limits['move_timeout_s']
        target = from_pose(plan['tcp_target_m_rad'])
        try:
            while True:
                if self.cancel.wait(.02):raise RuntimeError('Movement stopped by the user.')
                state = self.snapshot()
                progress = self.control.getAsyncOperationProgress()
                if progress < 0 and progress != before:break
                if time.monotonic() > deadline:raise RuntimeError('Movement timed out.')
            settle = time.monotonic() + limits['settle_timeout_s']
            while True:
                if self.cancel.is_set():raise RuntimeError('Movement cancelled.')
                current = self.snapshot()
                dt, dr = difference(target, from_pose(current['tcp_pose_m_rad']))
                if dt <= limits['position_tolerance_m'] and dr <= limits['rotation_tolerance_deg']:
                    self.snapshot(stationary=True)
                    return {'status': 'completed', 'actual_tcp_m_rad': current['tcp_pose_m_rad'],
                            'position_error_m': dt, 'rotation_error_deg': dr}
                if time.monotonic() > settle:raise RuntimeError('The robot did not reach the target within tolerance.')
                time.sleep(.02)
        except BaseException:
            self.stop()
            raise
