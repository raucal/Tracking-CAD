"""Replay one DINOBOT skill demo from the current, already aligned TCP.

Run by the DINOBOT interpreter in its own process (DINOBOT and CAD Tracking
both have a top-level ``settings`` module, and UR RTDE accepts one control
client). It uses DINOBOT's own robot wrapper, MotionControl and
``replay_demo(..., ask_start_pose=False)``, exactly as the DINOPDDL bridge does,
so the recorded trajectory is applied relative to the aligned TCP. SIGINT or
SIGTERM requests DINOBOT's stopJ braking. The outcome is written as JSON.
"""
import argparse
import json
import math
import signal
import sys
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summary(report):
    keys = ('status', 'error', 'completed_samples', 'total_samples', 'blocked_samples', 'speed_factor',
            'replay_origin', 'recorded_origin', 'log_path', 'elapsed_s', 'stop_error')
    return {**{k: report.get(k) for k in keys if k in report},
            'gripper_events': len(report.get('gripper_events', []))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dinobot', required=True, type=Path)
    parser.add_argument('--demos-root', required=True, type=Path)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--robot-ip', required=True)
    parser.add_argument('--expected-start', required=True, help='Aligned TCP pose, JSON [x,y,z,rx,ry,rz]')
    parser.add_argument('--tolerance-m', required=True, type=float)
    parser.add_argument('--tolerance-deg', required=True, type=float)
    parser.add_argument('--speed-factor', type=float, default=1.0)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args(argv)
    # Never resolve DINOBOT imports to CAD Tracking modules next to this script.
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE]
    sys.path.insert(0, str(args.dinobot.resolve()))
    result = {'status': 'error', 'demo_id': args.demo, 'robot_commands_sent': False}
    state = {'robot': None, 'motion': None, 'stopper': None}

    def on_signal(signum, frame):
        result['stop_requested'] = True
        motion, robot = state['motion'], state['robot']
        if motion is None:
            raise KeyboardInterrupt('Stopped before connecting to the robot.')
        motion.request_stop()
        if robot is not None and state['stopper'] is None:
            import settings
            def brake():
                try:
                    result['stop_status'] = motion.stop_robot(robot.rtde_c, settings.STOP_J_DECELERATION, robot.rtde_r)
                except Exception as exc:
                    result['stop_error'] = str(exc)
            state['stopper'] = threading.Thread(target=brake, daemon=True)
            state['stopper'].start()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation
        import settings
        settings.DEMO_DIR = str(args.demos_root.resolve())
        if str(settings.ROBOT_IP) != args.robot_ip:
            raise ValueError(f'DINOBOT is configured for robot {settings.ROBOT_IP}, not {args.robot_ip}.')
        from motion_control import MotionControl
        from demo_replay import replay_demo
        from teleop_setup.robots.ur5e_wrapper import teleop_robot
        state['motion'] = MotionControl()
        result['robot_commands_sent'] = True  # The wrapper activates and opens the gripper.
        state['robot'] = robot = teleop_robot(args.robot_ip)
        motion = state['motion']
        motion.prepare_task(robot)
        expected = np.asarray(json.loads(args.expected_start), dtype=float)
        actual = np.asarray(motion.call(robot.rtde_r.getActualTCPPose), dtype=float)
        distance = float(np.linalg.norm(expected[:3] - actual[:3]))
        angle = math.degrees((Rotation.from_rotvec(expected[3:]).inv() * Rotation.from_rotvec(actual[3:])).magnitude())
        result['start_check'] = {'expected_tcp': expected.tolist(), 'actual_tcp': actual.tolist(),
                                 'distance_m': distance, 'angle_deg': angle}
        if distance > args.tolerance_m or angle > args.tolerance_deg:
            raise RuntimeError(f'TCP is {distance * 1000:.2f} mm / {angle:.2f} deg away from the aligned pose; '
                               'the skill was not started.')
        report = replay_demo(robot, None, args.demo, ask_start_pose=False, strict=True,
                             motion_control=motion, speed_factor=args.speed_factor)
        result.update(status=report.get('status', 'error'), replay=summary(report))
    except BaseException as exc:
        result.update(status='stopped' if result.get('stop_requested') else 'error', error=str(exc) or type(exc).__name__,
                      traceback=traceback.format_exc())
        if getattr(exc, 'replay_report', None):
            result['replay'] = summary(exc.replay_report)
    finally:
        if state['stopper'] is not None:
            state['stopper'].join(10)
        robot = state['robot']
        if robot is not None:
            for interface in (robot.rtde_c, robot.rtde_r):
                try:
                    interface.disconnect()
                except Exception:
                    pass
        args.result.write_text(json.dumps(result, indent=2, default=str) + '\n')
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
