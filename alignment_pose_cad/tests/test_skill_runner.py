"""skill_runner.py in a real subprocess against a fake DINOBOT tree; no hardware."""
import json
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / 'skill_runner.py'
FAKE = {
    'settings.py': '''
        ROBOT_IP = "10.0.0.1"
        STOP_J_DECELERATION = 2.0
        DEMO_DIR = "unset"
    ''',
    'motion_control.py': '''
        import json, os, threading
        LOG = os.environ["FAKE_LOG"]
        def log(event):
            with open(LOG, "a") as f: f.write(json.dumps(event) + "\\n")
        class MotionStopped(RuntimeError): pass
        class MotionControl:
            def __init__(self): self.stop = threading.Event()
            def prepare_task(self, robot): log({"event": "prepare"})
            def call(self, fn, *a, **k): return fn(*a, **k)
            def request_stop(self): self.stop.set()
            def check(self):
                if self.stop.is_set(): raise MotionStopped("Robot task stopped by the user")
            def stop_robot(self, rtde_c, deceleration, rtde_r=None): log({"event": "stopJ", "deceleration": deceleration})
    ''',
    'demo_replay.py': '''
        import os, time
        import settings
        from motion_control import log
        def replay_demo(robot, mouse, demo, ask_start_pose=True, strict=True, motion_control=None, speed_factor=1.):
            log({"event": "replay", "demo": demo, "ask_start_pose": ask_start_pose, "speed": speed_factor,
                 "demo_dir": settings.DEMO_DIR})
            if os.environ.get("FAKE_BLOCK"):
                print("REPLAYING", flush=True)
                for _ in range(500):
                    motion_control.check(); time.sleep(.01)
            return {"status": "completed", "completed_samples": 2, "total_samples": 2, "gripper_events": [{}]}
    ''',
    'teleop_setup/__init__.py': '',
    'teleop_setup/robots/__init__.py': '',
    'teleop_setup/robots/ur5e_wrapper.py': '''
        import os
        from motion_control import log
        class Interface:
            def getActualTCPPose(self): return [float(v) for v in os.environ["FAKE_TCP"].split(",")]
            def disconnect(self): log({"event": "disconnect"})
        class teleop_robot:
            def __init__(self, robot_ip):
                log({"event": "connect", "ip": robot_ip})
                self.robot_ip = robot_ip; self.rtde_c = Interface(); self.rtde_r = Interface()
    ''',
}
ALIGNED = [-.4, .3, .3, 0., 3.1, 0.]


class SkillRunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root = Path(tmp.name)
        self.dinobot = self.root / 'dinobot'
        for name, text in FAKE.items():
            path = self.dinobot / name;path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text))
        self.log = self.root / 'events.jsonl';self.result = self.root / 'result.json'

    def run_runner(self, *, tcp=ALIGNED, ip='10.0.0.1', block=False, stop_after=None):
        env = {'FAKE_LOG': str(self.log), 'FAKE_TCP': ','.join(map(str, tcp)), 'PATH': '/usr/bin:/bin'}
        if block:env['FAKE_BLOCK'] = '1'
        command = [sys.executable, str(RUNNER), '--dinobot', str(self.dinobot), '--demos-root', str(self.root / 'demos'),
                   '--demo', 'hmi_demo', '--robot-ip', ip, '--expected-start', json.dumps(ALIGNED),
                   '--tolerance-m', '0.002', '--tolerance-deg', '1', '--result', str(self.result)]
        process = subprocess.Popen(command, cwd=self.dinobot, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if stop_after:
            self.assertEqual(process.stdout.readline().strip(), 'REPLAYING')
            process.send_signal(signal.SIGINT)
        process.communicate(timeout=30)
        events = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return process.returncode, json.loads(self.result.read_text()), [e['event'] for e in events], events

    def test_replays_from_current_aligned_tcp_with_dinobot_modules(self):
        code, result, names, events = self.run_runner()
        self.assertEqual(code, 0, result);self.assertEqual(result['status'], 'completed')
        self.assertEqual(names, ['connect', 'prepare', 'replay', 'disconnect', 'disconnect'])
        replay = events[2]
        self.assertFalse(replay['ask_start_pose']);self.assertEqual(replay['demo'], 'hmi_demo')
        self.assertEqual(Path(replay['demo_dir']), (self.root / 'demos').resolve())
        self.assertEqual(result['replay']['gripper_events'], 1)

    def test_tcp_away_from_aligned_pose_never_replays(self):
        code, result, names, _ = self.run_runner(tcp=[ALIGNED[0] + .01, *ALIGNED[1:]])
        self.assertEqual(code, 1);self.assertIn('away from the aligned pose', result['error'])
        self.assertNotIn('replay', names);self.assertIn('disconnect', names)

    def test_wrong_robot_never_connects(self):
        code, result, names, _ = self.run_runner(ip='10.0.0.2')
        self.assertEqual(code, 1);self.assertIn('not 10.0.0.2', result['error']);self.assertEqual(names, [])
        self.assertFalse(result['robot_commands_sent'])

    def test_sigint_requests_stopj_and_reports_stopped(self):
        code, result, names, events = self.run_runner(block=True, stop_after=True)
        self.assertEqual(code, 1);self.assertEqual(result['status'], 'stopped')
        self.assertIn('stopJ', names);self.assertEqual(next(e for e in events if e['event'] == 'stopJ')['deceleration'], 2.0)
        self.assertEqual(names[-2:], ['disconnect', 'disconnect'])


if __name__ == '__main__':
    unittest.main()
