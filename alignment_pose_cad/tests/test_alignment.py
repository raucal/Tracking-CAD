"""Geometry and robot interlocks tested with fakes; never connect to real hardware."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alignment_pose_cad.config import load, DEFAULT
from alignment_pose_cad.geometry import build_plan, from_pose, to_pose, difference, plane_frame
from alignment_pose_cad.robot import RTDERobot
from alignment_pose_cad.service import AlignmentService


def config():
    cfg, _ = load()
    cfg['alignment'].update(plane_normal_cad=[0, 0, -1], plane_x_cad=[1, 0, 0], plane_origin_cad_mm=[0,0,0])
    cfg['robot']['tcp_from_camera_m_rad'] = [0,0,0,0,0,0]
    cfg['motion'].update(max_translation_m=3, max_rotation_deg=180, workspace_min_m=[-5,-5,-5], workspace_max_m=[5,5,5])
    return cfg


def visual():
    T = np.eye(4);T[:3,3] = [100,200,700]
    return T


class GeometryTests(unittest.TestCase):
    def test_200mm_standoff_is_along_cad_normal_not_base_z(self):
        cfg=config();tcp=[.2,-.1,.4,.3,.2,-.4]
        cfg['robot']['tcp_from_camera_m_rad']=[.02,.11,-.13,.78,0,0]
        p=build_plan(visual(),tcp,cfg,[[-1,-1,-1],[1,1,1]],'part')
        plane=np.asarray(p['T_base_from_plane']);goal=np.asarray(p['T_base_from_tcp_target'])
        np.testing.assert_allclose(goal[:3,3]-plane[:3,3],plane[:3,2]*.2,atol=1e-10)
        np.testing.assert_allclose(goal[:3,2],-plane[:3,2],atol=1e-10)
        self.assertFalse(np.allclose(goal[:3,3]-plane[:3,3],[0,0,.2]))
        self.assertAlmostEqual(np.linalg.norm(goal[:3,3]-plane[:3,3]),.2)

    def test_mm_conversion_and_extrinsic_order(self):
        cfg=config();cfg['robot']['tcp_from_camera_m_rad']=[.03,0,0,0,0,np.pi/2]
        p=build_plan(visual(),[1,0,0,0,0,0],cfg,[[-1,-1,-1],[1,1,1]],'part')
        np.testing.assert_allclose(np.asarray(p['T_base_from_cad'])[:3,3],[.83,.1,.7],atol=1e-10)
        np.testing.assert_allclose(np.asarray(p['T_base_from_tcp_target'])[:3,3],[.83,.1,.5],atol=1e-10)

    def test_face_center_preserves_original_cad_origin(self):
        cfg=config()['alignment'];cfg.update(plane_normal_cad=[0,1,0],plane_origin_cad_mm=None)
        T=plane_frame(cfg,[[-110,0,0],[110,28,250]])
        np.testing.assert_allclose(T[:3,3],[0,.028,.125])
        np.testing.assert_allclose(T[:3,2],[0,1,0])
        self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1)

    def test_invalid_pose_wrong_normal_and_parallel_axes_rejected(self):
        cfg=config();bad=visual();bad[0,0]=2
        with self.assertRaises(ValueError):build_plan(bad,[0]*6,cfg,[[-1]*3,[1]*3],'x')
        cfg['alignment']['plane_normal_cad']=[0,0,1]
        with self.assertRaisesRegex(ValueError,'away'):build_plan(visual(),[0]*6,cfg,[[-1]*3,[1]*3],'x')
        cfg['alignment']['plane_x_cad']=[0,0,1]
        with self.assertRaisesRegex(ValueError,'parallel'):plane_frame(cfg['alignment'],[[-1]*3,[1]*3])

    def test_workspace_step_limits_and_oblique_plane_fail_closed(self):
        cfg=config();cfg['motion']['max_translation_m']=.01
        with self.assertRaisesRegex(ValueError,'step'):build_plan(visual(),[0]*6,cfg,[[-1]*3,[1]*3],'x')
        cfg=config();cfg['motion']['workspace_max_m']=[.01,.01,.01]
        with self.assertRaisesRegex(ValueError,'workspace'):build_plan(visual(),[0]*6,cfg,[[-1]*3,[1]*3],'x')
        cfg=config()['alignment'];cfg.update(plane_normal_cad=[0,1,1],plane_origin_cad_mm=None)
        with self.assertRaisesRegex(ValueError,'oblique'):plane_frame(cfg,[[-1]*3,[1]*3])

    def test_cad_override_and_yaw_remain_at_20cm(self):
        cfg=config();cfg['alignment']['cad_overrides']={'part':{'in_plane_offset_m':[.01,.02],'tool_yaw_deg':30}}
        p=build_plan(visual(),[0]*6,cfg,[[-1]*3,[1]*3],'part')
        T=np.asarray(p['T_plane_from_tcp_target'])
        np.testing.assert_allclose(T[:3,3],[.01,.02,.2])
        self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1)


class FakeReceive:
    def __init__(self,*args):
        self.connected=True;self.pose=[0]*6;self.q=[0]*6;self.speed=[0]*6;self.qd=[0]*6;self.bits=1;self.mode=7
    def disconnect(self):self.connected=False
    def isConnected(self):return self.connected
    def getSafetyStatusBits(self):return self.bits
    def getRobotMode(self):return self.mode
    def getTimestamp(self):return time.monotonic()
    def getActualTCPPose(self):return self.pose[:]
    def getActualQ(self):return self.q[:]
    def getActualTCPSpeed(self):return self.speed[:]
    def getActualQd(self):return self.qd[:]


class FakeControl:
    def __init__(self,*args):
        self.connected=True;self.calls=[];self.progress=-1;self.running=True;self.safe=True;self.ik=True;self.branch=[0]*6;self.qsafe=True;self.tcp=[0,0,.15,0,0,0]
    def disconnect(self):self.connected=False
    def isConnected(self):return self.connected
    def getTCPOffset(self):return self.tcp
    def isProgramRunning(self):return self.running
    def isPoseWithinSafetyLimits(self,p):return self.safe
    def getInverseKinematicsHasSolution(self,*args):return self.ik
    def getInverseKinematics(self,*args):return self.branch
    def isJointsWithinSafetyLimits(self,q):return self.qsafe
    def getAsyncOperationProgress(self):return self.progress
    def moveL(self,*args):self.calls.append(('moveL',args));return True
    def stopL(self,*args):self.calls.append(('stopL',args))


class RobotTests(unittest.TestCase):
    def setUp(self):
        self.robot=RTDERobot(config(),FakeReceive,FakeControl);self.robot.connect()
        self.addCleanup(self.robot.disconnect)
        self.ref=self.robot.snapshot(stationary=True)
        self.plan=build_plan(visual(),[0]*6,self.robot.cfg,[[-1]*3,[1]*3],'x')

    def test_constructor_never_opens_hardware(self):
        r=RTDERobot(config());self.assertFalse(r.connected)

    def test_drift_moving_fault_and_tcp_change_reject(self):
        self.robot.receive.pose[0]=.01
        with self.assertRaisesRegex(RuntimeError,'moved'):self.robot.check_reference(self.ref)
        self.robot.receive.pose[0]=0;self.robot.receive.speed[0]=.02
        with self.assertRaisesRegex(RuntimeError,'stationary'):self.robot.snapshot(stationary=True)
        self.robot.receive.speed[0]=0;self.robot.receive.bits=4
        with self.assertRaisesRegex(RuntimeError,'safety'):self.robot.snapshot()
        self.robot.receive.bits=1;self.robot.control.tcp=[0]*6
        with self.assertRaisesRegex(RuntimeError,'active TCP'):self.robot.validate_path(self.plan,self.ref)

    def test_path_rejection_or_joint_branch_never_moves(self):
        for field,value in [('safe',False),('ik',False),('qsafe',False),('branch',[2]*6)]:
            with self.subTest(field=field):
                old=getattr(self.robot.control,field);setattr(self.robot.control,field,value)
                with self.assertRaises(RuntimeError):self.robot.move(self.plan,self.ref)
                setattr(self.robot.control,field,old)
        self.assertFalse(any(name=='moveL' for name,args in self.robot.control.calls))

    def test_stop_before_dispatch_never_moves(self):
        self.robot.stop()
        with self.assertRaises(RuntimeError):self.robot.move(self.plan,self.ref)
        self.assertEqual([n for n,_ in self.robot.control.calls],['stopL'])

    def test_accepted_motion_waits_for_feedback(self):
        c=self.robot.control;r=self.robot.receive
        def dispatch(p,*args):
            c.calls.append(('moveL',p));r.pose=p;c.progress=-2;return True
        c.moveL=dispatch
        result=self.robot.move(self.plan,self.ref)
        self.assertEqual(result['status'],'completed');self.assertAlmostEqual(result['position_error_m'],0)

    def test_stop_during_motion_brakes(self):
        c=self.robot.control
        def dispatch(*args):c.calls.append(('moveL',args));self.robot.cancel.set();return True
        c.moveL=dispatch
        with self.assertRaisesRegex(RuntimeError,'stopped'):self.robot.move(self.plan,self.ref)
        self.assertEqual([n for n,_ in c.calls],['moveL','stopL'])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.file=Path(self.tmp.name)/'settings.json';self.cfg=config()
        self.cfg['robot'].update(calibration_confirmed=True,motion_enabled=True,camera_serial='123')
        self.file.write_text(json.dumps(self.cfg))
        self.app=SimpleNamespace(tracker=SimpleNamespace(busy=lambda:False))
        self.service=AlignmentService(self.app,self.file,lambda cfg:RTDERobot(cfg,FakeReceive,FakeControl))
        self.addCleanup(self.service.close)
        self.service.robot=RTDERobot(self.cfg,FakeReceive,FakeControl);self.service.robot.connect()

    def plan(self):
        _,digest=load(self.file)
        return {'id':'preview','settings_sha256':digest,'camera_serial':'123','planned_at':time.time(),
                'capture_timestamp':time.time(),'artifact_folder':self.tmp.name}

    def test_expiry_serial_calibration_and_settings_changes_block(self):
        p=self.plan();p['planned_at']-=100
        with self.assertRaisesRegex(ValueError,'expired'):self.service._authorize_motion(self.cfg,p)
        p=self.plan();p['camera_serial']='other'
        with self.assertRaisesRegex(ValueError,'camera'):self.service._authorize_motion(self.cfg,p)
        self.cfg['robot']['calibration_confirmed']=False
        with self.assertRaisesRegex(ValueError,'calibration'):self.service._authorize_motion(self.cfg,self.plan())
        self.cfg['robot']['calibration_confirmed']=True
        p=self.plan();self.service.plan=p;self.file.write_text(json.dumps({**self.cfg,'alignment':{**self.cfg['alignment'],'offset_m':.3}}))
        with self.assertRaisesRegex(ValueError,'Settings changed'):self.service.action('/api/alignment/execute',{'plan_id':'preview'})
        self.assertFalse(self.service.robot.control.calls)

    def test_cancel_during_estimation_keeps_busy_until_worker_finishes(self):
        self.service._update('estimating','working')
        self.service.action('/api/alignment/stop',{})
        self.assertTrue(self.service.status()['busy'])
        self.assertTrue(self.service.cancel.is_set())
        with self.assertRaises(ValueError):self.service.action('/api/alignment/plan',{})
        self.assertIsNone(self.service.plan)

    def test_execute_consumes_plan_once(self):
        self.service.plan=self.plan()
        with patch.object(self.service.pool,'submit') as submit:
            result=self.service.action('/api/alignment/execute',{'plan_id':'preview'})
            self.assertIsNone(self.service.plan);self.assertEqual(result['state'],'moving')
            submit.assert_called_once()
            with self.assertRaises(ValueError):self.service.action('/api/alignment/execute',{'plan_id':'preview'})
        self.service._update('ready','test finished')

    def test_configuration_validation(self):
        for change in [{'speed_m_s':float('nan')},{'path_samples':True},{'workspace_min_m':[1,1,1],'workspace_max_m':[0,0,0]}]:
            with self.subTest(change=change):
                cfg=copy.deepcopy(self.cfg);cfg['motion'].update(change);self.file.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):load(self.file)


if __name__=='__main__':unittest.main()
