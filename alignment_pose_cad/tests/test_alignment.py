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
from alignment_pose_cad.robot import RTDERobot, JointPathError
from alignment_pose_cad.service import AlignmentService


def config():
    cfg, _ = load()
    # This fixture explicitly tests the legacy path with synthetic transforms.
    # Production defaults use the shared hand-eye YAML in preview-only mode.
    cfg['robot'].update(hand_eye_file='', preview_only=False, execution_mode='validated')
    cfg['alignment'].update(orientation_mode='plane_frame', plane_normal_cad=[0, 0, -1], plane_x_cad=[1, 0, 0], plane_origin_cad_mm=[0,0,0])
    cfg['robot']['tcp_from_camera_m_rad'] = [0,0,0,0,0,0]
    cfg['motion'].update(max_translation_m=3, max_rotation_deg=180, workspace_min_m=[-5,-5,-5], workspace_max_m=[5,5,5])
    return cfg


def visual():
    T = np.eye(4);T[:3,3] = [100,200,700]
    return T


class GeometryTests(unittest.TestCase):
    def test_180_degree_allowance_has_no_axis_or_downward_direction_restriction(self):
        cfg=config();cfg['alignment'].update(plane_normal_cad=[0,1,0],orientation_mode='plane_frame')
        mapping=np.array([[1,0,0],[0,0,-1],[0,1,0]])
        T=visual();T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        for axis in 'xyz':
            for angle in (0,45,120,179.9,180):
                tcp_camera=np.eye(4)
                tcp_camera[:3,:3]=Rotation.from_euler(axis,angle,degrees=True).as_matrix()@(T[:3,:3]@mapping).T
                plan=build_plan(T,[0]*6,cfg,[[-1]*3,[1]*3],'part',tcp_from_camera=tcp_camera)
                self.assertAlmostEqual(plan['rotation_deg'],angle,places=6)
                self.assertFalse(plan['rotation_limit_exceeded'])
                self.assertEqual(plan['rotation_limit_deg'],180)

    def test_positive_tcp_y_aligns_with_positive_cad_z_and_tool_z_points_inward(self):
        for tcp in ([0]*6, [.2,-.1,.3,.3,-.7,1.1]):
            cfg=config()
            cfg['alignment'].update(orientation_mode='plane_frame',plane_normal_cad=[0,1,0],
                                    plane_x_cad=[1,0,0],tool_yaw_deg=0)
            cfg['motion']['max_rotation_deg']=90
            mapping=np.array([[1,0,0],[0,0,-1],[0,1,0]])
            T=visual();T[:3,:3]=Rotation.from_euler('z',-55,degrees=True).as_matrix()@mapping.T
            p=build_plan(T,tcp,cfg,[[-72.2,0,26],[72.2,30,231.5]],'02_control_panel')
            cad=np.asarray(p['T_base_from_cad']);goal=np.asarray(p['T_base_from_tcp_target'])
            np.testing.assert_allclose(goal[:3,0],cad[:3,0],atol=1e-10)
            np.testing.assert_allclose(goal[:3,1],cad[:3,2],atol=1e-10)
            np.testing.assert_allclose(goal[:3,2],-cad[:3,1],atol=1e-10)
            np.testing.assert_allclose(goal[:3,3],(cad@np.array([0,.23,.12875,1]))[:3],atol=1e-10)
            self.assertAlmostEqual(p['rotation_deg'],55)
            self.assertAlmostEqual(np.linalg.det(goal[:3,:3]),1)
            self.assertFalse(p['rotation_limit_exceeded'])

    def test_inward_axes_need_only_the_expected_tool_z_rotation(self):
        cfg=config()
        cfg['alignment'].update(orientation_mode='cad_axes_inward',plane_normal_cad=[0,1,0])
        cfg['motion']['max_rotation_deg']=90
        mapping=np.array([[0,1,0],[0,0,-1],[-1,0,0]])
        T=visual();T[:3,:3]=Rotation.from_euler('z',-35,degrees=True).as_matrix()@mapping.T
        p=build_plan(T,[0]*6,cfg,[[-72.2,0,26],[72.2,30,231.5]],'02_control_panel')
        cad=np.asarray(p['T_base_from_cad']);goal=np.asarray(p['T_base_from_tcp_target'])
        np.testing.assert_allclose(-goal[:3,0],cad[:3,2],atol=1e-10)
        np.testing.assert_allclose(goal[:3,1],cad[:3,0],atol=1e-10)
        np.testing.assert_allclose(goal[:3,2],-cad[:3,1],atol=1e-10)
        np.testing.assert_allclose(goal[:3,3],(cad@np.array([0,.23,.12875,1]))[:3],atol=1e-10)
        np.testing.assert_allclose(p['tcp_target_m_rad'][3:],[0,0,np.deg2rad(-35)],atol=1e-10)
        self.assertAlmostEqual(p['rotation_deg'],35)
        self.assertFalse(p['rotation_limit_exceeded'])
        cfg['alignment']['orientation_mode']='cad_axes'
        with self.assertRaisesRegex(ValueError,'step limit'):
            build_plan(T,[0]*6,cfg,[[-1]*3,[1]*3],'02_control_panel')

    def test_requested_tcp_axes_use_original_cad_axes_and_keep_front_center(self):
        for tcp in ([0]*6, [.2,-.1,.3,.3,-.7,1.1]):
            cfg=config()
            cfg['alignment'].update(orientation_mode='cad_axes',plane_normal_cad=[0,1,0],tool_yaw_deg=37)
            cfg['robot']['tcp_from_camera_m_rad']=[.03,.04,-.02,.2,.1,-.3]
            T=visual();T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
            p=build_plan(T,tcp,cfg,[[-72.2,0,26],[72.2,30,231.5]],'02_control_panel')
            cad=np.asarray(p['T_base_from_cad']);goal=np.asarray(p['T_base_from_tcp_target'])
            np.testing.assert_allclose(-goal[:3,0],cad[:3,2],atol=1e-10)
            np.testing.assert_allclose(-goal[:3,1],cad[:3,0],atol=1e-10)
            np.testing.assert_allclose(goal[:3,2],cad[:3,1],atol=1e-10)
            np.testing.assert_allclose(goal[:3,3],(cad@np.array([0,.23,.12875,1]))[:3],atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(goal[:3,:3]),1.)
            self.assertEqual(p['speed_m_s'],cfg['motion']['speed_m_s'])
        cfg['motion']['max_rotation_deg']=1
        with self.assertRaisesRegex(ValueError,'step limit'):
            build_plan(T,tcp,cfg,[[-1]*3,[1]*3],'02_control_panel')

    def test_minimum_z_preserves_already_aligned_yaw_and_uses_cad_origin(self):
        cfg=config()
        cfg['alignment'].update(orientation_mode='minimum_z', plane_normal_cad=[0,0,1])
        T=visual(); T[:3,:3]=Rotation.from_euler('x',180,degrees=True).as_matrix()
        tcp=[0,0,0,0,0,1.2]
        plan=build_plan(T,tcp,cfg,[[-1,-1,-1],[1,1,1]],'part')
        cad=np.asarray(plan['T_base_from_cad']);goal=np.asarray(plan['T_base_from_tcp_target'])
        np.testing.assert_allclose(goal[:3,3],cad[:3,3]+.2*cad[:3,2],atol=1e-10)
        np.testing.assert_allclose(goal[:3,:3],from_pose(tcp)[:3,:3],atol=1e-10)
        self.assertAlmostEqual(plan['rotation_deg'],0.)

    def test_hmi_override_targets_front_center_plus_200mm(self):
        cfg=config()
        cfg['alignment'].update(orientation_mode='minimum_z',plane_normal_cad=[0,1,0])
        cfg['alignment']['cad_overrides']['02_control_panel']={'plane_origin_cad_mm':[0,30,128.75]}
        T=visual();T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        plan=build_plan(T,[0,0,0,0,0,0],cfg,[[-72.2,0,26],[72.2,30,231.5]],'02_control_panel')
        cad=np.asarray(plan['T_base_from_cad']);goal=np.asarray(plan['T_base_from_tcp_target'])
        np.testing.assert_allclose(goal[:3,3],(cad@np.array([0,.23,.12875,1]))[:3],atol=1e-10)
        np.testing.assert_allclose(goal[:3,2],-cad[:3,1],atol=1e-10)

    def test_xz_plane_standoff_and_minimum_tool_z_against_cad_y(self):
        cfg=config()
        cfg['alignment'].update(orientation_mode='minimum_z',plane_normal_cad=[0,1,0])
        T=visual();T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        plan=build_plan(T,[0,0,0,0,0,.7],cfg,[[-1,-1,-1],[1,1,1]],'part')
        cad=np.asarray(plan['T_base_from_cad']);goal=np.asarray(plan['T_base_from_tcp_target'])
        np.testing.assert_allclose(goal[:3,3],cad[:3,3]+.2*cad[:3,1],atol=1e-10)
        np.testing.assert_allclose(goal[:3,2],-cad[:3,1],atol=1e-10)
        self.assertAlmostEqual(plan['rotation_deg'],0.)

    def test_z_alignment_uses_shortest_tilt_and_handles_opposite_axes(self):
        from alignment_pose_cad.geometry import align_z_minimum_rotation
        current=Rotation.from_euler('z',73,degrees=True).as_matrix()
        desired=Rotation.from_euler('x',25,degrees=True).apply([0,0,1])
        target=align_z_minimum_rotation(current,desired)
        np.testing.assert_allclose(target[:,2],desired,atol=1e-10)
        self.assertAlmostEqual(np.rad2deg(Rotation.from_matrix(target@current.T).magnitude()),25.)
        target=align_z_minimum_rotation(current,[0,0,-1])
        np.testing.assert_allclose(target[:,2],[0,0,-1],atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(target),1.)

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
        limit=self.robot.cfg['motion']['max_joint_delta_rad']
        for field,value in [('safe',False),('ik',False),('qsafe',False),('branch',[limit+.5]*6)]:
            with self.subTest(field=field):
                old=getattr(self.robot.control,field);setattr(self.robot.control,field,value)
                with self.assertRaises(RuntimeError):self.robot.move(self.plan,self.ref)
                setattr(self.robot.control,field,old)
        self.assertFalse(any(name=='moveL' for name,args in self.robot.control.calls))

    def test_stop_before_dispatch_never_moves(self):
        self.robot.stop()
        with self.assertRaises(RuntimeError):self.robot.move(self.plan,self.ref)
        self.assertEqual([n for n,_ in self.robot.control.calls],['stopL'])

    def test_direct_move_respects_configured_rotation_limit(self):
        plan={**self.plan,'tcp_target_m_rad':[0,0,.2,np.pi,0,0],
              'path_samples_m_rad':[[0,0,.2*t,np.pi*t,0,0] for t in np.linspace(0,1,21)]}
        self.robot.cfg['motion']['max_rotation_deg']=90
        with self.assertRaisesRegex(RuntimeError,'angular allowance'):
            self.robot.move(plan,self.ref)
        self.assertFalse(self.robot.control.calls)
        self.robot.cfg['motion']['max_rotation_deg']=180
        def dispatch(target,*args):
            self.robot.control.calls.append(('moveL',target))
            self.robot.receive.pose=target;self.robot.control.progress=-2;return True
        self.robot.control.moveL=dispatch
        self.assertEqual(self.robot.move(plan,self.ref)['status'],'completed')

    def test_joint_excursion_is_distinguished_from_an_adjacent_ik_jump(self):
        limit=self.robot.cfg['motion']['max_joint_delta_rad']
        step=limit/15
        count=[-1]
        def continuous(*args):
            count[0]+=1
            return [0,0,0,0,0,step*count[0]]
        self.robot.control.getInverseKinematics=continuous
        with self.assertRaises(JointPathError) as caught:self.robot.validate_path(self.plan,self.ref)
        detail=caught.exception.details
        self.assertEqual(detail['rejection_type'],'joint_excursion_from_start')
        self.assertEqual(detail['joint_name'],'wrist_3')
        self.assertAlmostEqual(detail['adjacent_delta_rad'][5],step)
        self.assertGreater(detail['from_initial_delta_rad'][5],limit)
        self.assertIn('starting posture',str(caught.exception))
        self.assertFalse(self.robot.control.calls)
        self.robot.control.getInverseKinematics=lambda *args:[0,0,0,0,0,limit+.5]
        with self.assertRaises(JointPathError) as caught:self.robot.validate_path(self.plan,self.ref)
        self.assertEqual(caught.exception.details['rejection_type'],'adjacent_joint_step')
        self.assertEqual(caught.exception.details['sample_index'],0)
        self.assertIn('joint 6',str(caught.exception))
        self.assertFalse(self.robot.control.calls)

    def test_authorization_revoked_during_controller_checks_never_moves(self):
        def expired():raise ValueError('The plan expired during checks')
        with self.assertRaisesRegex(ValueError,'expired'):
            self.robot.move(self.plan,self.ref,before_dispatch=expired)
        self.assertFalse(self.robot.control.calls)

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

    def test_check_path_in_preview_sends_no_motion_and_rejects_unsafe_path(self):
        self.cfg['robot'].update(preview_only=True,motion_enabled=False,calibration_confirmed=False)
        self.file.write_text(json.dumps(self.cfg));self.service.robot.cfg=self.cfg
        plan=self.plan();plan.update(reference=self.service.robot.snapshot(stationary=True),
                                    path_samples_m_rad=[[0]*6,[.01,0,0,0,0,0]])
        self.service.plan=plan
        self.service.action('/api/alignment/check-path',{'plan_id':'preview'})
        self.service.pool.submit(lambda:None).result(timeout=3)
        self.assertTrue(plan['controller_path_checked'])
        self.assertFalse(plan['controller_check']['motion_sent'])
        self.assertFalse(plan['controller_check']['collision_checked'])
        self.assertEqual(self.service.robot.control.calls,[])
        with self.assertRaisesRegex(ValueError,'Preview-only'):
            self.service._authorize_motion(self.cfg,plan)
        self.service.robot.control.safe=False
        self.service.action('/api/alignment/check-path',{'plan_id':'preview'})
        self.service.pool.submit(lambda:None).result(timeout=3)
        self.assertFalse(plan['controller_path_checked'])
        self.assertIn('rejects',plan['controller_check_error'])
        self.assertEqual(self.service.robot.control.calls,[])

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


# --- Alignment to recorded skill demos -------------------------------------
import hashlib
import pickle
import yaml
from alignment_pose_cad.geometry import build_demo_plan
from alignment_pose_cad import skill_demos


def make_demo(root, demo_id, cad_key, camera_from_cad_mm, tcp_pose, *, cad_sha256='cad-hash',
              tcp_offset=(0, 0, .15, 0, 0, 0), robot_id='192.168.1.11', image=b'rgb'):
    """A synthetic DINOBOT demo with the real file layout and hash links."""
    folder = Path(root) / 'demos' / demo_id;folder.mkdir(parents=True, exist_ok=True)
    (folder/'btn_rgb.png').write_bytes(image);(folder/'btn_depth.npy').write_bytes(b'depth-'+image)
    (folder/'rel_trajectory.pkl').write_bytes(pickle.dumps([([0.]*6, 0)]))
    hashes = {n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('btn_rgb.png','btn_depth.npy','rel_trajectory.pkl')}
    start = {'schema_version': 1, 'frame': 'robot_base', 'pose': list(tcp_pose), 'replay_start_pose': list(tcp_pose),
             'geometry': {'base_frame_id': '', 'robot_id': robot_id, 'tcp_offset': list(tcp_offset), 'camera_id': '0',
                          'camera_offset': [0, .115, -.13, .78, 0, 0]}, 'file_hashes': hashes}
    (folder/'trajectory_meta.json').write_text(json.dumps({'servo_dt': .01, 'reference_start': start}))
    pose_folder = folder/'sam6d_foundationpose_20261109';pose_folder.mkdir(exist_ok=True)
    (pose_folder/'source_provenance.json').write_text(json.dumps({'cad_key': cad_key, 'cad_sha256': cad_sha256,
        'original_sha256': {n: hashes[n] for n in ('btn_rgb.png','btn_depth.npy')}}))
    (pose_folder/'cad_pose_camera.json').write_text(json.dumps({'demo': demo_id, 'cad_key': cad_key, 'cad_name': cad_key,
        'status': 'estimated', 'T_camera_from_input_cad_mm': np.asarray(camera_from_cad_mm).tolist(), 'sam6d_score': .5}))
    return folder


def write_bindings(root, demos):
    path = Path(root)/'pddl_to_robot.yaml'
    path.write_text(yaml.safe_dump({'schema_version': 1, 'demos_root': 'demos', 'bindings': {'assembly': {'remove': {
        part: {'skill_id': 'remove_'+part+'_v1', 'demo_id': demo} for part, demo in demos.items()}}}}))
    return path


def random_pose(seed, scale=.3):
    rng = np.random.default_rng(seed)
    return np.r_[rng.uniform(-scale, scale, 3), Rotation.random(random_state=seed).as_rotvec()].tolist()


class SkillDemoGeometryTests(unittest.TestCase):
    def scene(self, moved):
        """Demo and current captures of one object, consistent with a hand-eye camera."""
        tcp_camera = from_pose([.01, .115, -.13, .78, .05, -.02])
        demo_tcp = from_pose([-.43, .34, .28, -2.55, 1.78, -.06])
        cad_demo = demo_tcp @ tcp_camera @ np.r_[np.c_[Rotation.from_euler('xyz', [80, 5, -10], degrees=True).as_matrix(),
                                                         [.02, .1, .39]], [[0, 0, 0, 1]]]
        cad_now = moved @ cad_demo
        def camera_from_cad(base_tcp, cad):
            T = np.linalg.inv(base_tcp @ tcp_camera) @ cad;T[:3, 3] *= 1000;return T
        demo = {'demo_id': 'd', 'skills': [], 'pose_folder': 'p', 'cad_key': 'k', 'sam6d_score': .5,
                'tcp_offset_m_rad': [0, 0, .15, 0, 0, 0], 'tcp_pose_m_rad': to_pose(demo_tcp),
                'T_camera_from_input_cad_mm': camera_from_cad(demo_tcp, cad_demo).tolist()}
        return tcp_camera, demo_tcp, cad_now, camera_from_cad, demo

    def test_target_reproduces_demo_tcp_relative_to_moved_cad_from_any_capture_pose(self):
        moved = from_pose([.07, -.05, .01, 0, 0, .4])
        tcp_camera, demo_tcp, cad_now, camera_from_cad, demo = self.scene(moved)
        cfg = config()
        for capture in ([-.40, .30, .30, -2.5, 1.8, 0], [-.38, .36, .33, -2.4, 1.9, -.2]):
            capture_tcp = from_pose(capture)
            plan = build_demo_plan(camera_from_cad(capture_tcp, cad_now), capture, cfg, demo, tcp_from_camera=tcp_camera)
            goal = np.asarray(plan['T_base_from_tcp_target'])
            np.testing.assert_allclose(goal, moved @ demo_tcp, atol=1e-9)
            # Same relative TCP/CAD pose as in the demo.
            np.testing.assert_allclose(np.linalg.inv(goal) @ cad_now,
                                       np.linalg.inv(demo_tcp) @ np.linalg.inv(moved) @ cad_now, atol=1e-9)
            np.testing.assert_allclose(plan['T_base_from_cad_demo'], np.linalg.inv(moved) @ cad_now, atol=1e-9)
            self.assertAlmostEqual(plan['object_rotation_from_demo_deg'], np.rad2deg(.4), places=6)
            self.assertEqual(plan['alignment_mode'], 'skill_demo')
            self.assertEqual(len(plan['path_samples_m_rad']), cfg['motion']['path_samples'])
            np.testing.assert_allclose(plan['path_samples_m_rad'][0], to_pose(capture_tcp), atol=1e-9)

    def test_unmoved_object_returns_exactly_to_the_demo_tcp(self):
        tcp_camera, demo_tcp, cad_now, camera_from_cad, demo = self.scene(np.eye(4))
        capture = from_pose([-.40, .31, .30, -2.5, 1.8, 0])
        plan = build_demo_plan(camera_from_cad(capture, cad_now), to_pose(capture), config(), demo, tcp_from_camera=tcp_camera)
        np.testing.assert_allclose(plan['tcp_target_m_rad'], demo['tcp_pose_m_rad'], atol=1e-9)
        self.assertAlmostEqual(plan['object_displacement_from_demo_m'], 0, places=9)

    def test_demo_target_keeps_step_and_workspace_limits(self):
        moved = from_pose([.5, 0, 0, 0, 0, 0])
        tcp_camera, demo_tcp, cad_now, camera_from_cad, demo = self.scene(moved)
        cfg = config();cfg['motion']['max_translation_m'] = .3
        capture = from_pose([-.40, .30, .30, -2.5, 1.8, 0])
        with self.assertRaisesRegex(ValueError, 'step limit'):
            build_demo_plan(camera_from_cad(capture, cad_now), to_pose(capture), cfg, demo, tcp_from_camera=tcp_camera)
        behind = copy.deepcopy(demo);behind['T_camera_from_input_cad_mm'][2][3] = -10
        with self.assertRaisesRegex(ValueError, 'behind'):
            build_demo_plan(camera_from_cad(capture, cad_now), to_pose(capture), cfg, behind, tcp_from_camera=tcp_camera)


class SkillDemoFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root = Path(self.tmp.name)
        self.pose = visual()
        make_demo(self.root, 'hmi_demo', '02_control_panel', self.pose, [-.4, .3, .3, 0, 3.1, 0])
        make_demo(self.root, 'cassette_demo', '03_option_cassette', self.pose, [-.4, .3, .3, 0, 3.1, 0], image=b'other')
        self.bindings = write_bindings(self.root, {'hmi_panel': 'hmi_demo', 'option_cassette': 'cassette_demo'})

    def test_catalog_and_selection_by_detected_cad(self):
        entries = {e['demo_id']: e for e in skill_demos.catalog(self.bindings)}
        self.assertEqual(entries['hmi_demo']['cad_key'], '02_control_panel');self.assertTrue(entries['hmi_demo']['available'])
        self.assertEqual(entries['hmi_demo']['skills'][0]['skill_id'], 'remove_hmi_panel_v1')
        demo = skill_demos.select(self.bindings, '02_control_panel')
        self.assertEqual(demo['demo_id'], 'hmi_demo');self.assertEqual(demo['tcp_pose_m_rad'], [-.4, .3, .3, 0, 3.1, 0])
        np.testing.assert_allclose(demo['T_camera_from_input_cad_mm'], self.pose)
        with self.assertRaisesRegex(ValueError, 'No skill demo'):skill_demos.select(self.bindings, '07_thermal_core')
        with self.assertRaisesRegex(ValueError, 'not bound'):skill_demos.load(self.bindings, 'unbound')

    def test_ambiguous_cad_requires_an_explicit_demo(self):
        make_demo(self.root, 'hmi_second', '02_control_panel', self.pose, [-.4, .3, .3, 0, 3.1, 0], image=b'2')
        bindings = write_bindings(self.root, {'a': 'hmi_demo', 'b': 'hmi_second'})
        with self.assertRaisesRegex(ValueError, 'Several'):skill_demos.select(bindings, '02_control_panel')
        self.assertEqual(skill_demos.select(bindings, '02_control_panel', 'hmi_second')['demo_id'], 'hmi_second')

    def test_rerecorded_demo_rejects_the_stale_pose(self):
        # Re-recording rewrites the image and reference_start, but not the pose.
        make_demo(self.root, 'fresh', '02_control_panel', self.pose, [-.4, .3, .3, 0, 3.1, 0], image=b'old')
        folder = self.root/'demos'/'fresh';old = (folder/'sam6d_foundationpose_20261109'/'source_provenance.json').read_text()
        make_demo(self.root, 'fresh', '02_control_panel', self.pose, [-.4, .3, .3, 0, 3.1, 0], image=b'new')
        (folder/'sam6d_foundationpose_20261109'/'source_provenance.json').write_text(old)
        bindings = write_bindings(self.root, {'p': 'fresh'})
        with self.assertRaisesRegex(ValueError, 'different reference image'):skill_demos.load(bindings, 'fresh')

    def test_changed_image_without_new_reference_and_later_changes_are_detected(self):
        demo = skill_demos.load(self.bindings, 'hmi_demo')
        skill_demos.verify(demo)
        (self.root/'demos'/'hmi_demo'/'btn_rgb.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed after'):skill_demos.verify(demo)
        with self.assertRaises(ValueError):skill_demos.load(self.bindings, 'hmi_demo')
