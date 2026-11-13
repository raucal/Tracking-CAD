"""HTTP and RGB-D integration with real storage and fake hardware/inference."""
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
import os
import hashlib
import base64
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import pytest
import yaml
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import trimesh

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pipeline'))
from sam6d_ui.server import App,handler
from sam6d_ui.camera.service import CameraService
from alignment_pose_cad.service import AlignmentService
from alignment_pose_cad.robot import RTDERobot
from alignment_pose_cad.config import load as load_settings, DEFAULT
from test_alignment import config,FakeReceive,FakeControl,make_demo,write_bindings
from alignment_pose_cad.geometry import from_pose


class Source:
    def __init__(self,options):self.number=0
    def read(self):
        time.sleep(.02);self.number+=1
        return {'rgb':np.full((480,640,3),120,dtype='uint8'),'depth':np.full((480,640),700,dtype='uint16'),
                'camera':{'cam_K':[500,0,320,0,500,240,0,0,1],'depth_scale':1,'serial':'123',
                          'rectified':True,'aligned_to':'color','optical_frame':'realsense_color_optical',
                          'dist_coeffs':[0]*5,'width':640,'height':480},
                'number':self.number,'timestamp':time.time(),'device_timestamp_ms':self.number*20}
    def close(self):pass


@pytest.mark.usefixtures("app_dataset")
class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.app=App(self.root/'results',self.root,self.root/'captures')
        self.app.alignment.close()
        self.cfg=config();self.cfg['robot'].update(calibration_confirmed=True,motion_enabled=True,camera_serial='123')
        f=self.root/'settings.json';f.write_text(json.dumps(self.cfg))
        self.app.alignment=AlignmentService(self.app,f,lambda cfg:RTDERobot(cfg,FakeReceive,FakeControl))
        self.app.camera=CameraService(self.root/'captures',Source)
        self.app.camera.start({'width':640,'height':480,'fps':15})
        self.addCleanup(self.app.camera.stop)
        self.addCleanup(lambda:self.app.pool.shutdown(wait=True))
        self.addCleanup(self.app.alignment.close)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.app))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        mesh=trimesh.creation.box(extents=[100,20,150]);mesh.export(self.root/'part.stl')
        self.pose_calls=[]
        def inference(job_id,folder,inputs):
            self.pose_calls.append(inputs)
            T=np.eye(4);T[:3,3]=[100,200,700]
            # Original CAD frame rather than centered-frame telemetry.
            result={'backend':'foundationpose','recognized':True,'cad_key':'uploaded_cad',
                    'pose':{'poses':[{'T_camera_from_input_cad_mm':T.tolist()}],'best_index':0}}
            Image.new('RGB',(640,480)).save(folder/'pose_overlay.png')
            with self.app.lock:self.app.jobs[job_id].update(status='complete',result=result)
        self.app._run=inference

    def request(self,route,payload=None,origin=None):
        headers={}
        if payload is not None:headers['Content-Type']='application/json'
        if origin:headers['Origin']=origin
        req=Request(self.url+route,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
        with urlopen(req,timeout=5) as response:
            raw=response.read()
            return json.loads(raw) if response.headers.get_content_type()=='application/json' else raw

    def wait_idle(self):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            value=self.request('/api/alignment/status')
            if not value['busy']:return value
            time.sleep(.02)
        self.fail('Service did not finish')

    def use_shared_hand_eye(self, serial='123'):
        """Production matrix_calibration.yaml + a copy of its Calibration.xml, bound to the fake camera."""
        production,_=load_settings(DEFAULT)
        source=Path(production['robot']['hand_eye_file'])
        data=yaml.safe_load(source.read_text())
        self.calibration_xml=self.root/'Calibration.xml'
        self.calibration_xml.write_bytes((source.parent/data['calibration_xml']).read_bytes())
        data.update(camera={'serial':serial},calibration_xml=self.calibration_xml.name)
        data.pop('measurements_xml',None)
        self.hand_eye=self.root/'matrix_calibration.yaml'
        self.hand_eye.write_text(yaml.safe_dump(data))
        self.cfg['robot'].update(hand_eye_file=str(self.hand_eye),preview_only=True)
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        from alignment_pose_cad import calibration
        return calibration._core.load_calibration(self.hand_eye)

    def test_status_assets_and_same_origin_guard(self):
        s=self.request('/api/alignment/status');self.assertFalse(s['connected'])
        self.assertIn(b'Detect pose',self.request('/alignment_pose_cad/panel.js'))
        self.assertIn(b'/alignment_pose_cad/panel.js',self.request('/'))
        with self.assertRaises(HTTPError) as exc:self.request('/api/alignment/connect',{},'https://untrusted.example')
        self.assertEqual(exc.exception.code,403)
        self.assertIsNone(self.app.alignment.robot)

    def test_fresh_capture_to_preview_and_single_use_execution(self):
        self.request('/api/alignment/connect',{});self.assertEqual(self.wait_idle()['state'],'ready')
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'},'options':{'segmentor':'fastsam','mode':'rgb','pose_backend':'pem'}})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        plan=state['plan'];folder=Path(plan['artifact_folder'])
        self.assertTrue((folder/'robot_capture.json').exists());self.assertTrue((folder/'robot_alignment_plan.json').exists())
        self.assertEqual(plan['camera_serial'],'123')
        self.assertAlmostEqual(plan['offset_m'],.2)
        options=self.pose_calls[0]['options']
        self.assertEqual((options['segmentor'],options['pose_backend'],options['mode']),('sam','foundationpose','rgbd'))
        self.assertIn('capture_id',self.pose_calls[0])
        self.assertTrue(self.request(plan['image_url']).startswith(b'\x89PNG'))
        self.assertFalse(self.app.alignment.robot.control.calls)
        c=self.app.alignment.robot.control;r=self.app.alignment.robot.receive
        def dispatch(p,*args):c.calls.append(('moveL',p));r.pose=p;c.progress=-2;return True
        c.moveL=dispatch
        self.request('/api/alignment/execute',{'plan_id':plan['id']})
        state=self.wait_idle();self.assertEqual(state['state'],'completed',state)
        self.assertTrue((folder/'robot_alignment_execution.json').exists())
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':plan['id']})
        self.assertEqual(len([n for n,*_ in c.calls if n=='moveL']),1)

    def test_no_detection_does_not_offer_motion(self):
        self.request('/api/alignment/connect',{});self.wait_idle()
        def empty(job_id,folder,inputs):
            with self.app.lock:self.app.jobs[job_id].update(status='complete',result={'backend':'foundationpose','recognized':False})
        self.app._run=empty
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'error');self.assertIsNone(state['plan'])
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_wrong_camera_serial_cannot_produce_a_plan(self):
        self.cfg['robot']['camera_serial']='different-camera'
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        self.request('/api/alignment/connect',{});self.wait_idle()
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'error');self.assertIsNone(state['plan'])
        self.assertIn('camera_serial',state['message'])
        self.assertFalse(self.pose_calls)
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_robot_motion_during_inference_invalidates_visual_result(self):
        self.request('/api/alignment/connect',{});self.wait_idle()
        original=self.app._run
        def displaced(*args):
            original(*args)
            self.app.alignment.robot.receive.pose[0]=.01
        self.app._run=displaced
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'error');self.assertIsNone(state['plan'])
        self.assertIn('moved',state['message'])
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_calibration_of_another_camera_blocks_targets(self):
        self.use_shared_hand_eye(serial='another-camera')
        state=self.request('/api/alignment/status')
        self.assertEqual(state['config']['hand_eye']['robot_model'],'UR5e')
        self.assertEqual(state['config']['camera_serial'],'another-camera')
        self.request('/api/alignment/connect',{});self.wait_idle()
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'error');self.assertIsNone(state['plan'])
        self.assertIn('camera',state['message'])
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_shared_yaml_is_used_for_200mm_preview_and_execution_is_blocked(self):
        data=self.use_shared_hand_eye()
        self.request('/api/alignment/connect',{});self.wait_idle()
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        plan=state['plan']
        np.testing.assert_allclose(plan['T_tcp_from_camera'],data['matrix'],atol=1e-12)
        expected=np.asarray(data['matrix'])@visual_matrix()
        np.testing.assert_allclose(plan['T_base_from_cad'],expected,atol=1e-12)
        target=np.asarray(plan['T_base_from_tcp_target']);plane=np.asarray(plan['T_base_from_plane'])
        np.testing.assert_allclose(target[:3,3]-plane[:3,3],.2*plane[:3,2])
        self.assertTrue(plan['preview_only']);self.assertFalse(plan['controller_path_checked'])
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':plan['id']})
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_yaml_change_invalidates_the_displayed_target(self):
        self.use_shared_hand_eye()
        self.request('/api/alignment/connect',{});self.wait_idle()
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        # A new Calibration.xml (recalibration) invalidates targets computed with the old one.
        self.calibration_xml.write_text(self.calibration_xml.read_text()+'\n')
        self.assertNotEqual(load_settings(self.app.alignment.settings_file)[1],state['plan']['settings_sha256'])
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':state['plan']['id']})
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_motion_flags_cannot_override_a_provisional_yaml(self):
        self.use_shared_hand_eye()
        self.cfg['robot']['preview_only']=False
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        _,digest=load_settings(self.app.alignment.settings_file)
        self.app.alignment.plan={'id':'forged','settings_sha256':digest}
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':'forged'})
        self.assertIsNone(self.app.alignment.robot)

    def setup_trial(self):
        data=self.use_shared_hand_eye()
        production,_=load_settings(DEFAULT)
        self.cfg['alignment']=production['alignment']
        self.cfg['robot'].update(execution_mode='supervised_trial',preview_only=False,
                                 motion_enabled=True,calibration_confirmed=False)
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        manifest=self.root/'trial_manifest.json'
        manifest.write_text(json.dumps({'parts':[{'key':'02_control_panel','source_stl':'part.stl',
              'source_sha256':hashlib.sha256((self.root/'part.stl').read_bytes()).hexdigest()}]}))
        self.app.manifest_path=manifest
        original=self.app._run
        def run(*args):
            original(*args)
            with self.app.lock:
                T=self.app.jobs[args[0]]['result']['pose']['poses'][0]['T_camera_from_input_cad_mm']
                T[1][:3]=[0,0,1];T[2][:3]=[0,-1,0]
        self.app._run=run
        self.request('/api/alignment/connect',{});self.assertEqual(self.wait_idle()['state'],'ready')
        return data

    def test_detect_checks_ik_automatically_and_execute_requires_confirmation(self):
        self.setup_trial()
        for selection in ({'target_cad_key':'wrong'}, {'target_cad_key':'02_control_panel','server_files':{'cad':'part.stl'}}):
            with self.assertRaises(HTTPError):self.request('/api/alignment/plan',{'live_camera':True,**selection})
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        p=state['plan'];self.assertEqual(p['cad_key'],'02_control_panel')
        self.assertFalse(p['preview_only']);self.assertFalse(p['motion_allowed'])
        self.assertTrue(p['controller_path_checked'])
        self.assertNotIn('auto_assembly',self.pose_calls[0])
        self.assertEqual(self.pose_calls[0]['options']['cad_units'],'mm')
        self.assertFalse(state['config']['calibration_confirmed'])
        self.assertEqual(p['hand_eye']['status'],'provisional')
        goal=np.asarray(p['T_base_from_tcp_target']);cad=np.asarray(p['T_base_from_cad'])
        np.testing.assert_allclose(goal[:3,3],(cad@np.array([0,.23,.12875,1]))[:3])
        np.testing.assert_allclose(goal[:3,0],cad[:3,0],atol=1e-10)
        np.testing.assert_allclose(goal[:3,1],cad[:3,2],atol=1e-10)
        np.testing.assert_allclose(goal[:3,2],-cad[:3,1],atol=1e-10)
        payload={'plan_id':p['id'],'confirmed':True}
        self.assertEqual(len(state['plan']['controller_check']['details']['samples']),21)
        for invalid in ({}, {'confirmed':False}, {'confirmed':'true'}):
            with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':p['id'],**invalid})
        self.assertFalse(self.app.alignment.robot.control.calls)
        c=self.app.alignment.robot.control;r=self.app.alignment.robot.receive
        def dispatch(target,*args):c.calls.append(('moveL',target));r.pose=target;c.progress=-2;return True
        c.moveL=dispatch
        self.request('/api/alignment/execute',payload)
        result=self.wait_idle();self.assertEqual(result['state'],'completed',result)
        self.assertFalse(result['result']['accuracy_validated'])
        self.assertFalse(result['result']['collision_checked'])
        self.assertEqual(yaml.safe_load(self.hand_eye.read_text())['status'],'provisional')
        self.assertTrue(list(Path(p['artifact_folder']).glob('trial_authorization_*.json')))
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',payload)

    def setup_multi_trial(self):
        self.setup_trial()
        cassette=trimesh.creation.box(extents=[80,44,120]);cassette.apply_translation([10,22,100])
        cassette.export(self.root/'cassette.stl')
        manifest=self.app.manifest_path
        parts=json.loads(manifest.read_text())['parts']
        parts[0].update(name='HMI',bounds_local_mm=[[-50,-10,-75],[50,10,75]])
        parts.append({'key':'03_option_cassette','name':'Option cassette','source_stl':'cassette.stl',
                      'source_sha256':hashlib.sha256((self.root/'cassette.stl').read_bytes()).hexdigest(),
                      'bounds_local_mm':cassette.bounds.tolist()})
        manifest.write_text(json.dumps({'parts':parts}))
        return manifest,parts

    def setup_skill_demo(self, shift_mm=(30, 0, 0)):
        """Trial setup plus a synthetic HMI demo whose CAD sits shift_mm away in the camera."""
        self.setup_trial();robot=self.app.alignment.robot
        T=np.eye(4);T[:3,3]=np.add([100,200,700],shift_mm);T[1,:3]=[0,0,1];T[2,:3]=[0,-1,0]
        digest=hashlib.sha256((self.root/'part.stl').read_bytes()).hexdigest()
        common=dict(cad_sha256=digest,tcp_offset=robot.control.getTCPOffset(),robot_id=self.cfg['robot']['ip'])
        self.demo=make_demo(self.root,'hmi_demo','02_control_panel',T,[-.4,.3,.3,0,3.1,0],**common)
        make_demo(self.root,'cassette_demo','03_option_cassette',T,[-.4,.3,.3,0,3.1,0],image=b'c',**common)
        self.cfg['alignment']['skill_bindings_file']=str(write_bindings(self.root,{'hmi_panel':'hmi_demo','option_cassette':'cassette_demo'}))
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        robot.cfg=load_settings(self.app.alignment.settings_file)[0]
        return T

    def test_skill_demo_target_reproduces_demo_relative_pose_and_executes_once(self):
        demo_T=self.setup_skill_demo()
        status=self.request('/api/alignment/status')
        self.assertEqual({d['demo_id']:d['cad_key'] for d in status['config']['skill_demos']},
                         {'hmi_demo':'02_control_panel','cassette_demo':'03_option_cassette'})
        target={'mode':'skill_demo','demo_id':None}
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel','alignment_target':target})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        plan=state['plan'];self.assertEqual(plan['alignment_mode'],'skill_demo')
        self.assertEqual(plan['skill_demo']['demo_id'],'hmi_demo');self.assertEqual(plan['alignment_target'],target)
        self.assertTrue(plan['controller_path_checked'])
        now=np.asarray(plan['T_camera_from_input_cad_m']);demo=np.asarray(demo_T,dtype=float);demo[:3,3]*=.001
        tcp_camera=np.asarray(plan['T_tcp_from_camera'])
        expected=np.asarray(plan['T_base_from_tcp_capture'])@tcp_camera@now@np.linalg.inv(demo)@np.linalg.inv(tcp_camera)
        np.testing.assert_allclose(plan['T_base_from_tcp_target'],expected,atol=1e-9)
        self.assertAlmostEqual(plan['translation_m'],.03,places=9)
        robot=self.app.alignment.robot;c=robot.control;r=robot.receive
        # The execute request must name the same target; a face-mode client cannot reuse it.
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True})
        def dispatch(p,*args):c.calls.append(('moveL',p));r.pose=p;c.progress=-2;return True
        c.moveL=dispatch
        self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True,'alignment_target':target})
        self.assertEqual(self.wait_idle()['state'],'completed')
        moves=[p for n,p in c.calls if n=='moveL'];self.assertEqual(len(moves),1)
        np.testing.assert_allclose(moves[0],plan['tcp_target_m_rad'],atol=1e-9)

    def test_skill_demo_rejects_wrong_cad_and_demo_changed_before_motion(self):
        self.setup_skill_demo();robot=self.app.alignment.robot
        with self.assertRaises(HTTPError):
            self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel',
                                                'alignment_target':{'mode':'skill_demo','demo_id':'cassette_demo'}})
        with self.assertRaises(HTTPError):
            self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'},
                                                'alignment_target':{'mode':'skill_demo','demo_id':'missing'}})
        target={'mode':'skill_demo','demo_id':'hmi_demo'}
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel','alignment_target':target})
        plan=self.wait_idle()['plan'];self.assertEqual(plan['skill_demo']['demo_id'],'hmi_demo')
        (self.demo/'btn_rgb.png').write_bytes(b're-recorded')
        with self.assertRaises(HTTPError) as rejected:
            self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True,'alignment_target':target})
        self.assertIn(b'skill demo changed',rejected.exception.read())
        self.assertFalse([n for n,*_ in robot.control.calls if n=='moveL'])

    def align_to_skill_demo(self):
        """Complete a skill-demo alignment with fake hardware; return the executed plan."""
        self.setup_skill_demo()
        self.cfg['alignment']['skill_python']=sys.executable
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        robot=self.app.alignment.robot;robot.cfg=load_settings(self.app.alignment.settings_file)[0]
        c,r=robot.control,robot.receive
        def dispatch(p,*args):
            c.calls.append(('moveL',p));r.pose=p
            c.progress=-1 if c.progress==-2 else -2
            return True
        c.moveL=dispatch
        target={'mode':'skill_demo','demo_id':'hmi_demo'}
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel','alignment_target':target})
        plan=self.wait_idle()['plan']
        self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True,'alignment_target':target})
        state=self.wait_idle();self.assertEqual(state['state'],'completed',state)
        return plan,state

    def test_realign_before_skill_uses_fresh_capture_and_replaces_skill_readiness(self):
        first,state=self.align_to_skill_demo()
        camera_thread=self.app.camera.thread
        camera_sequence=self.app.camera.sequence
        self.assertTrue(self.app.camera.status()['running'])
        reference=self.app.alignment.robot.receive.pose.copy()
        target={'mode':'skill_demo','demo_id':'hmi_demo'}
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel',
                                            'alignment_target':target})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        self.assertIsNone(state['skill_ready'])
        second=state['plan']
        self.assertNotEqual(second['id'],first['id'])
        self.assertNotEqual(second['capture_id'],first['capture_id'])
        np.testing.assert_allclose(second['reference']['tcp_pose_m_rad'],reference)
        with self.assertRaises(HTTPError):
            self.request('/api/alignment/execute-skill',{'plan_id':first['id'],'confirmed':True})
        self.request('/api/alignment/execute',{'plan_id':second['id'],'confirmed':True,'alignment_target':target})
        state=self.wait_idle();self.assertEqual(state['state'],'completed',state)
        self.assertEqual(state['skill_ready']['plan_id'],second['id'])
        self.assertTrue(self.app.camera.status()['running'])
        self.assertIs(self.app.camera.thread,camera_thread)
        self.assertGreater(self.app.camera.sequence,camera_sequence)
        np.testing.assert_allclose(state['skill_ready']['aligned_tcp_m_rad'],second['tcp_target_m_rad'])
        with self.assertRaises(HTTPError):
            self.request('/api/alignment/execute-skill',{'plan_id':first['id'],'confirmed':True})
        calls=self.fake_runner()
        self.request('/api/alignment/execute-skill',{'plan_id':second['id'],'confirmed':True})
        state=self.wait_idle();self.assertEqual(state['state'],'completed',state)
        self.assertEqual(len(calls),1)

    def fake_runner(self,status='completed',block=None):
        calls=[]
        class Process:
            returncode=None
            def __init__(self,command):
                self.command=command;self.signals=[]
            def send_signal(self,sig):
                self.signals.append(sig)
                if block:block.set()
            def wait(self):
                if block:block.wait(5)
                result=Path(self.command[self.command.index('--result')+1])
                final='stopped' if self.signals else status
                result.write_text(json.dumps({'status':final,'demo_id':'hmi_demo','error':None if final=='completed' else final,
                                              'replay':{'completed_samples':3,'total_samples':3}}))
                self.returncode=0 if final=='completed' else 1
                return self.returncode
        def launch(command,log):
            process=Process(command);calls.append(process);return process
        self.app.alignment.skill_launcher=launch
        return calls

    def test_execute_skill_after_skill_demo_alignment_hands_robot_to_dinobot_once(self):
        plan,state=self.align_to_skill_demo();calls=self.fake_runner()
        ready=state['skill_ready'];self.assertEqual(ready['demo_id'],'hmi_demo');self.assertEqual(ready['plan_id'],plan['id'])
        self.assertNotIn('demo_files',ready)
        for request in ({'plan_id':plan['id']},{'plan_id':'other','confirmed':True}):
            with self.assertRaises(HTTPError):self.request('/api/alignment/execute-skill',request)
        self.request('/api/alignment/execute-skill',{'plan_id':plan['id'],'confirmed':True})
        state=self.wait_idle();self.assertEqual(state['state'],'completed',state)
        self.assertIn('Skill hmi_demo completed',state['message']);self.assertFalse(state['connected'])
        self.assertIsNone(state['skill_ready']);self.assertEqual(len(calls),1)
        command=calls[0].command
        self.assertEqual(command[0],os.path.abspath(sys.executable));self.assertTrue(command[1].endswith('skill_runner.py'))
        self.assertEqual(command[command.index('--demo')+1],'hmi_demo')
        self.assertEqual(Path(command[command.index('--demos-root')+1]),self.root/'demos')
        np.testing.assert_allclose(json.loads(command[command.index('--expected-start')+1]),plan['tcp_target_m_rad'])
        folder=Path(plan['artifact_folder']);self.assertTrue(list(folder.glob('skill_execution_request_*.json')))
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute-skill',{'plan_id':plan['id'],'confirmed':True})

    def test_execute_skill_requires_unmoved_tcp_and_unchanged_demo(self):
        plan,_=self.align_to_skill_demo();calls=self.fake_runner()
        robot=self.app.alignment.robot;aligned=list(robot.receive.pose)
        robot.receive.pose=[aligned[0]+.01,*aligned[1:]]
        with self.assertRaises(HTTPError) as moved:
            self.request('/api/alignment/execute-skill',{'plan_id':plan['id'],'confirmed':True})
        self.assertIn(b'away from the aligned pose',moved.exception.read())
        robot.receive.pose=aligned;(self.demo/'btn_rgb.png').write_bytes(b'changed')
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute-skill',{'plan_id':plan['id'],'confirmed':True})
        self.assertFalse(calls)

    def test_stop_during_skill_signals_the_runner(self):
        plan,_=self.align_to_skill_demo();block=threading.Event();calls=self.fake_runner(block=block)
        self.request('/api/alignment/execute-skill',{'plan_id':plan['id'],'confirmed':True})
        deadline=time.monotonic()+5
        while not calls and time.monotonic()<deadline:time.sleep(.01)
        self.assertEqual(self.request('/api/alignment/status')['state'],'skill')
        with self.assertRaises(HTTPError):self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        self.request('/api/alignment/stop',{})
        state=self.wait_idle();self.assertEqual(state['state'],'error');self.assertIn('stopped',state['message'])
        self.assertEqual(len(calls[0].signals),1)

    def test_face_alignment_does_not_offer_skill_execution(self):
        self.setup_trial();robot=self.app.alignment.robot;c,r=robot.control,robot.receive
        def dispatch(p,*args):c.calls.append(('moveL',p));r.pose=p;c.progress=-2;return True
        c.moveL=dispatch
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        plan=self.wait_idle()['plan']
        self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True})
        state=self.wait_idle();self.assertEqual(state['state'],'completed');self.assertIsNone(state['skill_ready'])
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute-skill',{'plan_id':plan['id'],'confirmed':True})

    def test_switch_selected_cads_in_one_connection_and_invalidate_on_selection_change(self):
        self.setup_multi_trial();robot=self.app.alignment.robot
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'part.stl'},'selection_signature':'hmi'})
        hmi=self.wait_idle()['plan'];self.assertEqual(hmi['cad_key'],'02_control_panel')
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'cassette.stl'},'selection_signature':'cassette'})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        cassette=state['plan'];self.assertIs(self.app.alignment.robot,robot)
        self.assertEqual(cassette['cad_key'],'03_option_cassette');self.assertNotEqual(hmi['id'],cassette['id'])
        np.testing.assert_allclose(cassette['plane_origin_cad_mm'],[10,44,100],atol=1e-5)
        goal=np.asarray(cassette['T_base_from_tcp_target']);cad=np.asarray(cassette['T_base_from_cad'])
        np.testing.assert_allclose(goal[:3,3],(cad@np.array([.01,.244,.10,1]))[:3],atol=1e-7)
        np.testing.assert_allclose(goal[:3,1],cad[:3,2],atol=1e-10)
        np.testing.assert_allclose(goal[:3,2],-cad[:3,1],atol=1e-10)
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':cassette['id'],'confirmed':True,'selection_signature':'hmi'})
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':hmi['id'],'confirmed':True,'selection_signature':'hmi'})
        self.request('/api/alignment/invalidate',{})
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':cassette['id'],'confirmed':True,'selection_signature':'cassette'})
        self.assertFalse(robot.control.calls)

    def test_automatic_identification_uses_winner_mesh_bounds_and_does_not_inherit_target(self):
        manifest,parts=self.setup_multi_trial();original=self.app._run;winner=['03_option_cassette']
        def identified(*args):
            original(*args)
            with self.app.lock:
                self.app.jobs[args[0]]['result']['cad_key']=winner[0]
        self.app._run=identified
        with patch('sam6d_ui.assembly_identification.assembly_catalog',return_value=(manifest,parts)):
            request={'live_camera':True,'auto_assembly':'printed_assembly','selection_signature':'auto'}
            self.request('/api/alignment/plan',request)
            state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
            cassette=state['plan'];self.assertEqual(cassette['cad_key'],'03_option_cassette')
            self.assertEqual(cassette['selection_mode'],'automatic')
            self.assertEqual(self.pose_calls[-1]['auto_assembly'],'printed_assembly')
            np.testing.assert_allclose(cassette['plane_origin_cad_mm'],[10,44,100],atol=1e-5)
            self.assertEqual(Path(cassette['cad_source_path']).name,'cassette.stl')
            winner[0]='02_control_panel';self.request('/api/alignment/plan',request)
            hmi=self.wait_idle()['plan'];self.assertEqual(hmi['cad_key'],'02_control_panel')
            np.testing.assert_allclose(hmi['plane_origin_cad_mm'],[0,30,128.75])
            with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':cassette['id'],'confirmed':True,'selection_signature':'auto'})
            winner[0]='not_in_catalog';self.request('/api/alignment/plan',request)
            state=self.wait_idle();self.assertEqual(state['state'],'error');self.assertIsNone(state['plan'])
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_cad_edit_after_estimation_blocks_execution(self):
        self.setup_multi_trial()
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'cassette.stl'}})
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        (self.root/'cassette.stl').write_bytes(b'changed CAD')
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':state['plan']['id'],'confirmed':True})
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_uploaded_cad_is_identified_by_bytes_and_unknown_cad_uses_its_own_bounds(self):
        self.setup_multi_trial()
        payload={'live_camera':True,'cad':{'name':'renamed.stl','data':base64.b64encode((self.root/'cassette.stl').read_bytes()).decode()}}
        self.request('/api/alignment/plan',payload)
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        self.assertEqual(state['plan']['cad_key'],'03_option_cassette')
        self.assertEqual(Path(state['plan']['cad_source_path']).parent,Path(state['plan']['artifact_folder']))
        mesh=trimesh.creation.box(extents=[60,18,90]);mesh.apply_translation([25,9,80])
        payload['cad']={'name':'custom.stl','data':base64.b64encode(mesh.export(file_type='stl')).decode()}
        self.request('/api/alignment/plan',payload)
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        self.assertEqual(state['plan']['cad_key'],'uploaded_cad')
        self.assertEqual(state['plan']['cad_name'],'custom.stl')
        np.testing.assert_allclose(state['plan']['plane_origin_cad_mm'],[25,18,80],atol=1e-5)
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_selection_invalidation_cancels_pending_alignment_inference(self):
        self.setup_multi_trial();entered=threading.Event();release=threading.Event();original=self.app._run
        self.addCleanup(release.set)
        def delayed(*args):
            entered.set();release.wait(3);original(*args)
        self.app._run=delayed
        self.request('/api/alignment/plan',{'live_camera':True,'server_files':{'cad':'cassette.stl'}})
        self.assertTrue(entered.wait(3))
        state=self.request('/api/alignment/invalidate',{})
        self.assertIsNone(state['plan'])
        release.set();state=self.wait_idle()
        self.assertIsNone(state['plan']);self.assertFalse(self.app.alignment.robot.control.calls)

    def test_automatic_ik_rejection_does_not_publish_a_target_or_move(self):
        self.setup_trial()
        self.app.alignment.robot.control.ik=False
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        state=self.wait_idle()
        self.assertEqual(state['state'],'error');self.assertIsNone(state['plan'])
        self.assertIn('inverse kinematics',state['message'])
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_joint_rejection_reports_the_joint_and_saves_candidate_and_ik_evidence(self):
        self.setup_trial()
        self.app.alignment.robot.control.branch=[0,0,0,0,0,self.cfg['motion']['max_joint_delta_rad']+.5]
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        state=self.wait_idle();self.assertEqual(state['state'],'error')
        self.assertIsNone(state['plan']);self.assertIn('wrist_3 (joint 6)',state['message'])
        files=list(self.app.jobs_root.glob('foundationpose/*/robot_alignment_error.json'))
        self.assertEqual(len(files),1)
        report=json.loads(files[0].read_text())
        self.assertEqual(report['controller_check_details']['joint_index'],5)
        self.assertEqual(report['controller_check_details']['rejection_type'],'adjacent_joint_step')
        self.assertFalse(report['motion_sent'])
        candidate=json.loads((files[0].parent/'robot_alignment_candidate.json').read_text())
        self.assertFalse(candidate['controller_path_checked'])
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_large_rotation_is_allowed_at_180_and_smaller_configured_limits_still_block(self):
        self.setup_trial()
        self.cfg['alignment']['orientation_mode']='cad_axes'
        self.cfg['motion']['max_rotation_deg']=90
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        self.app.alignment.robot.cfg=load_settings(self.app.alignment.settings_file)[0]
        request={'live_camera':True,'target_cad_key':'02_control_panel'}
        self.request('/api/alignment/plan',request)
        state=self.wait_idle();plan=state['plan']
        self.assertTrue(plan['rotation_limit_exceeded']);self.assertGreater(plan['rotation_deg'],90)
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True})
        with self.assertRaises(HTTPError):self.request('/api/alignment/allow-rotation',{
            'plan_id':plan['id'],'settings_sha256':plan['settings_sha256'],'confirmed':True,'physical_path_reviewed':True})
        self.cfg['motion']['max_rotation_deg']=180
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True})
        self.request('/api/alignment/disconnect',{})
        self.request('/api/alignment/connect',{});self.assertEqual(self.wait_idle()['state'],'ready')
        self.request('/api/alignment/plan',request)
        state=self.wait_idle();self.assertEqual(state['state'],'planned',state)
        plan=state['plan'];self.assertFalse(plan['rotation_limit_exceeded'])
        self.assertGreater(plan['rotation_deg'],90);self.assertEqual(plan['rotation_limit_deg'],180)
        self.assertTrue(plan['controller_path_checked']);self.assertFalse(self.app.alignment.robot.control.calls)
        c=self.app.alignment.robot.control;r=self.app.alignment.robot.receive
        def dispatch(target,*args):c.calls.append(('moveL',target));r.pose=target;c.progress=-2;return True
        c.moveL=dispatch
        self.request('/api/alignment/execute',{'plan_id':plan['id'],'confirmed':True})
        self.assertEqual(self.wait_idle()['state'],'completed')

    def test_large_rotation_with_failed_ik_is_visible_but_cannot_be_allowed(self):
        self.setup_trial()
        self.cfg['alignment']['orientation_mode']='cad_axes'
        self.cfg['motion']['max_rotation_deg']=90
        self.app.alignment.settings_file.write_text(json.dumps(self.cfg))
        self.app.alignment.robot.cfg=load_settings(self.app.alignment.settings_file)[0]
        self.app.alignment.robot.control.ik=False
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        state=self.wait_idle();p=state['plan'];self.assertIsNotNone(p,state)
        self.assertFalse(p['controller_path_checked']);self.assertIn('inverse kinematics',p['controller_check_error'])
        with self.assertRaises(HTTPError):self.request('/api/alignment/allow-rotation',{'plan_id':p['id'],
            'settings_sha256':p['settings_sha256'],'confirmed':True,'physical_path_reviewed':True})
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_speed_change_is_persisted_invalidates_plan_and_preserves_other_settings(self):
        self.setup_trial()
        self.request('/api/alignment/plan',{'live_camera':True,'target_cad_key':'02_control_panel'})
        state=self.wait_idle();old_plan=state['plan']
        original=json.loads(self.app.alignment.settings_file.read_text())
        payload={'speed_m_s':.012,'settings_sha256':state['config']['settings_sha256']}
        updated=self.request('/api/alignment/speed',payload)
        self.assertIsNone(updated['plan']);self.assertFalse(updated['needs_reconnect'])
        self.assertEqual(updated['config']['speed_m_s'],.012)
        self.assertEqual(self.app.alignment.robot.cfg['motion']['speed_m_s'],.012)
        saved=json.loads(self.app.alignment.settings_file.read_text())
        original['motion']['speed_m_s']=.012
        self.assertEqual(saved,original)
        with self.assertRaises(HTTPError):self.request('/api/alignment/execute',{'plan_id':old_plan['id'],'confirmed':True})
        with self.assertRaises(HTTPError):self.request('/api/alignment/speed',payload)
        for speed in (0,-1,.251,True,'0.01',None):
            with self.assertRaises(HTTPError):self.request('/api/alignment/speed',{'speed_m_s':speed,'settings_sha256':updated['config']['settings_sha256']})
        self.app.alignment.robot.receive.speed[0]=.01
        with self.assertRaises(HTTPError):self.request('/api/alignment/speed',{'speed_m_s':.02,'settings_sha256':updated['config']['settings_sha256']})
        self.app.alignment.robot.receive.speed[0]=0
        self.app.alignment._update('moving','simulated movement')
        with self.assertRaises(HTTPError):self.request('/api/alignment/speed',{'speed_m_s':.02,'settings_sha256':updated['config']['settings_sha256']})
        self.app.alignment._update('ready','test complete')
        self.assertFalse(self.app.alignment.robot.control.calls)

    def test_trial_tcp_must_equal_calibration_tcp_and_mesh_hash_is_enforced(self):
        self.setup_trial()
        request={'live_camera':True,'target_cad_key':'02_control_panel'}
        self.app.alignment.robot.control.tcp=[0,0,.16,0,0,0]
        self.request('/api/alignment/plan',request)
        state=self.wait_idle();self.assertEqual(state['state'],'error')
        self.assertIn('TCP',state['message']);self.assertFalse(self.pose_calls)
        (self.root/'part.stl').write_bytes(b'changed')
        with self.assertRaises(HTTPError):self.request('/api/alignment/plan',request)
        self.assertFalse(self.app.alignment.robot.control.calls)


def visual_matrix():
    T=np.eye(4);T[:3,3]=[.1,.2,.7]
    return T


if __name__=='__main__':unittest.main()
