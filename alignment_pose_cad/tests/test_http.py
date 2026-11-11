"""HTTP and RGB-D integration with real storage and fake hardware/inference."""
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import pytest
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
from test_alignment import config,FakeReceive,FakeControl


class Source:
    def __init__(self,options):self.number=0
    def read(self):
        time.sleep(.02);self.number+=1
        return {'rgb':np.full((480,640,3),120,dtype='uint8'),'depth':np.full((480,640),700,dtype='uint16'),
                'camera':{'cam_K':[500,0,320,0,500,240,0,0,1],'depth_scale':1,'serial':'123'},
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

    def test_status_assets_and_same_origin_guard(self):
        s=self.request('/api/alignment/status');self.assertFalse(s['connected'])
        self.assertIn(b'Capture and calculate target',self.request('/alignment_pose_cad/panel.js'))
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


if __name__=='__main__':unittest.main()
