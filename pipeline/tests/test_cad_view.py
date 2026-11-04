"""CAD reference coordinates, black rendering and latest-frame delivery without CUDA."""
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam6d_ui.backends.cad_view import draw_reference_axes, render_cad_view
from sam6d_ui.server import handler


class CadViewTests(unittest.TestCase):
    def test_axes_are_drawn_in_rgb_and_invalid_origin_is_hidden(self):
        image=np.zeros((100,100,3),np.uint8)
        pose=np.eye(4);pose[2,3]=1
        K=np.array([[100,0,50],[0,100,50],[0,0,1]])
        draw_reference_axes(image,pose,K,.2)
        self.assertGreater(int(image[50,60,0]),int(image[50,60,1]))
        self.assertGreater(int(image[60,50,1]),int(image[60,50,0]))
        pose[2,3]=-1
        empty=np.zeros_like(image)
        draw_reference_axes(empty,pose,K,.2)
        self.assertFalse(empty.any())

    def test_renderer_uses_pose_and_scaled_intrinsics_without_camera_background(self):
        pose=np.eye(4);pose[2,3]=1
        K=np.array([[800,0,640],[0,800,360],[0,0,1]],float)
        render=Mock(return_value=(torch.zeros(1,360,640,3),None,None))
        with patch.dict(sys.modules,{'Utils':SimpleNamespace(nvdiffrast_render=render)}), patch(
                'sam6d_ui.backends.cad_view.draw_reference_axes', wraps=draw_reference_axes) as axes:
            jpeg=render_cad_view(SimpleNamespace(extents=np.array([.1,.2,.3])),pose,K,(720,1280,3),
                                 {'pos':torch.zeros(1,3)},object())
        np.testing.assert_allclose(axes.call_args.args[1],pose)
        self.assertEqual(axes.call_args.args[3],.1)
        image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(image.shape,(360,640,3))
        np.testing.assert_array_equal(image[:50,:50],0)
        self.assertTrue(image.any())
        np.testing.assert_allclose(render.call_args.kwargs['K'],[[400,0,320],[0,400,180],[0,0,1]])
        np.testing.assert_allclose(render.call_args.kwargs['ob_in_cams'][0].numpy(),pose)

    def test_cad_endpoint_delivers_its_own_frame_and_no_old_pose(self):
        tracker=Mock()
        app=SimpleNamespace(tracker=tracker)
        cls=handler(app);request=cls.__new__(cls)
        request.path='/api/cad/frame.jpg?after=-1'
        request.connection=Mock();request.headers={};request.request_version='HTTP/1.1'
        request.wfile=BytesIO()
        request.send_response=Mock();request.send_header=Mock();request.end_headers=Mock()
        tracker.latest_preview.return_value={'sequence':4,'age_s':.02,'jpeg':b'cad JPEG'}
        request.do_GET()
        tracker.latest_preview.assert_called_once_with('cad')
        request.send_response.assert_called_with(200)
        self.assertEqual(request.wfile.getvalue(),b'cad JPEG')
        request.path='/api/cad/frame.jpg?after=4'
        request.wfile=BytesIO();request.do_GET()
        request.send_response.assert_called_with(204)
        self.assertEqual(request.wfile.getvalue(),b'')
        tracker.latest_preview.return_value=None
        request.do_GET();request.send_response.assert_called_with(204)


if __name__ == '__main__':
    unittest.main()
