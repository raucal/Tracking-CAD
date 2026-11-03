"""Projection, RGB axes, exact CAD bounds and original-frame transforms."""
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
p=Path(__file__).resolve().parents[3]/'CADTracking/models/SAM-6D/SAM-6D/Pose_Estimation_Model/utils/draw_utils.py'
spec=importlib.util.spec_from_file_location('tested_pem_draw',p);draw=importlib.util.module_from_spec(spec);spec.loader.exec_module(draw)

class PemAxes(unittest.TestCase):
 def test_rgb_axes_project_with_metric_length(self):
  image=np.zeros((240,320,3),np.uint8);K=np.array([[200,0,160],[0,200,120],[0,0,1]])
  # A tilted model keeps the three axes visible and separated.
  a=np.pi/4;R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
  result=draw.draw_pose_axes(image,R,np.array([0,0,500]),K,100)
  for color in [(255,0,0),(0,255,0),(0,0,255)]:self.assertTrue(np.any(np.all(result==color,axis=2)),color)
 def test_original_cad_origin_uses_model_from_input_transform(self):
  K=np.array([[100,0,100],[0,100,100],[0,0,1]]);transform=np.eye(4);transform[0,3]=-50
  calls=[]
  with patch.object(draw.cv2,'arrowedLine',side_effect=lambda image,start,end,*a,**k:calls.append((start,end)) or image):
   draw.draw_pose_axes(np.zeros((240,240,3),np.uint8),np.eye(3),np.array([0,0,500]),K,100,transform)
  self.assertEqual(calls[0][0],(90,100));self.assertEqual(calls[0][1],(110,100))
 def test_behind_camera_axes_are_omitted(self):
  image=np.zeros((20,20,3),np.uint8)
  draw.draw_pose_axes(image,np.eye(3),np.array([0,0,-500]),np.eye(3),100)
  self.assertFalse(image.any())
 def test_box_uses_exact_bounds_not_sample_mean(self):
  image=np.zeros((240,320,3),np.uint8);K=np.array([[100,0,160],[0,100,120],[0,0,1]])
  points=np.array([[0,0,0],[0,0,0],[0,0,0],[20,40,0]],dtype=float);bounds=np.array([[-20,-40,-10],[20,40,10]])
  captured=[]
  with patch.object(draw,'draw_3d_bbox',side_effect=lambda img,p,*a:captured.append(p) or img):
   draw.draw_detections(image,np.eye(3)[None],np.array([[0,0,500]]),points,K[None],model_bounds=bounds,show_axes=False)
  # Opposite projected X corners are symmetric about the calibrated principal point.
  self.assertEqual(int(captured[0][:,0].min()),155);self.assertEqual(int(captured[0][:,0].max()),164)

if __name__=='__main__':unittest.main()
