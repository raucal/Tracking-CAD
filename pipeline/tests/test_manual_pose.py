from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam6d_ui.backends.manual_pose import validate_box,run

class ManualPoseTests(unittest.TestCase):
 def test_box_validation_rejects_invalid_or_outside_coordinates(self):
  self.assertEqual(validate_box([10,20,30,40],(640,480)),[10,20,30,40])
  for box in (None,[1,2],[True,0,10,10],[0,0,3,10],[-1,0,10,10],[639,0,10,10],[0,0,10.5,10],[0,0,float('nan'),10]):
   with self.assertRaises(ValueError):validate_box(box,(640,480))
 def test_manual_path_builds_only_selected_rectangle_and_calls_pose_directly(self):
  with tempfile.TemporaryDirectory() as temp:
   folder=Path(temp);Image.new('RGB',(100,80)).save(folder/'input_rgb.png')
   inputs={'manual_bbox':[10,20,30,40],'input_center_mm':[1,2,3],'cad_name':'part.stl','options':{'pose_backend':'foundationpose','mode':'rgbd','pose':True}}
   pose={'poses':[{'T_camera_from_input_cad_mm':np.eye(4).tolist()}]}
   with patch('sam6d_ui.backends.manual_pose.foundationpose_run',return_value=pose) as estimate:
    result=run(folder,inputs,lambda message:None)
   mask=np.asarray(Image.open(folder/'mask_000.png'))
   self.assertEqual(np.count_nonzero(mask),1200)
   self.assertTrue(np.all(mask[20:60,10:40]==255))
   self.assertEqual(estimate.call_args.args[1],folder/'object_mm.obj')
   np.testing.assert_equal(estimate.call_args.args[3][:3,3],[-1,-2,-3])
   self.assertIsNone(result['best_score']);self.assertFalse(result['recognized'])
   self.assertEqual(result['renderer'],'not_used');self.assertTrue(result['pose_6d_estimated'])
   self.assertEqual(json.loads((folder/'manual_selection.json').read_text())['bbox'],[10,20,30,40])

if __name__=='__main__':unittest.main()
