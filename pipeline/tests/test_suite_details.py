from pathlib import Path
import csv,json,sys,tempfile,unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from suites.sam6d_suite_report import score_components,export_report,pose_gallery,publish_dashboard
from suites.sam6d_suite_pose import run_case_pose,audit_cad
from unittest.mock import patch

class SuiteDetails(unittest.TestCase):
 def test_ism_formula_keeps_visibility_weight_and_missing_values(self):
  d={'score':.74,'semantic_score':.6,'appearance_score':.8,'geometry_score':.9,'visible_ratio':.5}
  c=score_components(d);self.assertAlmostEqual(c['score_reconstructed'],.74);self.assertTrue(c['ism_formula_matches']);self.assertEqual(c['geometry_contribution'],.45)
  d.update(score=.7,visible_ratio=0);self.assertAlmostEqual(score_components(d)['score_reconstructed'],.7)
  self.assertIsNone(score_components({'score':.8})['geometry_score']);self.assertIsNone(score_components({'score':.8})['score_reconstructed'])
 def test_pose_dispatch_preserves_detection_identity_and_cad_transform(self):
  with tempfile.TemporaryDirectory() as temp:
   folder=Path(temp);(folder/'detection_ism.json').write_text(json.dumps([{'part_key':'a','score':.5},{'part_key':'b','score':.2}]))
   audit={'T_model_from_input_cad_mm':np.eye(4).tolist()};seen=[]
   def runner(command,**kwargs):
    dets=json.loads(Path(command[command.index('--detections')+1]).read_text());seen.extend(dets)
    output=Path(command[command.index('--output')+1]);output.write_text(json.dumps({'poses':[{'pose_score':.9,'detection_index':0,'T_camera_from_input_cad_mm':np.eye(4).tolist()}]}));Image.new('RGB',(10,10)).save(output.parent/'pose_overlay.png')
   with patch('suites.sam6d_suite_pose.audit_cad',return_value=audit):
    result=run_case_pose(folder,{},[{'key':'a','name':'A','sam_ply':'a.ply','sam_template_dir':'templates'},{'key':'b','name':'B'}],folder,.5,runner=runner)
   self.assertEqual(len(seen),1);self.assertEqual(seen[0]['detection_index'],0);self.assertEqual(result['poses'][0]['part_key'],'a');self.assertEqual(result['status'],'estimated')
 def test_metric_audit_rejects_normalized_xyz_and_checks_original_origin(self):
  import trimesh
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);trimesh.creation.box(extents=[20,40,60]).export(base/'model.ply');folder=base/'templates';folder.mkdir()
   xyz=np.array([[[-10,-20,-30],[10,-20,-30],[-10,20,-30],[10,20,-30]],[[-10,-20,30],[10,-20,30],[-10,20,30],[10,20,30]]],dtype=np.float32)
   for i in range(42):np.save(folder/f'xyz_{i}.npy',xyz);Image.fromarray(np.full((2,4),255,np.uint8)).save(folder/f'mask_{i}.png')
   T=np.eye(4);T[:3,3]=[100,200,300]
   part={'sam_ply':'model.ply','sam_template_dir':'templates','T_local_from_sam_mm':T.tolist(),'bounds_local_mm':[[90,180,270],[110,220,330]]}
   audit=audit_cad(part,base);self.assertEqual(audit['views_checked'],42);self.assertEqual(audit['T_model_from_input_cad_mm'][0][3],-100)
   for i in range(42):np.save(folder/f'xyz_{i}.npy',xyz*.001)
   with self.assertRaisesRegex(ValueError,'Escala XYZ'):audit_cad(part,base)
 def test_no_accepted_detection_does_not_run_pem(self):
  with tempfile.TemporaryDirectory() as temp:
   folder=Path(temp);(folder/'detection_ism.json').write_text('[]')
   def runner(*args,**kwargs):raise AssertionError('PEM should not run without accepted masks')
   result=run_case_pose(folder,{},[],folder,.5,runner=runner)
   self.assertEqual(result['status'],'no_accepted_detection');self.assertEqual(result['poses'],[])
 def test_report_shows_components_and_original_cad_pose_without_changing_scores(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);source=base/'run';case=source/'cases/capture';case.mkdir(parents=True)
   part={'key':'cad_a','name':'A','category_id':1,'sam_template_dir':'no_templates'}
   manifest={'parts':[part],'frames':[{'name':'capture','camera':'camera.json','source_rgb':'original.jpg'}]};(base/'manifest.json').write_text(json.dumps(manifest));(base/'camera.json').write_text('{}')
   detection={'part_key':'cad_a','part_name':'A','category_id':1,'score':.74,'semantic_score':.6,'appearance_score':.8,'geometry_score':.9,'visible_ratio':.5}
   timing={'processing_s':1,'sam_s':.5,'dino_s':.2,'semantic_s':.1}
   frame={'name':'capture','frame_id':0,'device':'cuda','status':'ok','recognized':True,'candidate':detection,'timing':timing,'visible_components':['cad_a'],'proposal_counts':{'valid_depth_and_size':1},'proposal_semantic_scores':[[.6]],'comparisons':[{'part_key':'cad_a','part_name':'A','category_id':1,'semantic_score':.6,'best_template':0,'compared_proposals':1,'semantic_comparison_s':.1}],'comparison_region_proposal_id':0,'comparison_scope':'primary_detection'}
   run={'catalog':[part],'manifest':str(base/'manifest.json'),'frames':[frame],'settings':{'confidence':.5,'min_score':.3,'stability_score':.97,'stability_score_yaml':.85,'stability_score_effective':.97},'device':{'used':'cuda','torch':'test'},'status':'complete','selected_frames':['capture'],'setup_timing':{}}
   (source/'run.json').write_text(json.dumps(run));(case/'case.json').write_text(json.dumps(frame));(case/'detection_ism.json').write_text(json.dumps([detection]))
   for name in ['rgb.png','overlay.png']:Image.new('RGB',(16,12)).save(case/name)
   Image.fromarray(np.ones((12,16),np.uint8)*255).save(case/'mask_000.png')
   T=np.eye(4);T[:3,3]=[100,200,500]
   pose={'status':'estimated','backend':'pem','poses':[{'part_key':'cad_a','pose_score':.99,'segmentation_score':.74,'detection_index':0,'T_camera_from_input_cad_mm':T.tolist(),'T_camera_from_model_mm':T.tolist()}]}
   (case/'pose.json').write_text(json.dumps(pose))
   target=base/'report';export_report(source,target);page=(target/'index.html').read_text()
   for text in ['Semántica','Apariencia','Geométrica','0.740','0.600','0.800','0.900','0.97','100.0','T_camara_desde_CAD_mm']:self.assertIn(text,page)
   row=next(csv.DictReader((target/'confianza_sam6d_por_imagen.csv').read_text(encoding='utf-8-sig').splitlines()));self.assertEqual(float(row['confianza_final_sam6d']),.74);self.assertEqual(row['ism_formula_matches'],'True')
   row=next(csv.DictReader((target/'poses_pem.csv').read_text(encoding='utf-8-sig').splitlines()));self.assertEqual(float(row['score_ism']),.74);self.assertEqual(float(row['score_pem']),.99);self.assertEqual(float(row['tz_mm']),500)

 def test_dashboard_preserves_old_report_and_switches_to_latest(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);dashboard=base/'dashboard';dashboard.mkdir();(dashboard/'.sam6d_suite_export').touch();(dashboard/'index.html').write_text('old')
   first=base/'report1';first.mkdir();(first/'index.html').write_text('first')
   publish_dashboard(first,dashboard)
   self.assertTrue(dashboard.is_symlink());self.assertFalse(Path(__import__('os').readlink(dashboard)).is_absolute())
   self.assertEqual((dashboard/'index.html').read_text(),'first')
   legacy=list(base.glob('dashboard_legacy_*'));self.assertEqual(len(legacy),1);self.assertEqual((legacy[0]/'index.html').read_text(),'old')
   second=base/'report2';second.mkdir();(second/'index.html').write_text('second')
   publish_dashboard(second,dashboard);self.assertEqual((dashboard/'index.html').read_text(),'second');self.assertEqual(len(list(base.glob('dashboard_legacy_*'))),1)

 def test_pose_gallery_links_to_saved_cad_axes_and_matrices(self):
  frames=[{'name':'hmi panel','frame_id':7},{'name':'missing','frame_id':8}]
  poses={'hmi panel':{'poses':[{}],'overlays':[{'part_key':'cad','path':'pose/cad/pose_overlay.png'}]},'missing':{'status':'no_accepted_detection','poses':[]}}
  page=pose_gallery(frames,poses)
  for value in ['id="poses"','href="#frame7-pose"','cases/hmi%20panel/pose/cad/pose_overlay.png','Ver matrices R, t y T','Sin detección aceptada']:self.assertIn(value,page)

 def test_dashboard_does_not_replace_unrelated_user_files(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);dashboard=base/'dashboard';dashboard.mkdir();(dashboard/'mine.txt').write_text('keep')
   report=base/'report';report.mkdir();(report/'index.html').write_text('new')
   with self.assertRaisesRegex(ValueError,'ajenos'):publish_dashboard(report,dashboard)
   self.assertEqual((dashboard/'mine.txt').read_text(),'keep')

if __name__=='__main__':unittest.main()
