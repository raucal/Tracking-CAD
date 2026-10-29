"""Run official PEM in a separate process; return model-to-camera rotation and mm translation."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
PEM=ROOT/'SAM6D/SAM-6D/SAM-6D/Pose_Estimation_Model'
sys.path[:0]=[str(ROOT/'SAM6D/pipeline/native'),str(PEM),str(PEM/'provider'),str(PEM/'utils'),str(PEM/'model'),str(PEM/'model/pointnet2')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rgb','depth','camera','cad','templates','detections','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--min-score',type=float,default=0.5)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('PEM necesita CUDA en esta implementación.')
    random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.set_num_threads(8)
    start=time.perf_counter();os.chdir(PEM)
    # All trained backbone weights are loaded from the full PEM checkpoint below.
    cfg=OmegaConf.load(PEM/'config/base.yaml');cfg.model.feature_extraction.pretrained=False
    from pose_estimation_model import Net
    from run_inference_custom import get_templates,get_test_data
    from draw_utils import draw_detections
    model=Net(cfg.model).cuda().eval()
    checkpoint=torch.load(PEM/'checkpoints/sam-6d-pem-base.pth',map_location='cpu',weights_only=False)
    state=checkpoint.get('model',checkpoint.get('state_dict',checkpoint))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    torch.cuda.synchronize();loaded=time.perf_counter()
    detections=json.loads(a.detections.read_text())
    if not any(d['score']>a.min_score for d in detections):
        a.output.write_text(json.dumps({'status':'no_accepted_detection','poses':[],'timing':{'total_s':time.perf_counter()-start}}));return
    with torch.inference_mode():
        rgb,pts,choose=get_templates(str(a.templates),cfg.test_dataset)
        dense_pts,dense_feats=model.feature_extraction.get_obj_feats(rgb,pts,choose)
        torch.cuda.synchronize();references=time.perf_counter()
        try:
            data,image,_,model_points,selected=get_test_data(str(a.rgb),str(a.depth),str(a.camera),str(a.cad),str(a.detections),a.min_score,cfg.test_dataset)
        except RuntimeError as e:
            if 'non-empty TensorList' not in str(e):raise
            a.output.write_text(json.dumps({'status':'insufficient_valid_depth','poses':[],'timing':{'total_s':time.perf_counter()-start}}));return
        # Sequential instances bound the memory footprint regardless of the number of masks.
        poses=[]
        for i,det in enumerate(selected):
            sample={k:v[i:i+1] for k,v in data.items()}
            sample.update(dense_po=dense_pts,dense_fo=dense_feats)
            out=model(sample)
            R=out['pred_R'][0].cpu().numpy();t=out['pred_t'][0].cpu().numpy()*1000
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
            if not np.isfinite(T).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(R),1,atol=1e-3):
                raise ValueError('PEM devolvió una transformación no válida.')
            pose_score=float((out.get('pred_pose_score',out['score']*0+1)*out['score'])[0].cpu())
            poses.append({'segmentation_score':det['score'],'pose_score':pose_score,'R':R.tolist(),'t_mm':t.tolist(),'T_camera_from_model_mm':T.tolist(),'translation_unit':'mm','frame':'OpenCV camera: X right, Y down, Z forward'})
        torch.cuda.synchronize();finished=time.perf_counter()
        best=int(np.argmax([v['pose_score'] for v in poses]));one=poses[best]
        camera=np.array(json.loads(a.camera.read_text())['cam_K']).reshape(1,3,3)
        preview=a.output.with_name('pose_overlay.png')
        projected = draw_detections(image.copy(), np.array([one['R']]), np.array([one['t_mm']]), model_points*1000, camera, color=(30,255,140))
        canvas = Image.new('RGB', (image.shape[1]*2,image.shape[0]))
        canvas.paste(Image.fromarray(image),(0,0));canvas.paste(Image.fromarray(projected.astype(np.uint8)),(image.shape[1],0));canvas.save(preview)
    result={'status':'estimated','poses':poses,'best_index':best,'checkpoint':'sam-6d-pem-base.pth','device':'cuda','timing':{'model_load_s':loaded-start,'reference_features_s':references-loaded,'pose_inference_s':finished-references,'total_s':time.perf_counter()-start},'accuracy_validated':False}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'poses':len(poses),'timing':result['timing']}),flush=True)
if __name__=='__main__':main()
