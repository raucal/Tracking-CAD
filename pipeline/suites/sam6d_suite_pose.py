"""Optional PEM enrichment of saved ISM detections, with explicit CAD frame audits."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))
from settings import PIPELINE, PREPARED, run_dataset
import hashlib
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image


from file_utils import write_json


def audit_cad(part,base):
    import trimesh
    path=base/part['sam_ply'];mesh=trimesh.load_mesh(path,process=False);bounds=np.asarray(mesh.bounds)
    local_from_model=np.asarray(part.get('T_local_from_sam_mm',np.eye(4)),dtype=float)
    if local_from_model.shape!=(4,4) or not np.isfinite(local_from_model).all() or not np.allclose(local_from_model[3],[0,0,0,1]) or not np.allclose(local_from_model[:3,:3].T@local_from_model[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(local_from_model[:3,:3]),1,atol=1e-5):raise ValueError('Invalid or non-rigid CAD transformation')
    placed=mesh.copy();placed.apply_transform(local_from_model)
    if 'bounds_local_mm' in part and not np.allclose(placed.bounds,part['bounds_local_mm'],atol=.05):raise ValueError('CAD origin/scale does not match the manifest')
    templates=base/part['sam_template_dir'];metadata=templates/'render_metadata.json'
    if metadata.is_file() and json.loads(metadata.read_text()).get('xyz_unit')!='mm':raise ValueError('PEM templates are not in mm')
    low=np.full(3,np.inf);high=np.full(3,-np.inf);tolerance=max(.5,float(mesh.extents.max())*.01)
    for index in range(42):
        xyz=np.load(templates/f'xyz_{index}.npy',allow_pickle=False)
        with Image.open(templates/f'mask_{index}.png') as source:mask=np.asarray(source)>0
        if xyz.shape!=(*mask.shape,3) or not np.isfinite(xyz).all() or not mask.any():raise ValueError('Invalid template XYZ/mask')
        valid=xyz[mask]
        if np.any(valid<bounds[0]-tolerance) or np.any(valid>bounds[1]+tolerance):raise ValueError('XYZ coordinates do not match the metric CAD bounds')
        low=np.minimum(low,valid.min(axis=0));high=np.maximum(high,valid.max(axis=0))
    ratio=float((high-low).max()/mesh.extents.max())
    if not .5<=ratio<=1.05:raise ValueError('Incompatible XYZ scale: render normalization may not have been undone')
    return {'status':'passed','model_unit':'mm','template_xyz_unit':'mm','views_checked':42,
            'cad_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_bounds_mm':bounds.tolist(),
            'input_cad_bounds_mm':placed.bounds.tolist(),'T_input_cad_from_model_mm':local_from_model.tolist(),
            'T_model_from_input_cad_mm':np.linalg.inv(local_from_model).tolist(),'xyz_extent_ratio':ratio,
            'camera_convention':'OpenCV: X right, Y down, Z forward'}


def run_case_pose(folder,frame,parts,base,confidence,timeout=300,runner=subprocess.run):
    folder=Path(folder);started=time.perf_counter();poses=[];overlays=[];errors=[];audits={}
    detections=json.loads((folder/'detection_ism.json').read_text());accepted=[(i,d) for i,d in enumerate(detections) if d['score']>=confidence]
    for part in parts:
        selected=[dict(d,detection_index=i) for i,d in accepted if d['part_key']==part['key']]
        if not selected:continue
        out=folder/'pose'/part['key'];out.mkdir(parents=True,exist_ok=True)
        try:
            audit=audit_cad(part,base);audits[part['key']]=audit;write_json(out/'cad_frame_audit.json',audit)
            write_json(out/'detections.json',selected);write_json(out/'model_from_input.json',audit['T_model_from_input_cad_mm'])
            command=[sys.executable,str(PIPELINE/'workers/pose_worker.py'),'--rgb',str(folder/'rgb.png'),
                     '--depth',str(folder/'depth_raw.png'),'--camera',str(folder/'camera.json'),
                     '--cad',str((base/part['sam_ply']).resolve()),'--templates',str((base/part['sam_template_dir']).resolve()),
                     '--detections',str(out/'detections.json'),'--output',str(out/'pose.json'),
                     '--model-from-input',str(out/'model_from_input.json'),'--min-score',str(max(0,confidence-1e-7))]
            with (out/'pose.log').open('w') as log:runner(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
            result=json.loads((out/'pose.json').read_text())
            for pose in result.get('poses',[]):poses.append(dict(pose,part_key=part['key'],part_name=part['name']))
            if result.get('poses'):
                overlays.append({'part_key':part['key'],'path':str((out/'pose_overlay.png').relative_to(folder)),'axes_frame':'input_cad'})
            else:errors.append({'part_key':part['key'],'error':result.get('status','no pose')})
        except Exception as error:errors.append({'part_key':part['key'],'error':str(error),'log':str((out/'pose.log').relative_to(folder))})
    status='no_accepted_detection' if not accepted else 'partial' if poses and errors else 'estimated' if poses else 'error'
    result={'status':status,'backend':'pem','poses':poses,'best_index':int(np.argmax([p['pose_score'] for p in poses])) if poses else None,
            'overlays':overlays,'errors':errors,'cad_audits':audits,'elapsed_s':time.perf_counter()-started,'translation_unit':'mm',
            'axes_frame':'input_cad','pose_score_is_probability':False}
    write_json(folder/'pose.json',result);return result


def enrich_suite(source,overwrite=False,frame_names=None):
    source=Path(source).resolve();run=json.loads((source/'run.json').read_text())
    import torch
    if not torch.cuda.is_available():raise RuntimeError('PEM pose requires CUDA')
    if frame_names and set(frame_names)-{f['name'] for f in run['frames']}:raise ValueError('The capture does not belong to the saved suite')
    previous=source/'pose_run.json'
    if previous.exists() and not overwrite and json.loads(previous.read_text()).get('source_run_sha256')!=hashlib.sha256((source/'run.json').read_bytes()).hexdigest():raise ValueError('Saved poses belong to another run; use --overwrite-pose')
    manifest,base=run_dataset(run,source)
    data=json.loads(manifest.read_text());parts=data['parts']
    mapping={p['key']:p['category_id'] for p in parts}
    if any(mapping.get(p['key'])!=p['category_id'] for p in run['catalog']):raise ValueError('The available manifest does not preserve the suite CAD mapping')
    frames={f['name']:f for f in data['frames']};sidecar={'backend':'pem','source_run_sha256':hashlib.sha256((source/'run.json').read_bytes()).hexdigest(),'manifest_used':str(manifest),'manifest_used_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'frames':[]}
    for case in run['frames']:
        if frame_names and case['name'] not in frame_names:continue
        print('PEM: '+case['name'],flush=True)
        frame=frames[case['name']];folder=source/'cases'/case['name'];existing=folder/'pose.json'
        if existing.exists() and not overwrite:result=json.loads(existing.read_text())
        else:
            if existing.exists():shutil.copy2(existing,existing.with_name('pose.before.json'))
            shutil.copy2(base/frame['camera'],folder/'camera.json')
            result=run_case_pose(folder,frame,parts,base,run['settings']['confidence'])
        sidecar['frames'].append({'name':case['name'],'status':result['status'],'poses':len(result['poses']),'elapsed_s':result.get('elapsed_s')})
        write_json(source/'pose_run.json',sidecar)
    sidecar['status']='complete';sidecar['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(source/'pose_run.json',sidecar);return sidecar
