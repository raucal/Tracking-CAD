"""Persistent FoundationPose tracking initialized from a validated SAM6D job."""
import argparse
import json
import logging
import shutil
from pathlib import Path
import sys
import time

from foundationpose_worker import FOUNDATIONPOSE, NATIVE, valid_pose


def emit(value):
    print('@LIVE@'+json.dumps(value,allow_nan=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import trimesh
    sys.path[:0]=[str(NATIVE),str(FOUNDATIONPOSE)]
    import mycpp
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_seed
    logging.getLogger().setLevel(logging.ERROR)
    if not torch.cuda.is_available():raise RuntimeError('El seguimiento requiere CUDA.')
    torch.set_num_threads(4);set_seed(0)
    mesh=trimesh.load(args.seed/'object_mm.obj',force='mesh');mesh.apply_scale(.001)
    inputs=json.loads((args.seed/'inputs.json').read_text())
    result=json.loads((args.seed/'pose.json').read_text())
    best=result['poses'][result.get('best_index',0)]
    # The uploaded mesh was centered in mm. Preserve the original CAD frame.
    centered_from_input=np.eye(4);centered_from_input[:3,3]=-np.array(inputs['input_center_mm'])
    camera_from_input=np.asarray(best['T_camera_from_input_cad_mm'],dtype=float)
    camera_from_model=camera_from_input@np.linalg.inv(centered_from_input)
    camera_from_model[:3,3]*=.001
    valid_pose(camera_from_model)
    estimator=FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,mesh=mesh,
                            scorer=ScorePredictor(),refiner=PoseRefinePredictor(),
                            glctx=dr.RasterizeCudaContext(),debug=0,debug_dir=str(args.scratch/'debug'))
    # register() returns pose_last @ get_tf_to_centered_mesh(). Invert that
    # documented internal conversion to reuse the already estimated pose.
    tf=estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
    estimator.pose_last=torch.as_tensor(camera_from_model@np.linalg.inv(tf),device='cuda',dtype=torch.float32)
    if estimator.mesh_path:
        old_path=Path(estimator.mesh_path);new_path=args.scratch/'tracking-mesh.obj';shutil.move(str(old_path),str(new_path));estimator.mesh_path=str(new_path)
    to_origin,extents=trimesh.bounds.oriented_bounds(mesh)
    bounds=np.stack([-extents/2,extents/2])
    emit({'status':'ready'})
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            started=time.perf_counter()
            with np.load(args.scratch/'frame.npz',allow_pickle=False) as frame:
                rgb=frame['rgb'];depth=frame['depth'].astype(np.float32)*float(frame['scale'])*.001;K=frame['K']
            if np.count_nonzero(depth>.001)<30:raise ValueError('Sin profundidad suficiente para seguir la pieza.')
            with torch.inference_mode():
                pose=valid_pose(estimator.track_one(rgb=rgb,depth=depth,K=K,iteration=2))
            torch.cuda.synchronize()
            image=draw_posed_3d_box(K,img=rgb.copy(),ob_in_cam=pose@np.linalg.inv(to_origin),bbox=bounds)
            image=draw_xyz_axis(image,ob_in_cam=pose,scale=min(float(mesh.extents.max())/2,.1),K=K,thickness=2,is_input_rgb=True)
            ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise RuntimeError('No se pudo codificar el visor de seguimiento.')
            (args.scratch/'preview.jpg').write_bytes(encoded.tobytes())
            transform=pose.copy();transform[:3,3]*=1000
            emit({'status':'frame','seconds':time.perf_counter()-started,
                  'T_camera_from_input_cad_mm':(transform@centered_from_input).tolist()})
    finally:
        # FoundationPose writes a temporary mesh outside its debug directory.
        if estimator.mesh_path:Path(estimator.mesh_path).unlink(missing_ok=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        emit({'status':'error','error':str(error)})
        raise
