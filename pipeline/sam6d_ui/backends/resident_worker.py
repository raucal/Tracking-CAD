"""One GPU process for registration and live tracking, with weights loaded at startup."""
import contextlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import foundationpose_worker as backend
from geometry import projected_bbox


def reply(request_id,**value):
    print('@POSE@'+json.dumps(dict(request_id=request_id,**value),allow_nan=False),flush=True)


def main():
    runtime={}
    import atexit
    def cleanup():
        estimator=runtime.get('estimator')
        if estimator is not None and estimator.mesh_path:Path(estimator.mesh_path).unlink(missing_ok=True)
    atexit.register(cleanup)
    started=time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):runtime['models']=backend.load_models()
    reply(None,status='ready',load_s=time.perf_counter()-started)
    for line in sys.stdin:
        request=json.loads(line);rid=request['id'];command=request['command']
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if command=='warmup':result={'ready':True}
                elif command=='estimate':
                    args=SimpleNamespace(**{key:Path(value) if value is not None else None for key,value in request['args'].items()})
                    result=backend.estimate(args,runtime)
                elif command=='start_tracking':
                    if str(Path(request['folder']).resolve())!=runtime.get('last_folder') or runtime['estimator'].pose_last is None:
                        raise ValueError('Vuelve a estimar la pose inicial antes del seguimiento.')
                    import numpy as np
                    import trimesh
                    origin,extents=trimesh.bounds.oriented_bounds(runtime['mesh'])
                    runtime.update(inv_origin=np.linalg.inv(origin),bounds=np.stack([-extents/2,extents/2]))
                    result={'ready':True}
                elif command=='relocalize':
                    import numpy as np
                    import torch
                    scratch=Path(request['scratch']);started=time.perf_counter()
                    with np.load(scratch/'frame.npz',allow_pickle=False) as frame:
                        rgb=frame['rgb'];depth=frame['depth'].astype(np.float32)*float(frame['scale'])*.001;K=frame['K']
                    mask=np.load(scratch/'mask.npy',allow_pickle=False).astype(bool)
                    if mask.shape!=depth.shape or np.count_nonzero(mask & (depth>.001))<30:
                        raise ValueError('La nueva detección no tiene suficiente profundidad válida.')
                    estimator=runtime['estimator'];estimator.pose_last=None
                    with torch.inference_mode():pose=backend.valid_pose(estimator.register(K=K,rgb=rgb,depth=depth,ob_mask=mask,iteration=5))
                    if estimator.pose_last is None:raise ValueError('No se pudo registrar la nueva detección.')
                    torch.cuda.synchronize()
                    result={'seconds':time.perf_counter()-started}
                elif command=='track':
                    import cv2
                    import numpy as np
                    import torch
                    import trimesh
                    from Utils import draw_posed_3d_box, draw_xyz_axis
                    started=time.perf_counter();scratch=Path(request['scratch'])
                    with np.load(scratch/'frame.npz',allow_pickle=False) as frame:
                        rgb=frame['rgb'];depth=frame['depth'].astype(np.float32)*float(frame['scale'])*.001;K=frame['K']
                    if np.count_nonzero(depth>.001)<30:raise ValueError('No hay profundidad suficiente para seguir la pieza.')
                    with torch.inference_mode():pose=backend.valid_pose(runtime['estimator'].track_one(rgb=rgb,depth=depth,K=K,iteration=2))
                    torch.cuda.synchronize()
                    mesh=runtime['mesh']
                    image=draw_posed_3d_box(K,img=rgb.copy(),ob_in_cam=pose@runtime['inv_origin'],bbox=runtime['bounds'])
                    image=draw_xyz_axis(image,ob_in_cam=pose,scale=min(float(mesh.extents.max())/2,.1),K=K,thickness=2,is_input_rgb=True)
                    if image.shape[1]>640:image=cv2.resize(image,(640,round(image.shape[0]*640/image.shape[1])),interpolation=cv2.INTER_AREA)
                    ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,70])
                    if not ok:raise RuntimeError('Error al codificar el visor.')
                    (scratch/'preview.jpg').write_bytes(jpeg.tobytes())
                    transform=pose.copy();transform[:3,3]*=1000
                    result={'seconds':time.perf_counter()-started,'bbox_xyxy':projected_bbox(mesh.bounds,pose,K,rgb.shape),'T_camera_from_input_cad_mm':(transform@runtime['model_from_input']).tolist()}
                else:raise ValueError('Operación de pose no válida.')
            reply(rid,status='ok',result=result)
        except Exception as error:
            traceback.print_exc(file=sys.stderr);reply(rid,status='error',error=str(error))


if __name__=='__main__':main()
