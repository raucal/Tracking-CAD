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
from tracking_quality import PoseHealth
from settings import profile


def reply(request_id,**value):
    print('@POSE@'+json.dumps(dict(request_id=request_id,**value),allow_nan=False),flush=True)


def configure_tracking(runtime):
    import numpy as np
    import trimesh
    origin, extents = trimesh.bounds.oriented_bounds(runtime['mesh'])
    runtime.update(inv_origin=np.linalg.inv(origin), bounds=np.stack([-extents/2, extents/2]),
                   health=PoseHealth(runtime['mesh'], profile('interface')['tracking']['quality']))
    # Use the original mesh coordinates: estimator.mesh_tensors are internally centered.
    from Utils import make_mesh_tensors
    runtime['cad_view_tensors'] = make_mesh_tensors(runtime['mesh'])


def select_tracking_model(runtime, metadata, scratch):
    """Replace geometry when SAM identifies another CAD, reusing loaded networks."""
    import hashlib
    import numpy as np
    import trimesh
    from estimater import FoundationPose
    mesh_path = Path(metadata['mesh'])
    key = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if runtime.get('mesh_key') != key:
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(.001)
        scorer, refiner, context = runtime['models']
        old = runtime.pop('estimator', None)
        if old is not None and old.mesh_path:
            Path(old.mesh_path).unlink(missing_ok=True)
        runtime['estimator'] = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                             mesh=mesh, scorer=scorer, refiner=refiner, glctx=context,
                                             debug=0, debug_dir=str(scratch / 'foundationpose_debug'))
        runtime.update(mesh=mesh, mesh_key=key)
        configure_tracking(runtime)
    runtime['model_from_input'] = np.asarray(metadata['model_from_input'])


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
                    configure_tracking(runtime)
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
                    metadata = scratch / 'tracking_model.json'
                    if metadata.is_file():
                        select_tracking_model(runtime, json.loads(metadata.read_text()), scratch)
                    estimator=runtime['estimator'];estimator.pose_last=None
                    with torch.inference_mode():pose=backend.valid_pose(estimator.register(K=K,rgb=rgb,depth=depth,ob_mask=mask,iteration=5))
                    if estimator.pose_last is None:raise ValueError('No se pudo registrar la nueva detección.')
                    torch.cuda.synchronize()
                    runtime['health'].reset()
                    timestamp_path=scratch/'timestamp.json'
                    timestamp=json.loads(timestamp_path.read_text()) if timestamp_path.is_file() else time.time()
                    quality=runtime['health'].evaluate(pose,depth,K,timestamp)
                    if not quality['ok']:
                        estimator.pose_last=None
                        raise ValueError('La pose recuperada no concuerda con la profundidad: '+', '.join(quality['reasons']))
                    result={'seconds':time.perf_counter()-started,'quality':quality}
                elif command=='track':
                    import cv2
                    import numpy as np
                    import torch
                    import trimesh
                    from Utils import draw_posed_3d_box
                    from cad_view import draw_reference_axes, render_cad_view
                    started=time.perf_counter();scratch=Path(request['scratch'])
                    with np.load(scratch/'frame.npz',allow_pickle=False) as frame:
                        rgb=frame['rgb'];depth=frame['depth'].astype(np.float32)*float(frame['scale'])*.001;K=frame['K']
                    if np.count_nonzero(depth>.001)<30:raise ValueError('No hay profundidad suficiente para seguir la pieza.')
                    estimator=runtime['estimator'];accepted_pose=estimator.pose_last.clone()
                    with torch.inference_mode():pose=backend.valid_pose(estimator.track_one(rgb=rgb,depth=depth,K=K,iteration=2))
                    timestamp_path=scratch/'timestamp.json'
                    timestamp=json.loads(timestamp_path.read_text()) if timestamp_path.is_file() else time.time()
                    quality=runtime['health'].evaluate(pose,depth,K,timestamp)
                    if not quality['ok']: estimator.pose_last=accepted_pose
                    torch.cuda.synchronize()
                    mesh=runtime['mesh']
                    image=draw_posed_3d_box(K,img=rgb.copy(),ob_in_cam=pose@runtime['inv_origin'],bbox=runtime['bounds'])
                    image=draw_reference_axes(image,pose,K,min(float(mesh.extents.max())/2,.1))
                    if image.shape[1]>640:image=cv2.resize(image,(640,round(image.shape[0]*640/image.shape[1])),interpolation=cv2.INTER_AREA)
                    ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,70])
                    if not ok:raise RuntimeError('Error al codificar el visor.')
                    (scratch/'preview.jpg').write_bytes(jpeg.tobytes())
                    cad_path = scratch/'cad_preview.jpg'
                    cad_path.unlink(missing_ok=True)
                    cad_error = None
                    cad_started = time.perf_counter()
                    if quality['ok']:
                        try:
                            cad_path.write_bytes(render_cad_view(mesh, pose, K, rgb.shape,
                                                               runtime['cad_view_tensors'], runtime['models'][2]))
                        except Exception as error:
                            cad_error = str(error)
                    transform=pose.copy();transform[:3,3]*=1000
                    result={'seconds':time.perf_counter()-started,'quality':quality,'bbox_xyxy':projected_bbox(mesh.bounds,pose,K,rgb.shape),'T_camera_from_input_cad_mm':(transform@runtime['model_from_input']).tolist(),
                            'cad_view_error':cad_error, 'cad_render_s':time.perf_counter()-cad_started}
                else:raise ValueError('Operación de pose no válida.')
            reply(rid,status='ok',result=result)
        except Exception as error:
            traceback.print_exc(file=sys.stderr);reply(rid,status='error',error=str(error))


if __name__=='__main__':main()
