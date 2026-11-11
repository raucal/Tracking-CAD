"""Original FoundationPose: RGB-D registration and recording tracking."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


from settings import path
from foundationpose_sdu import install as install_sdu_compatibility
FOUNDATIONPOSE = Path(os.environ.get('FOUNDATIONPOSE_ROOT', path('foundationpose')))
NATIVE = Path(os.environ.get('FOUNDATIONPOSE_NATIVE', path('native')/'foundationpose'))


def probe():
    import torch
    sys.path[:0] = [str(NATIVE), str(FOUNDATIONPOSE / 'mycpp/build')]
    modules = ['pytorch3d', 'nvdiffrast', 'open3d', 'warp', 'kornia', 'transformations', 'mycpp']
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    weights = all((FOUNDATIONPOSE / 'weights' / stamp / name).is_file()
                  for stamp in ('2024-01-11-20-02-45', '2023-10-28-18-33-37')
                  for name in ('model_best.pth', 'config.yml'))
    available = not missing and weights and torch.cuda.is_available()
    return {'available': available, 'cuda': torch.cuda.is_available(), 'torch': str(torch.__version__),
            'missing_modules': missing, 'weights_available': weights, 'python': sys.executable,
            'reason': None if available else 'Missing dependencies, weights or CUDA access: ' + ', '.join(missing)}


def valid_pose(value):
    import numpy as np
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or pose[2, 3] <= 0
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3)):
        raise ValueError('FoundationPose returned an invalid transformation.')
    return pose


def load_models():
    import logging
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires CUDA.')
    sys.path[:0] = [str(NATIVE), str(FOUNDATIONPOSE)]
    import mycpp
    install_sdu_compatibility(FOUNDATIONPOSE)
    from estimater import ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    torch.set_num_threads(4)
    models = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    logging.getLogger().setLevel(logging.ERROR)
    return models


def estimate(args, runtime=None):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires a CUDA GPU.')
    sys.path[:0] = [str(NATIVE), str(FOUNDATIONPOSE)]
    # Preload the extension for this Python version before importing Utils.
    import mycpp
    install_sdu_compatibility(FOUNDATIONPOSE)
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_seed
    started = time.perf_counter()
    reused_models = bool(runtime and runtime.get("models"))
    runtime = {} if runtime is None else runtime
    torch.set_num_threads(4)
    set_seed(0)
    mesh = trimesh.load(args.mesh, force='mesh')
    mesh.apply_scale(.001)  # The SAM-6D engine supplies a mesh in millimetres.
    camera = json.loads((args.folder / 'camera.json').read_text())
    K = np.asarray(camera['cam_K'], dtype=float).reshape(3, 3)
    depth_scale = float(camera['depth_scale']) * .001
    model_from_input = np.asarray(json.loads(args.transform.read_text()), dtype=float)
    rgb = np.asarray(Image.open(args.folder / 'input_rgb.png').convert('RGB'))
    raw = np.asarray(Image.open(args.folder / 'depth.png'))
    depth = raw.astype(np.float32) * depth_scale
    mask = np.asarray(Image.open(args.mask)) > 0
    if np.count_nonzero(mask & (depth > .001)) < 30:
        raise ValueError('The mask contains insufficient valid depth for pose estimation.')
    if "models" not in runtime:
        runtime["models"] = load_models()
    scorer, refiner, context = runtime["models"]
    import hashlib
    mesh_key = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    if runtime.get('mesh_key') != mesh_key:
        old = runtime.pop('estimator',None)
        if old is not None and old.mesh_path:
            Path(old.mesh_path).unlink(missing_ok=True)
        runtime['estimator'] = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                             mesh=mesh, scorer=scorer, refiner=refiner, glctx=context,
                                             debug=0, debug_dir=str(args.folder / 'foundationpose_debug'))
        runtime['mesh_key'] = mesh_key
    estimator = runtime['estimator']
    estimator.pose_last = None  # Never accept register's insufficient-depth fallback from a previous job.
    runtime.update(last_folder=str(args.folder.resolve()),mesh=mesh,model_from_input=model_from_input,K=K)
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    pose = valid_pose(estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5))
    if estimator.pose_last is None:
        raise ValueError('FoundationPose could not register the object with the available depth.')
    torch.cuda.synchronize()
    registered = time.perf_counter()

    def pose_record(pose):
        transform = pose.copy()
        transform[:3, 3] *= 1000
        return {'R': transform[:3, :3].tolist(), 't_mm': transform[:3, 3].tolist(),
                'T_camera_from_model_mm': transform.tolist(),
                'T_camera_from_input_cad_mm': (transform @ model_from_input).tolist(),
                'translation_unit': 'mm', 'frame': 'OpenCV camera: X right, Y down, Z forward'}

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bounds = np.stack([-extents / 2, extents / 2])
    def overlay(color, transform):
        image = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=transform @ np.linalg.inv(to_origin), bbox=bounds)
        return draw_xyz_axis(image, ob_in_cam=transform, scale=min(float(mesh.extents.max()) / 2, .1),
                             K=K, thickness=2, is_input_rgb=True)
    preview = Image.new('RGB', (rgb.shape[1] * 2, rgb.shape[0]))
    preview.paste(Image.fromarray(rgb), (0, 0))
    preview.paste(Image.fromarray(overlay(rgb, pose)), (rgb.shape[1], 0))
    preview.save(args.folder / 'pose_overlay.png')
    first = pose_record(pose)
    first['pose_score'] = float(estimator.scores[0].detach().cpu())
    result = {'status': 'estimated', 'backend': 'foundationpose', 'poses': [first], 'best_index': 0,
              'pose_score_is_probability': False, 'device': 'cuda', 'accuracy_validated': False, 'models_reused': reused_models, 'worker_pid': os.getpid(),
              'timing': {'model_load_s': loaded - started, 'pose_inference_s': registered - loaded}}
    if args.recording:
        metadata = json.loads((args.recording / 'recording.json').read_text())
        rows = [json.loads(line) for line in (args.recording / 'frames.jsonl').read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError('The recording is empty.')
        writer = cv2.VideoWriter(str(args.folder / 'pose_tracking.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 metadata['fps'], (rgb.shape[1], rgb.shape[0]))
        if not writer.isOpened():
            raise RuntimeError('Could not create the tracking video.')
        try:
            with (args.folder / 'pose_tracking.jsonl').open('w') as output:
                for index, row in enumerate(rows):
                    color = np.asarray(Image.open(args.recording / row['rgb']).convert('RGB'))
                    if index:
                        depth = np.asarray(Image.open(args.recording / row['depth'])).astype(np.float32) * depth_scale
                        pose = valid_pose(estimator.track_one(rgb=color, depth=depth, K=K, iteration=2))
                    entry = {'index': index, 'timestamp': row['timestamp'], **pose_record(pose)}
                    output.write(json.dumps(entry, allow_nan=False) + '\n')
                    output.flush()
                    writer.write(cv2.cvtColor(overlay(color, pose), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        result['tracking'] = {'frames': len(rows), 'source_recording_id': metadata['id'],
                              'poses': 'pose_tracking.jsonl', 'video': 'pose_tracking.mp4'}
    result['timing']['total_s'] = time.perf_counter() - started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    for name in ('folder', 'mesh', 'mask', 'transform', 'output', 'recording'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe()))
        return
    if not all((args.folder, args.mesh, args.mask, args.transform, args.output)):
        parser.error('Se necesitan folder, mesh, mask, transform y output.')
    estimate(args)


if __name__ == '__main__':
    main()
