"""Process saved RGB-D demos using SAM6D masks and FoundationPose, without editing inputs.

Run with the configured SAM6D Python and an explicitly selected camera JSON.
All outputs go into a fresh subdirectory in each demo; originals are hash-checked.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from settings import PREPARED, ROOT, path

DEMO_PARTS = {
    'HMI_skill': '02_control_panel',
    'cassette_skill': '03_option_cassette',
    'communications_skill': '03b_vertical_option_b',
    'control_board_skill': '05_control_board',
    'front_housing_pick_and_place': '01_front_housing',
    'front_housing_skill': '01_front_housing',
    'io_terminals_skill': '04_io_terminals',
    'new_skill': '02_control_panel',
    'option_cassette_new': '03_option_cassette',
    'power_unit_skill': '06_power_unit',
    'sensor_skill': '03a_vertical_option_a',
    'termal_fan_skill': '07_thermal_core',
}


def digest(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def upload(file):
    file = Path(file)
    return {'name': file.name, 'data': base64.b64encode(file.read_bytes()).decode('ascii')}


def write_json(file, data):
    with Path(file).open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def draw_result(folder, result, part):
    """Draw the saved detection plus centered CAD axes, with an explicit frame record."""
    import cv2
    import numpy as np
    from PIL import Image
    rgb = np.array(Image.open(folder / 'input_rgb.png').convert('RGB'))
    accepted = [(i, d) for i, d in enumerate(result['detections'])
                if d['score'] >= result['options']['confidence']]
    if accepted:
        index, detection = accepted[0]
        mask = np.array(Image.open(folder / f'mask_{index:03d}.png')) > 0
        rgb[mask] = (rgb[mask] * 0.78 + np.array([30, 220, 90]) * 0.22).astype('uint8')
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (30, 220, 90), 2)
    pose = result.get('pose') or {}
    axes = None
    if pose.get('poses'):
        record = pose['poses'][pose.get('best_index', 0)]
        T = np.array(record['T_camera_from_model_mm'], dtype=float)
        K = np.array(json.loads((folder / 'camera.json').read_text())['cam_K']).reshape(3, 3)
        bounds = np.array(part['bounds_local_mm'])
        length = min(float((bounds[1] - bounds[0]).max()) * 0.35, 60.)
        points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        camera = points @ T[:3, :3].T + T[:3, 3]
        pixels = camera @ K.T
        if np.any(camera[:, 2] <= 0):
            raise ValueError('Axes project behind the camera')
        pixels = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
        origin = tuple(pixels[0].tolist())
        for i, color, label in [(1, (255, 0, 0), 'X'), (2, (0, 220, 0), 'Y'), (3, (0, 100, 255), 'Z')]:
            tip = tuple(pixels[i].tolist())
            cv2.arrowedLine(rgb, origin, tip, color, 3, cv2.LINE_AA, tipLength=.18)
            cv2.putText(rgb, label, tip, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
        cv2.circle(rgb, origin, 4, (255, 255, 255), -1)
        axes = {'frame': 'centered_cad_model', 'axis_length_mm': length,
                'T_camera_from_overlay_axes_mm': T.tolist(),
                'origin_pixel': list(origin), 'axis_endpoint_pixels': pixels[1:].tolist(),
                'note': 'Axes are at the prepared CAD center. Original CAD pose is T_camera_from_input_cad_mm.'}
    label = f"{part['key']} | SAM6D {result['best_score']:.3f}" if result['best_score'] is not None else part['key']
    cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 48), (20, 20, 20), -1)
    cv2.putText(rgb, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    state = 'FoundationPose | axes at CAD center' if axes else 'No accepted pose'
    cv2.putText(rgb, state, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(rgb).save(folder / 'detection_pose.png')
    return axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demos', type=Path, required=True)
    parser.add_argument('--camera', type=Path, required=True, help='BOP cam_K/depth_scale JSON; scale in mm/raw unit')
    parser.add_argument('--calibration-source', required=True, help='Origin and applicability of the selected calibration')
    parser.add_argument('--output-name', default='sam6d_foundationpose_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--demo', action='append', help='Select a subset by directory name')
    parser.add_argument('--confidence', type=float, default=.4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    demos = args.demos.resolve()
    camera = args.camera.resolve()
    if Path(args.output_name).name != args.output_name or args.output_name in ('.', '..', ''):
        parser.error('output-name must be a new directory name')
    manifest = json.loads((PREPARED / 'manifest.json').read_text())
    parts = {p['key']: p for p in manifest['parts']}
    folders = [p for p in sorted(demos.iterdir()) if p.is_dir() and (not args.demo or p.name in args.demo)]
    if not folders or (args.demo and set(args.demo) - {p.name for p in folders}):
        parser.error('Unknown or empty demo selection')
    originals = {}
    for demo in folders:
        if demo.name not in DEMO_PARTS:
            parser.error('No CAD mapping for ' + demo.name)
        for name in ('btn_rgb.png', 'btn_depth.npy', 'trajectory_meta.json'):
            if not (demo / name).is_file(): parser.error(f'Missing {demo / name}')
        if (demo / args.output_name).exists():
            parser.error('Output exists: ' + str(demo / args.output_name))
        originals.update({str(f): digest(f) for f in demo.rglob('*') if f.is_file()})
    camera_data = json.loads(camera.read_text())
    if 'cam_K' not in camera_data or 'depth_scale' not in camera_data:
        parser.error('Camera JSON must include cam_K and depth_scale in mm/raw unit')
    print(json.dumps({'demos': {d.name: DEMO_PARTS[d.name] for d in folders},
                      'camera': str(camera), 'output_name': args.output_name,
                      'original_files': len(originals)}, indent=2), flush=True)
    if args.dry_run: return 0
    # Match launcher environment without modifying any demo or model source.
    os.environ.setdefault('CAD_TRACKING_ROOT', str(ROOT))
    os.environ.setdefault('FOUNDATIONPOSE_PYTHON', str(path('foundationpose_python')))
    os.environ.setdefault('FOUNDATIONPOSE_ROOT', str(path('foundationpose')))
    os.environ.setdefault('FOUNDATIONPOSE_NATIVE', str(path('native') / 'foundationpose'))
    os.environ.setdefault('PYTHONPATH', str(PIPELINE))
    os.environ.setdefault('MPLCONFIGDIR', str(path('matplotlib_cache')))
    os.environ.setdefault('XDG_CACHE_HOME', str(path('runtime_cache')))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(path('runtime_cache') / 'torch_extensions'))
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    from sam6d_ui.inputs import prepare_upload
    from sam6d_ui.engine import Engine
    from sam6d_ui.backends.foundationpose import resident
    engine = Engine(ROOT / 'results/interface')
    summary = []
    try:
        for demo in folders:
            part = parts[DEMO_PARTS[demo.name]]
            folder = demo / args.output_name
            print(f'PROCESS {demo.name}: {part["key"]}', flush=True)
            provenance = {'rgb': str(demo / 'btn_rgb.png'), 'depth': str(demo / 'btn_depth.npy'),
                          'camera_source': str(camera), 'camera_sha256': digest(camera),
                          'calibration_applicability': args.calibration_source,
                          'original_sha256': {str(Path(k).relative_to(demo)): v for k, v in originals.items() if Path(k).is_relative_to(demo)},
                          'cad_key': part['key'], 'cad_source': str((PREPARED / part['source_stl']).resolve()),
                          'cad_sha256': part['source_sha256'],
                          'cad_selection': 'demo name' if demo.name != 'new_skill' else 'HMI panel visible in the saved RGB image; not inferred from filename',
                          'trajectory_used_for_pose': False,
                          'note': 'Pose is estimated from saved RGB-D, not from robot TCP trajectories.'}
            try:
                request = {'rgb': upload(demo / 'btn_rgb.png'), 'depth': upload(demo / 'btn_depth.npy'),
                           'camera': upload(camera), 'cad': upload(PREPARED / part['source_stl']),
                           'options': {'mode': 'rgbd', 'device': 'cuda', 'cad_units': 'mm', 'segmentor': 'sam',
                                       'backbone': 'vit_h', 'pose': True, 'pose_backend': 'foundationpose',
                                       'confidence': args.confidence, 'min_score': min(.3, args.confidence)}}
                inputs = prepare_upload(request, folder)
                write_json(folder / 'source_provenance.json', provenance)
                result = engine.run(folder, inputs, lambda message: print(demo.name + ': ' + message, flush=True))
                axes = draw_result(folder, result, part)
                poses = (result.get('pose') or {}).get('poses', [])
                best = poses[(result.get('pose') or {}).get('best_index', 0)] if poses else None
                T = best['T_camera_from_input_cad_mm'] if best else None
                output = {'schema_version': 1, 'demo': demo.name, 'cad_key': part['key'], 'cad_name': engine.names.get(part['key'], part['name']),
                          'status': 'estimated' if best else 'no_accepted_detection',
                          'method': 'SAM6D (SAM ViT-H + DINOv2 RGB-D ISM) + FoundationPose',
                          'frame': 'OpenCV camera: X right, Y down, Z forward',
                          'transform_definition': 'p_camera_mm = T_camera_from_input_cad_mm @ p_input_cad_mm',
                          'translation_unit': 'mm', 'T_camera_from_input_cad_mm': T,
                          'R_camera_from_input_cad': [row[:3] for row in T[:3]] if T else None,
                          't_camera_from_input_cad_mm': [row[3] for row in T[:3]] if T else None,
                          'T_camera_from_centered_model_mm': best['T_camera_from_model_mm'] if best else None,
                          'overlay_axes': axes, 'sam6d_score': result['best_score'],
                          'foundationpose_score': best.get('pose_score') if best else None,
                          'accuracy_validated': False, 'score_is_probability': False,
                          'camera': camera_data, 'source': provenance, 'timing': result['timing']}
                write_json(folder / 'cad_pose_camera.json', output)
                summary.append({'demo': demo.name, 'cad': part['key'], 'status': output['status'], 'score': result['best_score'], 'output': str(folder)})
                print('DONE ' + json.dumps(summary[-1]), flush=True)
            except Exception as error:
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / 'processing_error.json', {'error': str(error), 'traceback': traceback.format_exc(), 'source': provenance})
                summary.append({'demo': demo.name, 'status': 'error', 'error': str(error), 'output': str(folder)})
                print('ERROR ' + json.dumps(summary[-1]), flush=True)
    finally:
        resident().close()
        changed = [f for f, value in originals.items() if not Path(f).is_file() or digest(f) != value]
        print(json.dumps({'results': summary, 'originals_checked': len(originals), 'originals_changed': changed}, indent=2), flush=True)
        if changed: raise RuntimeError('Original files changed: ' + str(changed))
    return 0 if all(r['status'] == 'estimated' for r in summary) else 1


if __name__ == '__main__':
    raise SystemExit(main())
