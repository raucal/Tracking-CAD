"""Estimate the 6D pose of one catalog CAD in saved RGB-D, for external robot clients.

Used by DINOPDDL (``alignment.method: cad_pose``) to align the TCP to the pose a
skill was recorded from. Run through the launcher so the runtime environment is
set: ``bash run.sh estimate --serve ...``.

Serve mode keeps SAM-6D and the pose backend resident and reads one JSON request
per stdin line::

    {"id": "...", "rgb": "rgb.png", "depth": "depth.png|npy", "camera": "camera.json",
     "cad_key": "02_control_panel", "output": "/new/folder"}

Every response is one stdout line ``@@CADPOSE {json}``. Anything else the models
print goes to stderr, so the protocol channel stays clean. Inputs are never
modified; each request writes its artifacts into a new ``output`` folder.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import traceback

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from settings import PREPARED  # noqa: E402

PREFIX = '@@CADPOSE '
FIXED_OPTIONS = {'mode': 'rgbd', 'device': 'cuda', 'cad_units': 'mm', 'pose': True}


def upload(file):
    file = Path(file)
    return {'name': file.name, 'data': base64.b64encode(file.read_bytes()).decode('ascii')}


def selected_options(args):
    options = dict(FIXED_OPTIONS, segmentor=args.segmentor, backbone=args.backbone,
                   pose_backend=args.pose_backend, confidence=args.confidence,
                   min_score=args.min_score, fastsam_conf=args.fastsam_conf)
    if options['min_score'] > options['confidence']:
        raise ValueError('min-score cannot exceed confidence.')
    return options


def estimate(engine, parts, options, request, log):
    from sam6d_ui.inputs import prepare_upload
    key = request.get('cad_key')
    if key not in parts:
        raise ValueError(f'Unknown catalog CAD: {key!r}.')
    part = parts[key]
    output = Path(request['output']).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    inputs = prepare_upload({'rgb': upload(request['rgb']), 'depth': upload(request['depth']),
                             'camera': upload(request['camera']), 'cad': upload(PREPARED / part['source_stl']),
                             'options': options}, output)
    result = engine.run(output, inputs, lambda message: log(f'{key}: {message}'))
    pose = result.get('pose') or {}
    poses = pose.get('poses') or []
    response = {'status': 'no_accepted_detection', 'cad_key': key, 'cad_sha256': part['source_sha256'],
                'backend': result.get('backend'), 'options': inputs['options'],
                'sam6d_score': result.get('best_score'), 'output': str(output),
                'timing': result.get('timing'), 'model_reused': result.get('model_reused')}
    if result.get('recognized') and poses:
        best = poses[pose.get('best_index', 0)]
        response.update(status='estimated', T_camera_from_input_cad_mm=best['T_camera_from_input_cad_mm'],
                        pose_score=best.get('pose_score'),
                        frame='OpenCV camera: X right, Y down, Z forward', translation_unit='mm')
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--serve', action='store_true', help='Read JSON requests from stdin until EOF')
    parser.add_argument('--rgb', type=Path)
    parser.add_argument('--depth', type=Path)
    parser.add_argument('--camera', type=Path)
    parser.add_argument('--cad-key')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--segmentor', choices=('sam', 'fastsam'), default='sam')
    parser.add_argument('--backbone', choices=('vit_h', 'vit_l', 'vit_b'), default='vit_h')
    parser.add_argument('--pose-backend', choices=('foundationpose', 'pem'), default='foundationpose')
    parser.add_argument('--confidence', type=float, default=.4)
    parser.add_argument('--min-score', type=float, default=.3)
    parser.add_argument('--fastsam-conf', type=float, default=.05)
    args = parser.parse_args()
    if not args.serve and not all((args.rgb, args.depth, args.camera, args.cad_key, args.output)):
        parser.error('Use --serve, or give --rgb, --depth, --camera, --cad-key and --output.')
    options = selected_options(args)
    # Keep fd 1 for the protocol only; model/worker output goes to stderr.
    protocol = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(data):
        protocol.write(PREFIX + json.dumps(data, allow_nan=False) + '\n')
        protocol.flush()

    def log(message):
        print(message, file=sys.stderr, flush=True)

    from sam6d_ui.engine import Engine
    from sam6d_ui.backends.foundationpose import resident
    from settings import ROOT
    manifest = json.loads((PREPARED / 'manifest.json').read_text())
    parts = {part['key']: part for part in manifest['parts']}
    engine = Engine(ROOT / 'results/interface')
    try:
        if not args.serve:
            reply(estimate(engine, parts, options, {'rgb': args.rgb, 'depth': args.depth, 'camera': args.camera,
                                                    'cad_key': args.cad_key, 'output': args.output}, log))
            return 0
        try:
            timing = engine.warmup(options['segmentor'], options['backbone'], log)
            reply({'ready': True, 'options': options, 'cad_keys': sorted(parts), 'timing': timing})
        except Exception as exc:
            reply({'ready': False, 'error': str(exc)})
            return 1
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get('id')
                reply(dict(estimate(engine, parts, options, request, log), id=request_id))
            except Exception as exc:
                log(traceback.format_exc())
                reply({'id': request_id, 'status': 'error', 'error': str(exc)})
        return 0
    finally:
        resident().close()


if __name__ == '__main__':
    sys.exit(main())
