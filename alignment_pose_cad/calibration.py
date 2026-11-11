"""One YAML source of truth; reuse DINOPDDL's device-free calibration core."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from .geometry import from_pose

CORE = Path(__file__).resolve().parents[2] / 'PDDL/DINOPDDL/src/dinopddl/robot/hand_eye_calibration.py'
_spec = importlib.util.spec_from_file_location('_cadtracking_hand_eye_core', CORE)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)


def load(robot):
    return _core.load_calibration(robot['hand_eye_file']) if robot.get('hand_eye_file') else None


def describe(config):
    if config is None:
        return {'source': 'legacy_settings', 'status': 'legacy', 'preview_ready': True, 'pending': []}
    result = _core.inspect_calibration(config)
    result.update(source=config['config_path'], status=config['status'], camera_serial=config.get('camera_id'),
                  camera_frame=config['camera_frame'], camera_frame_confirmed=config.get('camera_frame_confirmed') is True,
                  preview_ready=bool(config.get('camera_id') and config.get('camera_frame_confirmed') is True
                                     and (config['parent_frame'] == 'flange' or config.get('tcp_offset_m_rad') is not None)),
                  tcp_offset_at_calibration=config.get('tcp_offset_m_rad'))
    return result


def camera_contract(camera):
    if (camera.get('optical_frame') != 'realsense_color_optical' or camera.get('aligned_to') != 'color'
            or camera.get('rectified') is not True or not camera.get('serial')):
        raise ValueError('A rectified RealSense capture aligned to color with a serial number is required.')
    if np.any(np.abs(camera.get('dist_coeffs', [])) > 1e-10):
        raise ValueError('The capture contains residual distortion.')


def resolve(config, active_tcp, camera):
    camera_contract(camera)
    if config.get('camera_id') != camera['serial'] or config.get('camera_frame_confirmed') is not True:
        raise ValueError('Register the physical camera and optical axes for this calibration first.')
    if config['camera_frame'] != camera['optical_frame']:
        raise ValueError('The optical frame does not match the calibration.')
    return _core.resolve_tcp_camera(config, active_tcp, from_pose)


def confirm_camera(robot, camera, *, expected_sha256, expected_serial, physical_confirmed, optical_confirmed):
    """An explicit operator acknowledgement, never an accuracy/motion approval."""
    camera_contract(camera)
    config = load(robot)
    if config is None:
        raise ValueError('No hay un YAML hand-eye seleccionado.')
    if expected_sha256 != config['config_sha256']:
        raise ValueError('Calibration changed; review and confirm again.')
    if physical_confirmed is not True or optical_confirmed is not True:
        raise ValueError('Explicitly confirm the calibrated camera and color optical frame.')
    if expected_serial != camera['serial']:
        raise ValueError('The active camera changed since it was displayed.')
    if config.get('camera_id') and config['camera_id'] != camera['serial']:
        raise ValueError('The active camera does not match the camera bound to the YAML.')
    if config['camera_frame'] != camera['optical_frame']:
        raise ValueError('The calibration declares a different optical frame.')
    path = Path(config['config_path'])
    data = yaml.safe_load(path.read_text())
    data.update(camera_id=camera['serial'], camera_frame_confirmed=True,
                camera_confirmation={'source': 'operator_in_cadtracking_preview',
                    'confirmed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'camera_metadata': camera, 'physical_camera_confirmed': True,
                    'optical_frame_confirmed': True, 'accuracy_validated': False})
    # Do not modify status, matrix, TCP, optimization result or motion flags.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.hand_eye_', suffix='.yaml', delete=False) as out:
            temporary = Path(out.name)
            yaml.safe_dump(data, out, sort_keys=False, allow_unicode=True)
        os.chmod(temporary, path.stat().st_mode & 0o777)
        if _core.sha256(path) != expected_sha256:
            raise ValueError('Calibration changed during confirmation.')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    result = load(robot)
    with path.with_name('hand_eye_confirmations.jsonl').open('a') as log:
        log.write(json.dumps({'event': 'camera_identity_confirmed', 'before_sha256': expected_sha256,
                   'after_sha256': result['config_sha256'], 'camera': data['camera_confirmation'],
                   'calibration_status': result['status'], 'motion_enabled_by_this_action': False})+'\n')
    return result
