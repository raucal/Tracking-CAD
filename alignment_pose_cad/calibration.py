"""Shared hand-eye calibration: DINOPDDL's matrix_calibration.yaml + SDU Calibration.xml.

The device-free loader lives in DINOPDDL so both projects read the same files.
"""
import importlib.util
from pathlib import Path

import numpy as np

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
    result.update(source=config['config_path'], source_xml=config.get('source_xml'), status=config['status'],
                  robot_model=config.get('robot_model'), camera_serial=config.get('camera_id'),
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
        raise ValueError(f"The active RealSense ({camera['serial']}) is not the calibrated camera "
                         f"({config.get('camera_id')}) declared in matrix_calibration.yaml.")
    if config['camera_frame'] != camera['optical_frame']:
        raise ValueError('The optical frame does not match the calibration.')
    return _core.resolve_tcp_camera(config, active_tcp, from_pose)
