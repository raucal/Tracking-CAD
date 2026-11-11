"""Strict settings loading; no Python configuration execution or hardware imports."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .geometry import from_pose, vector, plane_frame
from . import calibration

DEFAULT = Path(__file__).with_name('settings.json')


def load(filename=DEFAULT):
    filename = Path(filename)
    raw = filename.read_bytes()
    cfg = json.loads(raw)
    if set(cfg) != {'robot', 'alignment', 'motion'}:
        raise ValueError('settings.json requires robot, alignment and motion.')
    defaults = json.loads(DEFAULT.read_text())
    for section in cfg:
        optional = {'hand_eye_file', 'preview_only', 'execution_mode'} if section == 'robot' else set()
        if set(cfg[section]) - set(defaults[section]) or (set(defaults[section]) - optional) - set(cfg[section]):
            raise ValueError('Invalid settings keys: ' + section)
    robot, alignment, motion = cfg['robot'], cfg['alignment'], cfg['motion']
    robot.setdefault('hand_eye_file', '')
    robot.setdefault('preview_only', False)
    robot.setdefault('execution_mode', 'validated')
    if robot['execution_mode'] not in ('validated', 'supervised_trial'):
        raise ValueError('Invalid execution_mode.')
    if not isinstance(robot['ip'], str) or not robot['ip'] or any(c.isspace() for c in robot['ip']):
        raise ValueError('Invalid robot IP address.')
    for name in ('calibration_confirmed', 'motion_enabled', 'preview_only'):
        if type(robot[name]) is not bool: raise ValueError(name + ' must be a boolean.')
    if not isinstance(robot['camera_serial'], str): raise ValueError('camera_serial must be text.')
    from_pose(robot['tcp_offset_m_rad']); from_pose(robot['tcp_from_camera_m_rad'])
    if not isinstance(robot['hand_eye_file'], str):raise ValueError('hand_eye_file must be a path.')
    calibration_bytes = b''
    if robot['hand_eye_file']:
        robot['hand_eye_file'] = str((filename.resolve().parent / robot['hand_eye_file']).resolve())
        hand_eye = calibration.load(robot)
        if hand_eye['robot_id'] != robot['ip']:
            raise ValueError('The robot does not match the identity declared in the hand-eye YAML.')
        robot['camera_serial'] = hand_eye.get('camera_id') or ''
        calibration_bytes = Path(robot['hand_eye_file']).read_bytes()
    for name, value in {**motion, 'frequency_hz': robot['frequency_hz'], 'offset_m': alignment['offset_m']}.items():
        if name in ('workspace_min_m', 'workspace_max_m'):continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be positive and finite.')
    if not 1 <= robot['frequency_hz'] <= 500: raise ValueError('RTDE frequency is out of range.')
    if type(motion['path_samples']) is not int or not 2 <= motion['path_samples'] <= 101:
        raise ValueError('path_samples must be between 2 and 101.')
    if motion['speed_m_s'] > .25 or motion['acceleration_m_s2'] > .5:
        raise ValueError('Speed/acceleration exceeds the limits of this module.')
    if motion['max_rotation_deg'] > 180:
        raise ValueError('max_rotation_deg must be at most 180 (shortest-arc orientation changes).')
    low = vector(motion['workspace_min_m'], 3, 'workspace_min_m')
    high = vector(motion['workspace_max_m'], 3, 'workspace_max_m')
    if np.any(low >= high): raise ValueError('Invalid workspace.')
    if not isinstance(alignment['cad_overrides'], dict):raise ValueError('cad_overrides must be an object.')
    for extra in [{}, *alignment['cad_overrides'].values()]:
        if not isinstance(extra, dict) or set(extra) - {'plane_normal_cad','plane_x_cad','plane_origin_cad_mm','in_plane_offset_m','tool_yaw_deg','offset_m','orientation_mode'}:
            raise ValueError('Invalid CAD override.')
        merged = {**alignment, **extra}
        if merged.get('orientation_mode', 'plane_frame') not in ('plane_frame', 'minimum_z', 'cad_axes', 'cad_axes_inward'):raise ValueError('Invalid orientation_mode.')
        if robot['execution_mode']=='supervised_trial' and merged.get('orientation_mode')=='plane_frame':
            if (not np.array_equal(merged['plane_normal_cad'],[0,1,0]) or
                    not np.array_equal(merged['plane_x_cad'],[1,0,0]) or merged['tool_yaw_deg']!=0):
                raise ValueError('Plane-frame trial alignment requires TCP +Y = CAD +Z and TCP +Z = CAD -Y: use CAD normal +Y, plane X +X and zero yaw.')
        vector(merged['in_plane_offset_m'], 2, 'in_plane_offset_m')
        if not isinstance(merged['tool_yaw_deg'], (int,float)) or not math.isfinite(merged['tool_yaw_deg']):raise ValueError('Invalid tool_yaw_deg.')
        if not isinstance(merged['offset_m'], (int,float)) or not math.isfinite(merged['offset_m']) or merged['offset_m']<=0:raise ValueError('Invalid offset_m.')
        plane_frame(merged, [[-1,-1,-1],[1,1,1]])
    if robot['execution_mode'] == 'supervised_trial':
        if not robot['hand_eye_file']:
            raise ValueError('Supervised trial requires the shared hand-eye YAML.')
    return cfg, hashlib.sha256(raw + calibration_bytes).hexdigest()
