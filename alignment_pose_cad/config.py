"""Strict settings loading; no Python configuration execution or hardware imports."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .geometry import from_pose, vector, plane_frame

DEFAULT = Path(__file__).with_name('settings.json')


def load(filename=DEFAULT):
    filename = Path(filename)
    raw = filename.read_bytes()
    cfg = json.loads(raw)
    if set(cfg) != {'robot', 'alignment', 'motion'}:
        raise ValueError('settings.json requires robot, alignment and motion.')
    defaults = json.loads(DEFAULT.read_text())
    for section in cfg:
        if set(cfg[section]) != set(defaults[section]):
            raise ValueError('Invalid settings keys: ' + section)
    robot, alignment, motion = cfg['robot'], cfg['alignment'], cfg['motion']
    if not isinstance(robot['ip'], str) or not robot['ip'] or any(c.isspace() for c in robot['ip']):
        raise ValueError('Invalid robot IP address.')
    for name in ('calibration_confirmed', 'motion_enabled'):
        if type(robot[name]) is not bool: raise ValueError(name + ' must be a boolean.')
    if not isinstance(robot['camera_serial'], str): raise ValueError('camera_serial must be text.')
    from_pose(robot['tcp_offset_m_rad']); from_pose(robot['tcp_from_camera_m_rad'])
    for name, value in {**motion, 'frequency_hz': robot['frequency_hz'], 'offset_m': alignment['offset_m']}.items():
        if name in ('workspace_min_m', 'workspace_max_m'):continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be positive and finite.')
    if not 1 <= robot['frequency_hz'] <= 500: raise ValueError('RTDE frequency is out of range.')
    if type(motion['path_samples']) is not int or not 2 <= motion['path_samples'] <= 101:
        raise ValueError('path_samples must be between 2 and 101.')
    if motion['speed_m_s'] > .25 or motion['acceleration_m_s2'] > .5:
        raise ValueError('Speed/acceleration exceeds the limits of this module.')
    low = vector(motion['workspace_min_m'], 3, 'workspace_min_m')
    high = vector(motion['workspace_max_m'], 3, 'workspace_max_m')
    if np.any(low >= high): raise ValueError('Invalid workspace.')
    if not isinstance(alignment['cad_overrides'], dict):raise ValueError('cad_overrides must be an object.')
    for extra in [{}, *alignment['cad_overrides'].values()]:
        if not isinstance(extra, dict) or set(extra) - {'plane_normal_cad','plane_x_cad','plane_origin_cad_mm','in_plane_offset_m','tool_yaw_deg','offset_m'}:
            raise ValueError('Invalid CAD override.')
        merged = {**alignment, **extra}
        vector(merged['in_plane_offset_m'], 2, 'in_plane_offset_m')
        if not isinstance(merged['tool_yaw_deg'], (int,float)) or not math.isfinite(merged['tool_yaw_deg']):raise ValueError('Invalid tool_yaw_deg.')
        if not isinstance(merged['offset_m'], (int,float)) or not math.isfinite(merged['offset_m']) or merged['offset_m']<=0:raise ValueError('Invalid offset_m.')
        plane_frame(merged, [[-1,-1,-1],[1,1,1]])
    return cfg, hashlib.sha256(raw).hexdigest()
