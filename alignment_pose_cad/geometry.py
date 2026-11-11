"""Explicit rigid transforms: original CAD -> camera -> robot base. SI internally."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name}: {size} finite values are required.')
    return a


def rigid(value):
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-3)):
        raise ValueError('Invalid rigid transformation.')
    return T.copy()


def from_pose(pose):
    p = vector(pose, 6, 'pose TCP')
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def to_pose(transform):
    T = rigid(transform)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()].tolist()


def difference(a, b):
    a, b = rigid(a), rigid(b)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())))


def plane_frame(config, bounds_mm):
    normal = vector(config['plane_normal_cad'], 3, 'normal CAD')
    x = vector(config['plane_x_cad'], 3, 'plane X axis')
    if min(np.linalg.norm(normal), np.linalg.norm(x)) < 1e-8:
        raise ValueError('Plane axes cannot be zero.')
    normal = normal / np.linalg.norm(normal)
    x = x - normal * np.dot(x, normal)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('The X axis cannot be parallel to the normal.')
    x /= np.linalg.norm(x)
    if config['plane_origin_cad_mm'] is None:
        # Derive a front-face center only for a principal CAD plane. An oblique
        # plane needs an explicit measured point, not a guessed bounding surface.
        if np.count_nonzero(np.abs(normal) > 1e-6) != 1:
            raise ValueError('Set plane_origin_cad_mm for an oblique plane.')
        bounds = np.asarray(bounds_mm, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError('Invalid CAD bounds.')
        origin = bounds.mean(axis=0) + normal * np.dot(np.abs(normal), (bounds[1] - bounds[0]) / 2)
    else:
        origin = vector(config['plane_origin_cad_mm'], 3, 'CAD plane origin')
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, np.cross(normal, x), normal))
    T[:3, 3] = origin * .001
    return T


def build_plan(camera_from_cad_mm, capture_tcp_pose, settings, bounds_mm, cad_key):
    camera_from_cad = rigid(camera_from_cad_mm)
    if camera_from_cad[2, 3] <= 0:
        raise ValueError('The CAD is behind the camera.')
    camera_from_cad[:3, 3] *= .001
    cfg = {**settings['alignment'], **settings['alignment']['cad_overrides'].get(cad_key, {})}
    cad_from_plane = plane_frame(cfg, bounds_mm)
    camera_from_plane = camera_from_cad @ cad_from_plane
    if np.dot(camera_from_plane[:3, 2], -camera_from_plane[:3, 3]) <= 0:
        raise ValueError('The plane normal points away from the camera. Check plane_normal_cad.')
    base_from_tcp = from_pose(capture_tcp_pose)
    tcp_from_camera = from_pose(settings['robot']['tcp_from_camera_m_rad'])
    base_from_cad = base_from_tcp @ tcp_from_camera @ camera_from_cad
    base_from_plane = base_from_cad @ cad_from_plane
    plane_from_target = np.eye(4)
    # Tool Z points toward the plane; X follows plane X plus configurable yaw.
    plane_from_target[:3, :3] = (Rotation.from_euler('z', cfg['tool_yaw_deg'], degrees=True).as_matrix()
                                       @ np.diag([1., -1., -1.]))
    plane_from_target[:3, 3] = [*cfg['in_plane_offset_m'], cfg['offset_m']]
    target = base_from_plane @ plane_from_target
    translation, rotation = difference(base_from_tcp, target)
    limits = settings['motion']
    if translation > limits['max_translation_m'] or rotation > limits['max_rotation_deg']:
        raise ValueError(f'Target exceeds the step limit: {translation:.3f} m / {rotation:.1f} degrees.')
    low, high = np.asarray(limits['workspace_min_m']), np.asarray(limits['workspace_max_m'])
    # Convex Cartesian box; both endpoints imply the straight segment is inside.
    if np.any(np.minimum(base_from_tcp[:3, 3], target[:3, 3]) < low) or np.any(np.maximum(base_from_tcp[:3, 3], target[:3, 3]) > high):
        raise ValueError('The trajectory leaves the configured workspace.')
    times = np.linspace(0, 1, limits['path_samples'])
    rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([base_from_tcp[:3, :3], target[:3, :3]])))(times)
    waypoints = [np.r_[(1-t)*base_from_tcp[:3, 3]+t*target[:3, 3], r.as_rotvec()].tolist()
                 for t, r in zip(times, rotations)]
    return {'cad_key': cad_key, 'offset_m': cfg['offset_m'], 'offset_frame': 'CAD plane outward normal',
            'target_frame': 'TCP', 'translation_unit': 'm', 'rotation_unit': 'rad (rotation vector)',
            'T_base_from_tcp_capture': base_from_tcp.tolist(), 'T_tcp_from_camera': tcp_from_camera.tolist(),
            'T_camera_from_input_cad_m': camera_from_cad.tolist(), 'T_cad_from_plane': cad_from_plane.tolist(),
            'T_base_from_cad': base_from_cad.tolist(), 'T_base_from_plane': base_from_plane.tolist(),
            'T_plane_from_tcp_target': plane_from_target.tolist(), 'T_base_from_tcp_target': target.tolist(),
            'tcp_target_m_rad': to_pose(target), 'translation_m': translation, 'rotation_deg': rotation,
            'path_samples_m_rad': waypoints}
