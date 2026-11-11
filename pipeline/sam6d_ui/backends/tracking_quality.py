"""Bounded CPU checks on metric CAD surface/depth and accepted pose motion.

This is a tracking-loss heuristic, not an accuracy/confidence certificate.
"""
import numpy as np


class PoseHealth:
    def __init__(self, mesh, config):
        self.config = dict(config)
        self.previous = None
        self.previous_time = None
        self.center = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
        triangles = np.asarray(mesh.triangles, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        if not len(triangles) or not np.isfinite(areas).all() or areas.sum() <= 0:
            raise ValueError('The CAD has no valid surface for depth checks.')
        rng = np.random.default_rng(0)
        selected = rng.choice(len(triangles), self.config['sample_points'], p=areas / areas.sum())
        uv = rng.random((len(selected), 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        faces = triangles[selected]
        self.points = faces[:, 0] + uv[:, :1] * (faces[:, 1] - faces[:, 0]) + uv[:, 1:] * (faces[:, 2] - faces[:, 0])
        self.normals = np.asarray(mesh.face_normals, dtype=float)[selected]

    def reset(self):
        self.previous = self.previous_time = None

    def evaluate(self, pose, depth_m, K, timestamp):
        cfg = self.config
        result = {'ok': True, 'enabled': cfg['enabled'], 'reasons': [], 'motion_checked': False}
        if not cfg['enabled']:
            return result
        pose = np.asarray(pose, dtype=float)
        depth = np.asarray(depth_m)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or pose[2, 3] <= 0:
            return dict(result, ok=False, reasons=['invalid_pose'])
        camera = self.points @ pose[:3, :3].T + pose[:3, 3]
        normals = self.normals @ pose[:3, :3].T
        # Test visible-facing surface samples; avoid comparing back surfaces to front depth.
        front = (normals * camera).sum(axis=1) < 0
        valid = front & np.isfinite(camera).all(axis=1) & (camera[:, 2] > .001)
        projected = camera @ np.asarray(K).T
        pixels = np.zeros((len(camera), 2), dtype=int)
        pixels[valid] = np.rint(projected[valid, :2] / projected[valid, 2:]).astype(int)
        h, w = depth.shape
        inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        visible_fraction = float(inside.sum() / max(1, valid.sum()))
        result['visible_fraction'] = visible_fraction
        if visible_fraction < cfg['min_visible_fraction']:
            result['reasons'].append('outside_image')
        indices = np.flatnonzero(inside)
        if len(indices):
            # Coarse z buffer: retain the nearest sample per 4x4 pixel tile.
            order = indices[np.argsort(camera[indices, 2])]
            tiles = (pixels[order, 1] // 4) * ((w + 3) // 4) + pixels[order, 0] // 4
            _, first = np.unique(tiles, return_index=True)
            chosen = order[first]
            observed = depth[pixels[chosen, 1], pixels[chosen, 0]]
            available = np.isfinite(observed) & (observed > .001)
            predicted = camera[chosen, 2]
            tolerance = cfg['depth_tolerance_mm'] / 1000
            occluded = available & (observed < predicted - tolerance)
            occlusion = float(occluded.sum() / max(1, available.sum()))
            considered = available & ~occluded
            residual = np.abs(observed[considered] - predicted[considered])
            support = float(np.mean(residual <= tolerance)) if len(residual) else 0.0
            result.update(depth_samples=int(considered.sum()), depth_support=support,
                          valid_depth_fraction=float(available.mean()), occlusion_fraction=occlusion,
                          median_depth_error_mm=float(np.median(residual) * 1000) if len(residual) else None)
            if occlusion > cfg['max_occlusion_fraction']:
                result['reasons'].append('occluded')
            if considered.sum() < cfg['min_depth_samples'] or available.mean() < cfg['min_valid_depth_fraction']:
                result['reasons'].append('insufficient_object_depth')
            elif support < cfg['min_depth_support']:
                result['reasons'].append('depth_mismatch')
        else:
            result.update(depth_samples=0, depth_support=0.0)
            result['reasons'].append('no_visible_surface')
        if self.previous is not None:
            elapsed = float(timestamp - self.previous_time)
            if 0 < elapsed <= cfg['max_motion_gap_s']:
                old_center = self.previous[:3, :3] @ self.center + self.previous[:3, 3]
                new_center = pose[:3, :3] @ self.center + pose[:3, 3]
                distance = float(np.linalg.norm(new_center - old_center) * 1000)
                relative_rotation = pose[:3, :3] @ self.previous[:3, :3].T
                angle = float(np.rad2deg(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
                result.update(motion_checked=True, translation_jump_mm=distance, rotation_jump_deg=angle, elapsed_s=elapsed)
                if distance > cfg['translation_slack_mm'] + cfg['max_translation_speed_mm_s'] * elapsed:
                    result['reasons'].append('translation_jump')
                if angle > cfg['rotation_slack_deg'] + cfg['max_rotation_speed_deg_s'] * elapsed:
                    result['reasons'].append('rotation_jump')
        result['ok'] = not result['reasons']
        if result['ok']:
            self.previous = pose.copy()
            self.previous_time = float(timestamp)
        return result
