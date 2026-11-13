"""Read-only access to recorded DINOBOT skill demos and their saved CAD poses.

A demo contributes two measurements taken at the same instant: the TCP pose in
robot base (``trajectory_meta.json`` -> ``reference_start.pose``) and the
FoundationPose CAD pose in the camera frame of the reference image
(``sam6d_foundationpose_*/cad_pose_camera.json``). Both refer to the same
``btn_rgb.png``; that link is checked through file hashes, so a demo that was
re-recorded after its pose was estimated is rejected instead of silently used.
Nothing here opens hardware or writes files.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from .geometry import rigid

DEMO_REFERENCE = (Path(__file__).resolve().parents[2]
                  / 'PDDL/DINOBOT/alignment_and_replay-main/demo_reference.py')
POSE_FILE = 'cad_pose_camera.json'
PROVENANCE_FILE = 'source_provenance.json'
REFERENCE_IMAGE = ('btn_rgb.png', 'btn_depth.npy')
_reference = None


def _demo_reference():
    """DINOBOT's own reference_start validation, loaded lazily (one source of truth)."""
    global _reference
    if _reference is None:
        spec = importlib.util.spec_from_file_location('_cadtracking_demo_reference', DEMO_REFERENCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _reference = module
    return _reference


def digest(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def _bindings(bindings_file):
    bindings_file = Path(bindings_file)
    data = yaml.safe_load(bindings_file.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('bindings'), dict) or not data.get('demos_root'):
        raise ValueError('Skill bindings YAML requires demos_root and bindings.')
    root = Path(data['demos_root'])
    root = (root if root.is_absolute() else bindings_file.resolve().parent / root).resolve()
    skills = {}
    for assembly, actions in data['bindings'].items():
        for action, parts in (actions or {}).items():
            for part, binding in (parts or {}).items():
                demo_id = (binding or {}).get('demo_id')
                if not demo_id:
                    continue
                if not isinstance(demo_id, str) or Path(demo_id).name != demo_id or demo_id in ('.', '..'):
                    raise ValueError('Invalid demo_id in skill bindings: ' + repr(demo_id))
                skills.setdefault(demo_id, []).append({'assembly_id': assembly, 'action': action, 'part_id': part,
                                                       'skill_id': binding.get('skill_id')})
    return root, skills


def _pose_candidates(folder):
    return sorted((p for p in folder.glob('sam6d_foundationpose_*/' + POSE_FILE)), key=lambda p: p.parent.name,
                  reverse=True)


def catalog(bindings_file):
    """Cheap listing for the UI: bound demos and the CAD each saved pose refers to."""
    root, skills = _bindings(bindings_file)
    entries = []
    for demo_id, bound in sorted(skills.items()):
        folder = root / demo_id
        entry = {'demo_id': demo_id, 'skills': bound, 'available': False, 'cad_key': None, 'cad_name': None}
        candidates = _pose_candidates(folder) if folder.is_dir() else []
        if not folder.is_dir():
            entry['problem'] = 'Demo folder is missing.'
        elif not candidates:
            entry['problem'] = 'No saved FoundationPose pose; run pipeline/tools/process_saved_demos.py.'
        else:
            try:
                pose = json.loads(candidates[0].read_text())
                entry.update(cad_key=pose.get('cad_key'), cad_name=pose.get('cad_name'),
                             available=pose.get('status') == 'estimated', pose_folder=candidates[0].parent.name)
                if not entry['available']:
                    entry['problem'] = 'The saved pose has no accepted detection.'
            except (OSError, ValueError) as exc:
                entry['problem'] = 'Unreadable pose: ' + str(exc)
        entries.append(entry)
    return entries


def load(bindings_file, demo_id):
    """Fully validated demo reference used to compute a target; fails closed."""
    root, skills = _bindings(bindings_file)
    if demo_id not in skills:
        raise ValueError('The demo is not bound to any skill: ' + str(demo_id))
    folder = root / demo_id
    if not folder.is_dir():
        raise ValueError('Demo folder is missing: ' + str(folder))
    metadata = json.loads((folder / 'trajectory_meta.json').read_text())
    start = _demo_reference().read_reference_start(folder, metadata)
    if start is None:
        raise ValueError(f'{demo_id}: legacy demo without reference_start; the TCP pose of its image is unknown.')
    rejected = []
    for pose_file in _pose_candidates(folder):
        provenance = json.loads((pose_file.parent / PROVENANCE_FILE).read_text())
        recorded = provenance.get('original_sha256', {})
        # The saved pose must come from this exact reference image, which is the
        # one whose TCP pose reference_start recorded.
        stale = [name for name in REFERENCE_IMAGE
                 if recorded.get(name) != digest(folder / name) or recorded.get(name) != start['file_hashes'].get(name)]
        if stale:
            rejected.append(pose_file.parent.name)
            continue
        pose = json.loads(pose_file.read_text())
        if pose.get('status') != 'estimated' or pose.get('demo') != demo_id:
            raise ValueError(f'{demo_id}: the saved pose has no accepted detection.')
        camera_from_cad = rigid(pose['T_camera_from_input_cad_mm'])
        if camera_from_cad[2, 3] <= 0:
            raise ValueError(f'{demo_id}: the saved CAD pose is behind the camera.')
        return {'demo_id': demo_id, 'skills': skills[demo_id], 'folder': str(folder),
                'pose_file': str(pose_file), 'pose_folder': pose_file.parent.name,
                'cad_key': pose['cad_key'], 'cad_name': pose.get('cad_name'),
                'cad_sha256': provenance.get('cad_sha256'),
                'T_camera_from_input_cad_mm': camera_from_cad.tolist(),
                'sam6d_score': pose.get('sam6d_score'), 'foundationpose_score': pose.get('foundationpose_score'),
                'camera': pose.get('camera'),
                'tcp_pose_m_rad': list(start['pose']), 'tcp_offset_m_rad': list(start['geometry']['tcp_offset']),
                'robot_id': start['geometry']['robot_id'],
                'file_hashes': fingerprint(folder, pose_file)}
    if rejected:
        raise ValueError(f'{demo_id}: the saved pose ({", ".join(rejected)}) was estimated from a different '
                         'reference image than the current recording. Re-run process_saved_demos.py for this demo.')
    raise ValueError(f'{demo_id}: no saved FoundationPose pose; run pipeline/tools/process_saved_demos.py.')


def fingerprint(folder, pose_file):
    folder = Path(folder)
    files = [folder / 'trajectory_meta.json', *(folder / n for n in REFERENCE_IMAGE), Path(pose_file)]
    return {str(f): digest(f) for f in files}


def verify(demo):
    """Recheck before motion that the demo used for the target has not changed."""
    for name, value in demo['file_hashes'].items():
        if not Path(name).is_file() or digest(name) != value:
            raise ValueError('The skill demo changed after the target was calculated; detect the pose again.')


def select(bindings_file, cad_key, demo_id=None):
    """Explicit demo, or the single bound demo whose saved pose shows ``cad_key``."""
    if demo_id:
        return load(bindings_file, demo_id)
    matches = [e['demo_id'] for e in catalog(bindings_file) if e['cad_key'] == cad_key]
    if not matches:
        raise ValueError(f'No skill demo with a saved pose of {cad_key}.')
    if len(matches) > 1:
        raise ValueError(f'Several skill demos show {cad_key} ({", ".join(matches)}); select one explicitly.')
    return load(bindings_file, matches[0])
