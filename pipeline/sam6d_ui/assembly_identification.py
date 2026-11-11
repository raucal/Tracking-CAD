"""Resolve a complete prepared assembly catalog before scheduling inference."""
import json
from pathlib import Path

from assembly_config import load_registry
from settings import PREPARED, dataset_path


def assembly_catalog(name):
    dataset = dataset_path()
    config = json.loads(dataset.read_text())
    registry = load_registry(dataset.parent / config['assembly_registry'])
    if name not in registry['assemblies']:
        raise ValueError('Unknown assembly: ' + str(name))
    manifest = (PREPARED if name == registry['default'] else
                PREPARED.parent / 'assemblies' / name / 'prepared') / 'manifest.json'
    if not manifest.is_file():
        raise ValueError(f'Missing templates for {name}. Prepare its CADs and render templates before automatic identification.')
    data = json.loads(manifest.read_text())
    spec = registry['assemblies'][name]
    if data.get('assembly_id') != spec['assembly_id'] or (
            data.get('assembly_name') is not None and data['assembly_name'] != name):
        raise ValueError('The manifest does not belong to the selected assembly.')
    parts = data['parts']
    if len(parts) != len(spec['parts']) or {p['key'] for p in parts} != set(spec['parts']):
        raise ValueError('The prepared catalog does not contain every assembly part.')
    for part in parts:
        if 'T_local_from_sam_mm' not in part:
            raise ValueError(f'Missing prepared CAD transformation for {part["key"]}. Prepare the assembly again.')
        required = [manifest.parent / part['sam_ply'], manifest.parent / part['source_stl']]
        required.extend(manifest.parent / part['sam_template_dir'] / f'{prefix}_{i}.png'
                        for i in range(42) for prefix in ('rgb', 'mask'))
        if not all(p.is_file() for p in required):
            raise ValueError(f'Missing meshes or templates for {part["key"]}. Rebuild the assembly templates.')
    return manifest, parts
