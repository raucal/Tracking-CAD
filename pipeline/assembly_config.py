"""Load CAD inventories and select an assembly; paths stay config-relative."""
import json
import os
from pathlib import Path


def load_registry(registry_path):
    """Support per-assembly inventories as well as legacy registry JSON files."""
    registry_path = Path(registry_path).resolve()
    if not registry_path.is_dir():
        return json.loads(registry_path.read_text(encoding='utf-8'))
    assemblies = {}
    for inventory in sorted((registry_path / 'Assembly parts').glob('*/assembly.json')):
        spec = json.loads(inventory.read_text(encoding='utf-8'))
        key = inventory.parent.name
        spec['cad_dir'] = inventory.parent.relative_to(registry_path).as_posix()
        spec.setdefault('photos_dir', f'../../CADTracking/captures/{key}')
        assemblies[key] = spec
    if not assemblies:
        raise ValueError(f'No assembly inventories found in {registry_path / "Assembly parts"}')
    return {'default': 'printed_assembly', 'assemblies': assemblies}


def registry_root(registry_path):
    registry_path = Path(registry_path).resolve()
    return registry_path if registry_path.is_dir() else registry_path.parent


def select_assembly(config_path, config, name=None):
    config_path = Path(config_path).resolve()
    selected = name or config.get('assembly')
    if not selected:
        return config, None
    if not config.get('assembly_registry'):
        raise ValueError('assembly requires assembly_registry in the dataset configuration')
    registry_path = (config_path.parent / config['assembly_registry']).resolve()
    registry = load_registry(registry_path)
    root = registry_root(registry_path)
    if selected not in registry['assemblies']:
        raise ValueError(f'Unknown assembly {selected!r}; choose from {list(registry["assemblies"])}')
    spec = registry['assemblies'][selected]
    config = dict(config, assembly=selected, assembly_id=spec['assembly_id'])
    config['cad_dir'] = os.path.relpath(root / spec['cad_dir'], config_path.parent)
    if selected != registry['default']:
        # Never label the first assembly's captures as observations of another CAD.
        photos = root / spec['photos_dir']
        config['photos_dir'] = os.path.relpath(photos, config_path.parent)
        config['photo_aliases'] = {}
        config['display_names'] = {}
    return config, spec
