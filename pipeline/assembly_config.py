"""Select an assembly from the shared CAD registry; paths stay config-relative."""
import json
import os
from pathlib import Path


def select_assembly(config_path, config, name=None):
    config_path = Path(config_path).resolve()
    selected = name or config.get('assembly')
    if not selected:
        return config, None
    if not config.get('assembly_registry'):
        raise ValueError('assembly requires assembly_registry in the dataset configuration')
    registry_path = (config_path.parent / config['assembly_registry']).resolve()
    registry = json.loads(registry_path.read_text())
    if selected not in registry['assemblies']:
        raise ValueError(f'Unknown assembly {selected!r}; choose from {list(registry["assemblies"])}')
    spec = registry['assemblies'][selected]
    config = dict(config, assembly=selected, assembly_id=spec['assembly_id'])
    config['cad_dir'] = os.path.relpath(registry_path.parent / spec['cad_dir'], config_path.parent)
    if selected != registry['default']:
        # Never label the first assembly's captures as observations of another CAD.
        photos = registry_path.parent / spec['photos_dir']
        config['photos_dir'] = os.path.relpath(photos, config_path.parent)
        config['photo_aliases'] = {}
        config['display_names'] = {}
    return config, spec
