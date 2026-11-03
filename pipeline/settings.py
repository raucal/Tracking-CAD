"""Configuration and paths, independent of the checkout name and current directory."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config'
PIPELINE = ROOT / 'pipeline'

def read_json(path):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value

def relative_path(value, config_file):
    path = Path(os.path.expandvars(str(value))).expanduser()
    # Keep the venv executable path: dereferencing its symlink selects the base Python.
    joined = Path(config_file).resolve().parent / path if not path.is_absolute() else path
    return Path(os.path.abspath(joined))

SETTINGS_FILE = Path(os.environ.get('CAD_TRACKING_SETTINGS', CONFIG / 'settings.json')).expanduser().resolve()
SETTINGS = read_json(SETTINGS_FILE)

def path(key):
    try:
        value = SETTINGS['paths'][key]
    except KeyError as error:
        raise ValueError(f'Missing path {key!r} in {SETTINGS_FILE}') from error
    return relative_path(value, SETTINGS_FILE)

def profile(kind, filename=None):
    default_file = CONFIG / f'{kind}.json'
    source = Path(filename or os.environ.get('CAD_TRACKING_' + kind.upper() + '_CONFIG', default_file)).expanduser().resolve()
    defaults = read_json(default_file)
    value = read_json(source) if source != default_file.resolve() else defaults
    # Only explicitly provided paths use the custom profile's directory.
    paths = {'interface': {'server': ('data_dir', 'capture_dir', 'file_root')},
             'suite': {None: ('manifest', 'output_root', 'export_root', 'reference_cache')}}[kind]
    for section, keys in paths.items():
        base = defaults if section is None else defaults[section]
        selected = value if section is None else value.get(section, {})
        for key in keys:
            origin = source if key in selected else default_file
            target = selected[key] if key in selected else base[key]
            base[key] = relative_path(target, origin)
    if kind == 'interface':
        for section, fields in value.items():
            if section not in defaults or not isinstance(fields, dict):
                raise ValueError(f'{source}: unknown/invalid section {section}')
            unknown = set(fields) - set(defaults[section])
            if unknown: raise ValueError(f'{source}: unknown keys {sorted(unknown)}')
            for key, item in fields.items():
                if key not in paths.get(section, ()):
                    if isinstance(defaults[section][key], dict):
                        if not isinstance(item, dict): raise ValueError(f'{source}: invalid {section}.{key}')
                        unknown = set(item) - set(defaults[section][key])
                        if unknown: raise ValueError(f'{source}: unknown keys {sorted(unknown)}')
                        defaults[section][key].update(item)
                    else: defaults[section][key] = item
    else:
        unknown = set(value) - set(defaults)
        if unknown: raise ValueError(f'{source}: unknown keys {sorted(unknown)}')
        for key, item in value.items():
            if key not in paths[None]: defaults[key] = item
    if kind == 'suite':
        for key, choices in {'device': ('auto','cuda','cpu'), 'segmentor': ('sam','fastsam'), 'backbone': ('vit_h','vit_l','vit_b')}.items():
            if defaults[key] not in choices: raise ValueError(f'{source}: invalid {key}')
        if not isinstance(defaults['pose'], bool): raise ValueError(f'{source}: pose must be boolean')
    else:
        server, camera = defaults['server'], defaults['camera']
        if not isinstance(server['port'], int) or not 1 <= server['port'] <= 65535: raise ValueError(f'{source}: invalid port')
        if server['preload'] not in ('sam','fastsam','none') or server['backbone'] not in ('vit_h','vit_l','vit_b'): raise ValueError(f'{source}: invalid preload/backbone')
        if (camera['width'],camera['height']) not in ((640,480),(1280,720)) or camera['fps'] not in (15,30): raise ValueError(f'{source}: unsupported camera resolution/FPS')
        if not all(isinstance(camera[k],int) and camera[k]>0 for k in ('preview_width','preview_height','jpeg_quality')) or camera['jpeg_quality']>100: raise ValueError(f'{source}: invalid preview settings')
        interval=defaults['tracking']['redetect_interval_s']
        import math
        if isinstance(interval,bool) or not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval!=0 and not 1<=interval<=3600: raise ValueError(f'{source}: invalid redetection interval')
        tracking = defaults['tracking']; quality = tracking['quality']
        for key in ('recovery_retry_s','max_detection_age_s'):
            value=tracking[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0: raise ValueError(f'{source}: invalid {key}')
        if type(tracking['quality_bad_frames']) is not int or not 1<=tracking['quality_bad_frames']<=30: raise ValueError(f'{source}: invalid quality_bad_frames')
        if type(quality['enabled']) is not bool: raise ValueError(f'{source}: invalid quality.enabled')
        for key in ('sample_points','min_depth_samples'):
            if type(quality[key]) is not int or quality[key]<1: raise ValueError(f'{source}: invalid quality.{key}')
        if not 32<=quality['sample_points']<=10000 or quality['min_depth_samples']>quality['sample_points']: raise ValueError(f'{source}: invalid quality sample limits')
        for key,value in quality.items():
            if key in ('enabled','sample_points','min_depth_samples'): continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0: raise ValueError(f'{source}: invalid quality.{key}')
            if key in ('min_depth_support','min_valid_depth_fraction','min_visible_fraction','max_occlusion_fraction') and value>1: raise ValueError(f'{source}: invalid quality.{key}')

    return defaults

SAM6D = path('sam6d')
CORE = SAM6D / 'SAM-6D'
ISM = CORE / 'Instance_Segmentation_Model'
PEM = CORE / 'Pose_Estimation_Model'
PREPARED = path('prepared')
RESULTS = path('results')
NATIVE = path('native')


def dataset_path():
    return path('dataset')


def display_names():
    return read_json(dataset_path()).get('display_names', {})


def run_dataset(run, source):
    """Historical manifest bytes stay with the run; shared assets stay in prepared/."""
    snapshot = Path(source) / 'dataset_manifest.json'
    if snapshot.is_file():
        recorded = Path(run['manifest'])
        return snapshot, recorded.parent if recorded.is_file() else PREPARED
    recorded = Path(run['manifest'])
    if recorded.is_file(): return recorded, recorded.parent
    return PREPARED / 'manifest.json', PREPARED
