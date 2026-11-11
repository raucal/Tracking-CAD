"""Small deterministic RGB-D/CAD fixtures; no sibling project or hardware required."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh
import yaml


@pytest.fixture(scope='module')
def source_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic source data')
    cads = root / 'cads'; photos = root / 'photos'
    cads.mkdir(); photos.mkdir()
    definitions = [('00_rear_mounting_plate','rear_mounting'),('01_front_housing','front_housing'),
                   ('02_control_panel','hmi_panel'),('03_option_cassette','option_cassette'),
                   ('03a_vertical_option_a','sensor_module'),('03b_vertical_option_b','communications_module'),
                   ('04_io_terminals','io_terminals'),('05_control_board','control_board'),
                   ('06_power_unit','power_unit'),('07_thermal_core','thermal_core')]
    catalog = {'parts':{},'fixed_parts':{}}
    for i,(key,name) in enumerate(definitions):
        mesh = trimesh.creation.box(extents=[10+i,12+i,14+i])
        if i == 0:
            other = mesh.copy(); other.apply_translation([10,12,0])
            mesh = trimesh.util.concatenate([mesh,other]); mesh.merge_vertices()
            assert not mesh.is_watertight
        mesh.export(cads/(key+'.stl'))
        catalog['fixed_parts' if i==0 else 'parts'][name]={'cad_stem':key,'optional':False}
    (root/'catalog.yaml').write_text(yaml.safe_dump(catalog),encoding='utf-8')
    names = ['control_board','mounted_front_housing','termal_fan','unlabelled_capture'] + [f'capture_{i:02d}' for i in range(4,22)]
    for i,name in enumerate(names):
        Image.new('RGB',(64,48),(20+i,60+i,120+i)).save(photos/(name+'.jpg'))
        depth = np.full((48,64),500+i,dtype=np.uint16); depth[0,0]=0
        np.save(photos/(name+'_depth.npy'),depth)
        metadata={'aligned_to':'color','height':48,'width':64,'dtype':'uint16','invalid_depth_value':0,
                  'depth_scale_m_per_unit':.001,'color_image':name+'.jpg','depth_image':name+'_depth.npy',
                  'intrinsics':{'fx':100.,'fy':101.,'ppx':32.,'ppy':24.,'coefficients':[0.,0.,0.,0.,0.]}}
        (photos/(name+'_depth.json')).write_text(json.dumps(metadata),encoding='utf-8')
    Image.new('RGB',(64,48)).save(photos/'front_housing.jpg')
    config={'assembly_id':'synthetic_test_assembly','cad_dir':'cads','photos_dir':'photos','catalog':'catalog.yaml',
            'photo_aliases':{'termal_fan':'thermal_core'},'display_names':{}}
    path=root/'dataset.json';path.write_text(json.dumps(config),encoding='utf-8')
    return path


@pytest.fixture
def app_dataset(tmp_path, monkeypatch):
    """HTTP behaviour uses an explicit empty catalog, never local example data."""
    from sam6d_ui import server
    (tmp_path/'manifest.json').write_text(
        json.dumps({'schema_version':1,'parts':[],'frames':[]}), encoding='utf-8')
    monkeypatch.setattr(server, 'PREPARED', tmp_path)
