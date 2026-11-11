"""Regression checks for dataset provenance and safe metadata writes."""
import json
import shutil
import pytest
from file_utils import sha256, write_json


def test_directory_digest_tracks_contents_and_names_but_not_location(tmp_path):
    source=tmp_path/'original';source.mkdir()
    (source/'nested').mkdir();(source/'nested'/'cad.stl').write_bytes(b'original geometry')
    moved=tmp_path/'relocated';shutil.copytree(source,moved)
    initial=sha256(source)
    assert sha256(moved)==initial
    (moved/'nested'/'cad.stl').write_bytes(b'changed geometry')
    assert sha256(moved)!=initial
    shutil.copyfile(source/'nested'/'cad.stl',moved/'nested'/'cad.stl')
    (moved/'nested'/'cad.stl').rename(moved/'nested'/'other.stl')
    assert sha256(moved)!=initial


def test_invalid_json_preserves_previous_metadata(tmp_path):
    target=tmp_path/'metadata.json'
    original={'label':'CAD \u03b1','score':0.75}
    write_json(target,original)
    with pytest.raises(ValueError):write_json(target,{'score':float('nan')})
    assert json.loads(target.read_text(encoding='utf-8'))==original
    assert not target.with_suffix('.json.tmp').exists()
