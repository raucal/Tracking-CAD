"""Shared weights must remain reusable and protected from conflicting links."""
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
from tools import download_runtime_assets as assets


@pytest.mark.parametrize("conflicting", [False, True])
def test_installer_resolves_shared_cache_aliases(tmp_path, monkeypatch, conflicting):
    shared=tmp_path/'shared'; shared.mkdir()
    physical=shared/'descriptor.pth';physical.write_bytes(b'checkpoint bytes')
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'descriptor.pth').symlink_to(physical)
    ism=tmp_path/'model';link=ism/'checkpoints/dinov2/descriptor.pth'
    link.parent.mkdir(parents=True)
    destination=cache/'descriptor.pth'
    if conflicting:
        destination=shared/'other.pth';destination.write_bytes(b'other weights')
    link.symlink_to(destination)
    monkeypatch.setattr(assets,'ISM',ism)
    monkeypatch.setattr(assets,'ASSETS',[("descriptor.pth","https://unused.invalid/weights","dinov2")])
    monkeypatch.setattr(sys,'argv',['download_runtime_assets.py','--cache',str(cache)])
    download=Mock(side_effect=AssertionError('Existing shared weights must not be downloaded'))
    monkeypatch.setattr(assets.urllib.request,'urlopen',download)
    if conflicting:
        with pytest.raises(ValueError,match='already points elsewhere'):assets.main()
    else:
        assets.main()
        assert link.resolve()==physical
        assert link.read_bytes()==b'checkpoint bytes'
    download.assert_not_called()
