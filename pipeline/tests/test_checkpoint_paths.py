"""Configured physical weights must work without checkpoint aliases or downloads."""
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
from tools import download_runtime_assets as assets
from foundationpose_sdu import configure_predictor_weights
import settings


def test_foundationpose_ui_checks_the_configured_weights_directory(tmp_path, monkeypatch):
    from sam6d_ui.backends import foundationpose
    paths = {key: tmp_path / key for key in ('foundationpose', 'foundationpose_weights', 'foundationpose_python')}
    paths['foundationpose'].mkdir()
    paths['foundationpose_python'].touch()
    for stamp in ('2024-01-11-20-02-45', '2023-10-28-18-33-37'):
        folder = paths['foundationpose_weights'] / stamp
        folder.mkdir(parents=True)
        for name in ('config.yml', 'model_best.pth'):(folder / name).touch()
    monkeypatch.delenv('FOUNDATIONPOSE_ROOT', raising=False)
    monkeypatch.delenv('FOUNDATIONPOSE_PYTHON', raising=False)
    monkeypatch.setattr(foundationpose, 'path', paths.__getitem__)
    status = foundationpose.configured()
    assert status['available'] and not status['verified']
    assert status['reason'] is None
    assert not (paths['foundationpose'] / 'weights').exists()
    missing = paths['foundationpose_weights'] / '2024-01-11-20-02-45/model_best.pth'
    missing.unlink()
    status = foundationpose.configured()
    assert not status['available']
    assert status['missing_weight_files'] == [str(missing)]
    assert str(missing) in status['reason']


def test_installer_reuses_configured_external_weights_without_links(tmp_path, monkeypatch):
    target = tmp_path/'shared'/'custom-name.pth'
    target.parent.mkdir()
    target.write_bytes(b'checkpoint bytes')
    monkeypatch.setattr(assets, 'checkpoint', lambda *args: target)
    monkeypatch.setattr(sys, 'argv', ['download_runtime_assets.py'])
    download = Mock(side_effect=AssertionError('Existing weights must not be downloaded'))
    monkeypatch.setattr(assets.urllib.request, 'urlopen', download)
    assets.main()
    download.assert_not_called()
    assert list(tmp_path.rglob('*')) == [target.parent, target]
    assert not target.is_symlink()


def test_custom_model_paths_use_settings_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'SETTINGS_FILE', tmp_path/'config'/'settings.json')
    monkeypatch.setattr(settings, 'SETTINGS', {'paths': {'dinov2_checkpoint': '../../shared/encoder.pth'}})
    assert settings.checkpoint('dinov2') == tmp_path.parent/'shared'/'encoder.pth'


class FakePredictor:
    def __init__(self, amp=True):
        self.run_name = 'refiner'
        code_dir = '/upstream/source'
        self.checkpoint = f'{code_dir}/../../weights/{self.run_name}/model_best.pth'
        self.config = f'{code_dir}/../../weights/{self.run_name}/config.yml'
        self.amp = amp


def test_foundationpose_paths_can_be_reconfigured_without_changing_source(tmp_path, monkeypatch):
    class Predictor(FakePredictor):
        pass
    original = Predictor.__init__
    configure_predictor_weights(Predictor, tmp_path/'first')
    assert Predictor(False).amp is False
    assert Predictor().checkpoint == str(tmp_path/'first/refiner/model_best.pth')
    configure_predictor_weights(Predictor, tmp_path/'second')
    assert Predictor().config == str(tmp_path/'second/refiner/config.yml')
    assert FakePredictor.__init__ is original


def test_unknown_foundationpose_loader_is_rejected():
    class Unsupported:
        def __init__(self):
            pass
    with pytest.raises(RuntimeError, match='differs'):
        configure_predictor_weights(Unsupported, '/unused')
