"""Assembly metadata layouts and capture isolation, without model loading."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_config import load_registry, select_assembly
from tools.prepare import sha256


class AssemblyConfigTests(unittest.TestCase):
    def test_directory_inventories_select_cad_and_isolate_captures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cad = root / 'PDDL/Assembly CAD'
            for key in ('printed_assembly', 'printed_assembly_2'):
                folder = cad / 'Assembly parts' / key
                folder.mkdir(parents=True)
                (folder / 'assembly.json').write_text(json.dumps({
                    'assembly_id': key, 'label': key, 'parts': {},
                }))
            dataset = root / 'CADTracking/data/dataset.json'
            config = {'assembly_registry': '../../PDDL/Assembly CAD',
                      'photos_dir': 'original captures',
                      'photo_aliases': {'old': 'alias'}, 'display_names': {'old': 'label'}}
            default, _ = select_assembly(dataset, config, 'printed_assembly')
            self.assertEqual(default['photos_dir'], 'original captures')
            selected, _ = select_assembly(dataset, config, 'printed_assembly_2')
            self.assertEqual((dataset.parent / selected['cad_dir']).resolve(),
                             cad / 'Assembly parts/printed_assembly_2')
            self.assertEqual((dataset.parent / selected['photos_dir']).resolve(),
                             root / 'CADTracking/captures/printed_assembly_2')
            self.assertEqual(selected['photo_aliases'], {})
            self.assertEqual(selected['display_names'], {})
            with self.assertRaisesRegex(ValueError, 'Unknown assembly'):
                select_assembly(dataset, config, '../../wrong')

    def test_legacy_registry_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / 'assemblies.json'
            value = {'default': 'demo', 'assemblies': {'demo': {
                'assembly_id': 'demo', 'cad_dir': 'parts', 'parts': {}}}}
            registry.write_text(json.dumps(value))
            self.assertEqual(load_registry(registry), value)
            config, _ = select_assembly(Path(temporary) / 'dataset.json',
                                       {'assembly_registry': 'assemblies.json'}, 'demo')
            self.assertEqual(config['cad_dir'], 'parts')

    def test_sha256_accepts_directory_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'assembly_registry'
            directory.mkdir()
            digest = sha256(directory)
            self.assertTrue(isinstance(digest, str))
            self.assertEqual(len(digest), 64)


if __name__ == '__main__':
    unittest.main()
