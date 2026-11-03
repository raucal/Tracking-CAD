"""SDU extraction and compatibility must preserve exact source and model frames."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import unittest

from settings import ROOT, path

HERE = ROOT / 'integrations/foundationpose_sdu'
spec = importlib.util.spec_from_file_location('sdu_runtime_test', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class FoundationPoseSDU(unittest.TestCase):
    def test_added_sdu_files_are_preserved_byte_for_byte(self):
        provenance = json.loads((HERE / 'provenance.json').read_text())
        for name, expected in provenance['added_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE / 'files' / name).read_bytes()).hexdigest(), expected)

    def test_utils_patch_reproduces_sdu_without_modifying_official_source(self):
        official = path('foundationpose')
        before = (official / 'Utils.py').read_bytes()
        provenance = json.loads((HERE / 'provenance.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(['git', 'init', '-q', temporary], check=True)
            generated = runtime.patched_utils(official, Path(temporary) / 'nested-cache')
            self.assertEqual(hashlib.sha256(generated.read_bytes()).hexdigest(), provenance['adapted_utils_sha256'])
            self.assertEqual(runtime.patched_utils(official, Path(temporary) / 'nested-cache'), generated)
            self.assertEqual((official / 'Utils.py').read_bytes(), before)

    def test_wrong_official_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'Utils.py').write_text('different source')
            with self.assertRaisesRegex(RuntimeError, 'revision differs'):
                runtime.patched_utils(root, root / 'cache')


if __name__ == '__main__':
    unittest.main()
