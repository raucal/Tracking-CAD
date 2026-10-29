"""Remote browsing must select server inputs and respect the configured directory."""
import base64
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam6d_ui.server_files import ServerFiles


class ServerFilesTests(unittest.TestCase):
    def test_browse_filters_formats_and_keeps_directories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'subfolder').mkdir()
            for name in ('photo.png', 'model.stl', 'camera.json', '.hidden.png'):
                (root/name).write_bytes(b'content')
            browser = ServerFiles(root)
            self.assertEqual([x['name'] for x in browser.browse('', 'rgb')['entries']], ['subfolder', 'photo.png'])
            self.assertIsNone(browser.browse('', 'cad')['parent'])
            self.assertEqual(browser.browse('subfolder', 'cad')['parent'], str(root))

    def test_traversal_and_symlinks_cannot_escape_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'allowed'
            root.mkdir()
            outside = Path(tmp)/'outside.json'
            outside.write_text('{}')
            (root/'link.json').symlink_to(outside)
            browser = ServerFiles(root)
            for value in ('../outside.json', str(outside), 'link.json'):
                with self.assertRaises(ValueError):
                    browser.resolve(value)
            self.assertEqual(browser.browse('', 'camera')['entries'], [])

    def test_server_request_reads_files_without_client_upload(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'photo.png').write_bytes(b'RGB bytes')
            (root/'mesh.stl').write_bytes(b'CAD bytes')
            browser=ServerFiles(root)
            files, paths=browser.read_request({'rgb':'photo.png','cad':'mesh.stl'},'rgb')
            self.assertEqual(base64.b64decode(files['rgb']['data']),b'RGB bytes')
            self.assertEqual(paths['cad'],str(root/'mesh.stl'))
            with self.assertRaisesRegex(ValueError,'depth'):
                browser.read_request({'rgb':'photo.png','cad':'mesh.stl'},'rgbd')


if __name__ == '__main__':
    unittest.main()
