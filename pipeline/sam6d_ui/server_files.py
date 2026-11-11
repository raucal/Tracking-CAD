"""Browse and read inference inputs from explicit server directories."""
import base64
from pathlib import Path

FORMATS = {'rgb': ('.jpg', '.jpeg', '.png'), 'cad': ('.stl', '.obj', '.ply'),
           'depth': ('.npy', '.png'), 'camera': ('.json',)}
LIMITS = {'rgb': 20*1024**2, 'cad': 50*1024**2, 'depth': 20*1024**2, 'camera': 1024**2}


class ServerFiles:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('The file root must be a directory.')

    def resolve(self, path):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve(strict=True)
        try:
            relative = candidate.relative_to(self.root)
        except ValueError:
            raise ValueError('Select a file inside ' + str(self.root)) from None
        if any(part.startswith('.') for part in relative.parts):
            raise ValueError('Hidden files and directories are unavailable in this selector.')
        return candidate

    def browse(self, path, kind):
        if kind not in FORMATS:
            raise ValueError('Unknown file type.')
        folder = self.resolve(path or self.root)
        if not folder.is_dir():
            raise ValueError('Enter a server directory.')
        entries = []
        for item in folder.iterdir():
            if item.name.startswith('.'):
                continue
            try:
                resolved = self.resolve(item)
                directory = resolved.is_dir()
                if directory or (resolved.is_file() and item.suffix.lower() in FORMATS[kind]):
                    entries.append({'name': item.name, 'path': str(resolved), 'directory': directory})
            except (OSError, ValueError, RuntimeError):
                continue
        entries.sort(key=lambda entry: (not entry['directory'], entry['name'].casefold()))
        return {'root': str(self.root), 'path': str(folder),
                'parent': str(folder.parent) if folder != self.root else None,
                'entries': entries[:1000], 'truncated': len(entries) > 1000}

    def read_request(self, paths, mode):
        if not isinstance(paths, dict):
            raise ValueError('Server file paths are required.')
        kinds = ('rgb', 'cad', 'depth', 'camera') if mode == 'rgbd' else ('rgb', 'cad')
        request, sources = {}, {}
        for kind in kinds:
            if not isinstance(paths.get(kind), str) or not paths[kind].strip():
                raise ValueError('No server file selected: ' + kind)
            path = self.resolve(paths[kind].strip())
            if not path.is_file() or path.suffix.lower() not in FORMATS[kind]:
                raise ValueError('Unsupported file format for ' + kind)
            if path.stat().st_size > LIMITS[kind]:
                raise ValueError('File exceeds the maximum size for ' + kind)
            with path.open('rb') as source:
                content = source.read(LIMITS[kind] + 1)
            if len(content) > LIMITS[kind]:
                raise ValueError('File exceeds the maximum size for ' + kind)
            request[kind] = {'name': path.name, 'data': base64.b64encode(content).decode()}
            sources[kind] = str(path)
        return request, sources
