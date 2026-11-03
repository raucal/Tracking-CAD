"""Apply SDU's Utils compatibility in memory, keeping the official checkout clean."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent


def patched_utils(official, cache):
    official = Path(official).resolve()
    provenance = json.loads((HERE / 'provenance.json').read_text())
    original = official / 'Utils.py'
    if hashlib.sha256(original.read_bytes()).hexdigest() != provenance['original_utils_sha256']:
        raise RuntimeError('FoundationPose revision differs from the pinned SDU compatibility base')
    destination = Path(cache) / provenance['adapted_utils_sha256'] / 'Utils.py'
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == provenance['adapted_utils_sha256']:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        candidate = Path(temporary) / 'Utils.py'
        candidate.write_bytes(original.read_bytes())
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(destination.parent.resolve()))
        subprocess.run(['git', 'apply', '--include=Utils.py', str(HERE / 'sdu.patch')],
                       cwd=temporary, env=env, check=True, capture_output=True, text=True)
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != provenance['adapted_utils_sha256']:
            raise RuntimeError('SDU Utils patch did not reproduce the recorded source')
        candidate.replace(destination)
    return destination


def install(official, cache):
    if 'Utils' in sys.modules:
        if not getattr(sys.modules['Utils'], '_cadtracking_sdu', False):
            raise RuntimeError('Install SDU compatibility before importing FoundationPose Utils')
        return sys.modules['Utils']
    official = Path(official).resolve()
    patched = patched_utils(official, cache)
    sys.path.insert(0, str(official))
    module = types.ModuleType('Utils')
    # Resources and extension lookup keep the official source directory.
    module.__file__ = str(official / 'Utils.py')
    module.__package__ = ''
    module._cadtracking_sdu = True
    sys.modules['Utils'] = module
    try:
        exec(compile(patched.read_text(), module.__file__, 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop('Utils', None)
        raise
    return module
