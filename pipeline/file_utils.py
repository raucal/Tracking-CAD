"""Shared deterministic hashes and atomic UTF-8 JSON output."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    """Hash file bytes, or a relocation-independent inventory of directory contents."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for file in sorted(p for p in path.rglob('*') if p.is_file()):
            relative = file.relative_to(path).as_posix().encode('utf-8')
            digest.update(len(relative).to_bytes(8,'big'))
            digest.update(relative)
            digest.update(bytes.fromhex(sha256(file)))
    else:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace JSON only after complete, finite-value serialization."""
    path = Path(path)
    payload = json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(payload,encoding='utf-8')
    temporary.replace(path)
