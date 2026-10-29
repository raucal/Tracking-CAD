"""Download the official ISM checkpoints into a selected cache, then link them."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.request

PROJECTS = Path(__file__).resolve().parents[2]
ISM = PROJECTS / "SAM6D/SAM-6D/SAM-6D/Instance_Segmentation_Model"
ASSETS = [
    ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "segment-anything"),
    ("dinov2_vitl14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", "dinov2"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    report = []
    for name, url, group in ASSETS:
        target = cache / name
        if not target.exists():
            temporary = target.with_suffix(".download")
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as destination:
                size = response.headers.get("Content-Length")
                size = int(size) if size else None
                total = 0
                last = time.monotonic()
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    total += len(block)
                    if time.monotonic() - last > 10:
                        print(f"{name}: {total / 1e6:.0f} MB" + (f" / {size / 1e6:.0f} MB" if size else ""), flush=True)
                        last = time.monotonic()
            if size is not None and total != size:
                raise ValueError(f"Incomplete checkpoint: {name}")
            temporary.replace(target)
        digest = hashlib.sha256()
        with target.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
        link = ISM / "checkpoints" / group / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() != target:
            raise ValueError(f"Checkpoint link already points elsewhere: {link}")
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
        report.append({"name": name, "url": url, "cache": str(target),
                       "bytes": target.stat().st_size, "sha256": digest.hexdigest()})
        print(f"Ready: {name}, {target.stat().st_size / 1e6:.0f} MB", flush=True)
    output = PROJECTS / "SAM6D/pipeline/prepared/runtime_checkpoints.json"
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
