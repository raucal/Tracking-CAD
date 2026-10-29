"""Run or export an all-CAD SAM-6D benchmark without requiring a display."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from sam6d_suite_report import export_report

PROJECTS = Path(__file__).resolve().parents[2]
WORKER = PROJECTS / "SAM6D/SAM-6D/SAM-6D/Instance_Segmentation_Model/run_suite_pddl.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "prepared/manifest.json")
    parser.add_argument("--output", type=Path, help="Run directory; created with a timestamp if omitted")
    parser.add_argument("--export-dir", type=Path, help="Offline HTML, CSV, images and ZIP destination")
    parser.add_argument("--frame", action="append", help="Capture name; repeat to select several; default: all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--stability-score", type=float, default=0.97)
    parser.add_argument("--segmentor", choices=("sam", "fastsam"), default="sam")
    parser.add_argument("--backbone", choices=("vit_h", "vit_l", "vit_b"), default="vit_h")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reference-cache", type=Path, default=Path("/dev/shm/pddl-assembly-runtime/reference_features"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    if args.command == "report" and args.output is None:
        parser.error("report requiere --output con una ejecución guardada.")
    manifest = args.manifest.resolve()
    dataset = json.loads(manifest.read_text())
    if dataset.get("schema_version") != 1:
        parser.error("Manifest incompatible.")
    known = {f["name"] for f in dataset["frames"]}
    unknown = set(args.frame or ()) - known
    if unknown:
        parser.error(f"Capturas desconocidas: {sorted(unknown)}")
    if not (0 <= args.min_score <= args.confidence <= 1) or args.threads < 1:
        parser.error("Se requiere 0 <= min-score <= confidence <= 1 y threads >= 1.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or manifest.parent / "sam6d/suites" / stamp).resolve()
    exported = (args.export_dir or PROJECTS / "SAM6D/results" / f"sam6d_suite_{output.name}").resolve()
    returncode = 0
    if args.command == "run":
        command = [sys.executable, str(WORKER), "--manifest", str(manifest), "--output", str(output),
                   "--device", args.device, "--min-score", str(args.min_score), "--confidence", str(args.confidence),
                   "--stability-score", str(args.stability_score), "--segmentor", args.segmentor, "--backbone", args.backbone, "--threads", str(args.threads),
                   "--reference-cache", str(args.reference_cache.resolve())]
        for frame in args.frame or ():
            command.extend(("--frame", frame))
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        if output.exists() and any(output.iterdir()):
            parser.error("La salida ya contiene archivos; usa una carpeta nueva.")
        output.parent.mkdir(parents=True, exist_ok=True)
        logfile = output.with_name(output.name + ".log")
        with logfile.open("w") as log:
            with subprocess.Popen(command, cwd=WORKER.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        if output.is_dir():
            shutil.copy2(logfile, output / "execution.log")
    if not args.no_export and (output / "run.json").is_file():
        bundle = export_report(output, exported)
        print(f"Informe: {exported / 'index.html'}\nZIP: {bundle}", flush=True)
    elif args.command == "report":
        parser.error(f"Falta run.json en {output}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
