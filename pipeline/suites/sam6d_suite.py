"""Run or export an all-CAD SAM-6D benchmark without requiring a display."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from suites.sam6d_suite_report import export_report, publish_dashboard

from settings import ISM, PREPARED, RESULTS, profile, relative_path, dataset_path
WORKER = ISM / "run_suite_pddl.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report", "pose"))
    parser.add_argument('--config', type=Path, help='Suite JSON profile; CLI flags take precedence')
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path)
    selected, _ = selector.parse_known_args()
    config = profile('suite', selected.config)

    parser.add_argument("--assembly", default=config.get("assembly", "printed_assembly"))
    parser.add_argument("--manifest", type=Path, default=config['manifest'])
    parser.add_argument("--output", type=Path, help="Run directory; created with a timestamp if omitted")
    parser.add_argument("--export-dir", type=Path, help="Offline HTML, CSV, images and ZIP destination")
    parser.add_argument("--frame", action="append", help="Capture name; repeat to select several; default: all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=config['device'])
    parser.add_argument("--min-score", type=float, default=config['min_score'])
    parser.add_argument("--confidence", type=float, default=config['confidence'])
    parser.add_argument("--stability-score", type=float, default=config['stability_score'])
    parser.add_argument("--segmentor", choices=("sam", "fastsam"), default=config['segmentor'])
    parser.add_argument("--backbone", choices=("vit_h", "vit_l", "vit_b"), default=config['backbone'])
    parser.add_argument("--threads", type=int, default=config['threads'])
    parser.add_argument("--reference-cache", type=Path, default=config['reference_cache'])
    parser.add_argument("--pose",action=argparse.BooleanOptionalAction,default=config["pose"],help="Calculate PEM poses for accepted detections")
    parser.add_argument("--overwrite-pose",action="store_true",help="Recompute poses of a saved suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Crear también un ZIP del informe")
    args = parser.parse_args()
    dataset_config = dataset_path()
    data_config = json.loads(dataset_config.read_text())
    if data_config.get('assembly_registry'):
        from assembly_config import load_registry
        registry = load_registry(dataset_config.parent / data_config['assembly_registry'])
        if args.assembly not in registry['assemblies']:
            parser.error(f'Ensamblaje desconocido: {args.assembly}')
    if args.frame is None: args.frame = config["frame"]
    if not isinstance(args.frame, list) or not all(isinstance(f, str) for f in args.frame): parser.error("frame must be a list of names")
    if args.command in ("report","pose") and args.output is None:
        parser.error("report requiere --output con una ejecución guardada.")
    if (args.pose or args.command=="pose") and args.device=="cpu":parser.error("PEM requiere CUDA; usa --device cuda o auto")
    if args.assembly != "printed_assembly" and args.manifest == PREPARED / "manifest.json" and not any(a == "--manifest" or a.startswith("--manifest=") for a in sys.argv):
        args.manifest = PREPARED.parent / 'assemblies' / args.assembly / 'prepared/manifest.json'
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        parser.error(f'Falta el manifest de {args.assembly}: {manifest}. Ejecuta prepare --assembly {args.assembly} y prepara sus capturas y plantillas.')
    dataset = json.loads(manifest.read_text())
    recorded_assembly = dataset.get('assembly_name', 'printed_assembly' if dataset.get('assembly_id') == 'converter_demo_v1' else dataset.get('assembly_id'))
    if recorded_assembly != args.assembly:
        parser.error(f'El manifest pertenece a {recorded_assembly}, no a {args.assembly}')
    if dataset.get("schema_version") != 1:
        parser.error("Manifest incompatible.")
    if args.command == 'run' and not dataset.get('frames'):
        parser.error(f'{args.assembly} no tiene capturas RGB-D preparadas; añade sus propias capturas y vuelve a preparar los datos.')
    known = {f["name"] for f in dataset["frames"]}
    unknown = set(args.frame or ()) - known
    if unknown:
        parser.error(f"Capturas desconocidas: {sorted(unknown)}")
    if not (0 <= args.min_score <= args.confidence <= 1) or not 0<=args.stability_score<=1 or args.threads < 1:
        parser.error("Se requiere 0 <= min-score <= confidence <= 1 y threads >= 1.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = config['output_root'] if args.assembly == 'printed_assembly' else config['output_root'] / args.assembly
    output = (args.output or run_root / stamp).resolve()
    exported = (args.export_dir or config['export_root'] / f'sam6d_suite_{output.name}').resolve()
    returncode = 0
    if args.command == "run":
        command = [sys.executable, str(WORKER), "--manifest", str(manifest), "--output", str(output),
                   "--device", args.device, "--min-score", str(args.min_score), "--confidence", str(args.confidence),
                   "--stability-score", str(args.stability_score), "--segmentor", args.segmentor, "--backbone", args.backbone, "--threads", str(args.threads),
                   "--reference-cache", str(args.reference_cache.resolve())]
        if args.pose:command.append("--pose")
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
            shutil.copy2(manifest, output / "dataset_manifest.json")
    if args.command=="pose":
        from suites.sam6d_suite_pose import enrich_suite
        enrich_suite(output,args.overwrite_pose,args.frame)
    if not args.no_export and (output / "run.json").is_file():
        bundle = export_report(output, exported, create_archive=args.zip)
        dashboard = publish_dashboard(exported)
        print(f"Informe: {exported / 'index.html'}\nDashboard actualizado: {dashboard}", flush=True)
        if bundle: print(f"ZIP: {bundle}", flush=True)
    elif args.command == "report":
        parser.error(f"Falta run.json en {output}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
