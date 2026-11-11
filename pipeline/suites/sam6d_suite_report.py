"""Portable report of measured SAM-6D runs; no model execution or inferred ground truth."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))

import csv
import copy
import hashlib
import html
from settings import PREPARED, display_names, run_dataset
import json
import math
from pathlib import Path
import shutil
import statistics
from urllib.parse import quote
import zipfile
from suites.ism_report_details import reference_gallery, methodology, view_audit_section


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_style(score, selected=False, threshold=0.4):
    if score is None:
        return "score-empty"
    if selected:
        return "score-winner"
    return "score-alternative" if score > threshold else "score-low"


def reference_preview(folder, view, target):
    from PIL import Image, ImageOps
    with Image.open(folder / f"rgb_{view}.png") as source, Image.open(folder / f"mask_{view}.png") as mask:
        box = mask.getbbox()
        if box:
            x1, y1, x2, y2 = box
            box = (max(0, x1 - 6), max(0, y1 - 6), min(source.width, x2 + 6), min(source.height, y2 + 6))
        picture = ImageOps.contain(source.convert("RGB").crop(box), (240, 200))
        canvas = Image.new("RGB", (250, 210), source.convert("RGB").getpixel((0, 0)))
        canvas.paste(picture, ((250 - picture.width) // 2, (210 - picture.height) // 2))
        canvas.save(target)


def decision_overlay(folder, rows, chosen):
    """Show one best saved mask per CAD; color is a display rule, not a new inference."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(folder / "rgb.png") as source:
        pixels = np.array(source.convert("RGB"))
    selected = [r for r in rows if r["confianza_final_sam6d"] is not None
                and (r["cad_predicho"] == chosen or r["confianza_final_sam6d"] > 0.4)]
    selected.sort(key=lambda r: r["cad_predicho"] == chosen)
    boxes = []
    for row in selected:
        with Image.open(folder / f'mask_{row["indice_deteccion"]:03d}.png') as source:
            mask = np.array(source) > 0
        color = (25, 160, 100) if row["cad_predicho"] == chosen else (240, 151, 30)
        pixels[mask] = (0.62 * pixels[mask] + 0.38 * np.array(color)).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs):
            boxes.append((row, color, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    label_boxes = []
    for row, color, box in boxes:
        draw.rectangle(box, outline=color, width=3)
        text = f'{row["cad_predicho"].split("_")[0]}  {row["confianza_final_sam6d"]:.3f}'
        x, y = box[0], max(0, box[1] - 15)
        bounds = draw.textbbox((x, y), text, font=font)
        if any(bounds[0] < old[2] and bounds[2] > old[0] and bounds[1] < old[3] and bounds[3] > old[1] for old in label_boxes):
            y = max(0, y - 16)
            bounds = draw.textbbox((x, y), text, font=font)
        label_boxes.append(bounds)
        draw.rectangle(bounds, fill=color)
        draw.text((x, y), text, fill="black", font=font)
    image.save(folder / "overlay_decision.png")


SCORE_FIELDS=['semantic_score','appearance_score','geometry_score','visible_ratio','geometry_contribution','score_reconstructed','ism_formula_matches']


def score_components(detection):
    values={key:detection.get(key) for key in SCORE_FIELDS[:4]}
    complete=all(isinstance(v,(int,float)) and math.isfinite(v) for v in values.values())
    geometric=values['geometry_score']*values['visible_ratio'] if complete else None
    reconstructed=(values['semantic_score']+values['appearance_score']+geometric)/(2+values['visible_ratio']) if complete else None
    return {**values,'geometry_contribution':geometric,'score_reconstructed':reconstructed,
            'ism_formula_matches':math.isclose(reconstructed,detection['score'],abs_tol=2e-5) if complete and detection.get('score') is not None else None}


def pose_section(pose,relative):
    e=html.escape
    if pose is None:return '<section class="pose"><h3>6D pose · PEM</h3><p>This run did not estimate poses. Use <code>suite run --pose</code> or <code>suite pose --output SUITE_DIRECTORY</code>.</p></section>'
    block=['<section class="pose"><h3>6D pose · PEM</h3><p>Translation in mm. T maps original CAD points into OpenCV camera coordinates: X right, Y down, Z forward. The combined PEM score incorporates ISM and ranks poses; it is not another detection confidence or a calibrated probability.</p>']
    if pose.get('elapsed_s') is not None:block.append(f'<p>PEM time, separate from ISM: {pose["elapsed_s"]:.3f} s.</p>')
    for overlay in pose.get('overlays',[]):
        url=relative+'/'+quote(overlay['path'],safe='/')
        block.append(f'<figure><img style="max-width:100%" src="{url}" loading="lazy" alt="PEM pose with CAD coordinate axes"><figcaption>{e(overlay["part_key"])} · X red, Y green, Z blue · original CAD origin</figcaption></figure>')
    for index,item in enumerate(pose.get('poses',[])):
        transform=item.get('T_camera_from_input_cad_mm',item.get('T_camera_from_model_mm'))
        translation=[row[3] for row in transform[:3]] if transform else item.get('t_mm')
        rotation=[row[:3] for row in transform[:3]] if transform else item.get('R')
        title=f'{item.get("part_key","CAD")} · detection {item.get("detection_index",index)}'
        values={'R_camara_desde_CAD':rotation,'t_CAD_mm':translation,'T_camara_desde_CAD_mm':transform,
                'T_camara_desde_modelo_PEM_mm':item.get('T_camera_from_model_mm')}
        matrix_text=lambda matrix:'\n'.join('[ '+' '.join(f'{float(value): .6f}' for value in row)+' ]' for row in matrix)
        t_text=', '.join(f'{float(value):.3f}' for value in translation) if translation else '—'
        block.append(f'<h4>{e(title)}</h4><p>ISM: {item.get("segmentation_score",0):.3f} · combined PEM × ISM score: {item.get("pose_score",0):.3f} · <b>t CAD = ({e(t_text)}) mm</b></p>')
        if rotation is not None:block.append('<p>R camera ← CAD:</p><pre>'+e(matrix_text(rotation))+'</pre>')
        if transform is not None:block.append('<p>T camera ← CAD (translation in mm):</p><pre>'+e(matrix_text(transform))+'</pre>')
        block.append('<details><summary>Full values and centred-model transformation</summary><pre>'+e(json.dumps(values,ensure_ascii=False,indent=2))+'</pre></details>')
    if not pose.get('poses'):block.append('<p>No pose was calculated: '+e(pose.get('status','no result'))+'.</p>')
    for error in pose.get('errors',[]):block.append('<p class="error">'+e(error.get('part_key','CAD')+': '+error['error'])+'</p>')
    block.append(f'<a href="{relative}/pose.json">Pose JSON and geometry audit</a></section>')
    return ''.join(block)


def pose_gallery(frames, poses_by_frame):
    """Link the overview directly to saved CAD-frame pose matrices and overlays."""
    cards=[]
    for frame in frames:
        pose=poses_by_frame.get(frame['name']) or {}
        anchor=f"frame{frame['frame_id']}-pose"
        label=html.escape(frame['name'])
        overlays=pose.get('overlays',[])
        if not overlays:
            status='No accepted detection' if pose.get('status')=='no_accepted_detection' else 'Pose not calculated'
            cards.append(f'<div><a href="#{anchor}">{label}</a><p>{status}</p></div>')
        for overlay in overlays:
            url=quote('cases/'+frame['name']+'/'+overlay['path'],safe='/')
            cad=html.escape(overlay['part_key'])
            cards.append(f'<a href="#{anchor}" class="pose-card"><img src="{url}" loading="lazy" alt="PEM coordinate axes for {cad}"><strong>{label}</strong><span>{cad}</span><span>View R, t and T matrices</span></a>')
    return ('<section class="panel" id="poses"><h2>Matrices and coordinate axes · Pose 6D PEM</h2>'
            '<p>Select an image to inspect rotation R, translation t in millimetres and the 4×4 transformation T. '
            'Axes are X red, Y green and Z blue, at the original CAD origin.</p>'
            '<div class="pose-gallery">'+''.join(cards)+'</div></section>')


def publish_dashboard(report_dir, dashboard=None):
    """Keep the served dashboard on the last exported report; preserve a legacy export."""
    import os
    import uuid
    from datetime import datetime,timezone
    from settings import RESULTS
    report_dir=Path(report_dir).resolve()
    dashboard=Path(dashboard) if dashboard is not None else RESULTS/'dashboard'
    if not (report_dir/'index.html').is_file(): raise ValueError('Report index.html is missing')
    if dashboard.resolve()==report_dir: return dashboard
    if dashboard.resolve() in report_dir.parents: raise ValueError('The report cannot be inside the dashboard being replaced')
    if dashboard.exists() and not dashboard.is_symlink():
        if not (dashboard/'.sam6d_suite_export').is_file(): raise ValueError('The dashboard contains unrelated files; it is preserved unchanged')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        dashboard.rename(dashboard.with_name(dashboard.name+'_legacy_'+stamp+'_'+uuid.uuid4().hex[:6]))
    dashboard.parent.mkdir(parents=True,exist_ok=True)
    temporary=dashboard.with_name('.'+dashboard.name+'-'+uuid.uuid4().hex)
    try:
        temporary.symlink_to(os.path.relpath(report_dir,dashboard.parent),target_is_directory=True)
        temporary.replace(dashboard)
    finally:
        if temporary.is_symlink():temporary.unlink()
    return dashboard


def final_confidences(run, source):
    """Read actual final ISM scores; an unassigned CAD has no saved final score."""
    catalog = {p["key"]: p for p in run["catalog"]}
    detections_by_frame, per_cad = {}, []
    for frame in run["frames"]:
        detections = json.loads((source / "cases" / frame["name"] / "detection_ism.json").read_text())
        detections_by_frame[frame["name"]] = detections
        for detection in detections:
            part = catalog[detection["part_key"]]
            if detection["category_id"] != part["category_id"] or detection["part_name"] != part["name"]:
                raise ValueError("The saved label does not match the catalog used by this run.")
            if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
                raise ValueError("Invalid final SAM-6D score.")
        for part in run["catalog"]:
            assigned = [(i, d) for i, d in enumerate(detections) if d["part_key"] == part["key"]]
            best = max(assigned, key=lambda item: item[1]["score"]) if assigned else None
            per_cad.append({"imagen": frame["name"], "cad_predicho": part["key"], "nombre_catalogo": part["name"],
                            "confianza_final_sam6d": best[1]["score"] if best else None,
                            **score_components(best[1] if best else {}),
                            "detecciones_conservadas": len(assigned), "indice_deteccion": best[0] if best else None,
                            "supera_umbral": best[1]["score"] >= run["settings"]["confidence"] if best else False,
                            "origen_score": "detection_ism.json:score",
                            "alcance": "maximum among detections assigned to this CAD; not an all-CAD comparison on one mask"})
    return detections_by_frame, per_cad


def comparison_coverage(run):
    """Verify saved all-CAD comparisons independently of detection filtering."""
    frames = []
    for frame in run["frames"]:
        count = frame["proposal_counts"].get("valid_depth_and_size", 0)
        matrix = frame.get("proposal_semantic_scores", [])
        valid_matrix = len(matrix) == count and all(len(row) == len(run["catalog"]) for row in matrix)
        entries = {c["part_key"]: c for c in frame["comparisons"]}
        unique = len(entries) == len(frame["comparisons"])
        candidates = []
        for part in run["catalog"]:
            item = entries.get(part["key"], {})
            score = item.get("semantic_score")
            compared = (count > 0 and valid_matrix and unique and item.get("compared_proposals") == count
                        and isinstance(score, (int, float)) and math.isfinite(score))
            candidates.append({"part_key": part["key"], "compared": compared,
                               "semantic_score": score, "compared_proposals": count if compared else 0})
        frames.append({"name": frame["name"], "valid_proposals": count, "candidates": candidates,
                       "all_cads_compared": all(c["compared"] for c in candidates)})
    expected = len(run["frames"]) * len(run["catalog"])
    verified = sum(c["compared"] for f in frames for c in f["candidates"])
    return {"images": len(frames), "cad_models": len(run["catalog"]), "expected_image_cad_pairs": expected,
            "verified_image_cad_pairs": verified, "complete": expected > 0 and expected == verified,
            "reference_views_per_cad": run.get("algorithm", {}).get("views_per_cad"),
            "scope": "Semantic comparison of all valid proposals against all CADs; this does not imply a final score for every class.",
            "frames": frames}


def chart(frames, target):
    if not frames:
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    names = [f["name"] for f in frames]
    fig, ax = plt.subplots(figsize=(12, max(4, len(frames) * 0.35)))
    left = [0.0] * len(frames)
    groups = [("sam_s", "SAM"), ("dino_s", "DINOv2"), ("semantic_s", "All-CAD comparison"),
              ("refinement_s", "Appearance and geometry"), ("other_s", "Reading and postprocessing")]
    for key, label in groups:
        values = []
        for frame in frames:
            t = frame["timing"]
            refine = sum(t.get(k, 0) for k in ("appearance_s", "projection_s", "geometry_s"))
            if key == "refinement_s":
                value = refine
            elif key == "other_s":
                value = max(0, t["processing_s"] - sum(t[k] for k in ("sam_s", "dino_s", "semantic_s")) - refine)
            else:
                value = t[key]
            values.append(value)
        ax.barh(names, values, left=left, label=label)
        left = [a + b for a, b in zip(left, values)]
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per image · model/reference setup excluded")
    ax.set_title("SAM-6D: measured processing times")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def export_report(source, target, create_archive=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Export must be separate from the original run.")
    run = json.loads((source / "run.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".sam6d_suite_export"
    if not marker.exists() and any(target.iterdir()):
        raise ValueError("The export directory contains files not owned by this tool.")
    marker.touch()
    shutil.copy2(source / "run.json", target / "resultados_completos.json")
    if (source/"dataset_manifest.json").is_file(): shutil.copy2(source/"dataset_manifest.json", target/"dataset_manifest.json")
    if (source / "execution.log").is_file():
        shutil.copy2(source / "execution.log", target / "execution.log")
    if (source / "cases").is_dir():
        shutil.copytree(source / "cases", target / "cases", dirs_exist_ok=True)
    manifest_path, base = run_dataset(run, source)
    dataset = json.loads(manifest_path.read_text())
    frame_lookup = {f["name"]: f for f in dataset["frames"]}
    expected_cad_by_frame = {f["name"]: f.get("expected_cad_key") for f in dataset["frames"] if f.get("expected_cad_key") is not None}
    expected_total = len(expected_cad_by_frame)
    detections_by_frame, final_per_cad = final_confidences(run, source)
    for row in final_per_cad:
        row["expected_cad_key"] = expected_cad_by_frame.get(row["imagen"])
        row["matches_expected_cad_key"] = bool(row["expected_cad_key"]) and row["cad_predicho"] == row["expected_cad_key"]
    poses_by_frame={};pose_rows=[]
    if (source/'pose_run.json').is_file():
        pose_run=json.loads((source/'pose_run.json').read_text())
        if pose_run.get('source_run_sha256')!=hashlib.sha256((source/'run.json').read_bytes()).hexdigest():raise ValueError('Poses do not belong to this ISM suite')
        shutil.copy2(source/'pose_run.json',target/'pose_run.json')
    for frame in run['frames']:
        pose=frame.get('pose');saved_pose=source/'cases'/frame['name']/'pose.json'
        if saved_pose.is_file():pose=json.loads(saved_pose.read_text())
        poses_by_frame[frame['name']]=pose
        for item in (pose or {}).get('poses',[]):
            transform=item.get('T_camera_from_input_cad_mm',item.get('T_camera_from_model_mm'))
            pose_rows.append({'imagen':frame['name'],'cad':item.get('part_key'),'deteccion':item.get('detection_index'),
                              'score_ism':item.get('segmentation_score'),'score_pem':item.get('pose_score'),
                              'tx_mm':transform[0][3] if transform else None,'ty_mm':transform[1][3] if transform else None,'tz_mm':transform[2][3] if transform else None,
                              'R_camara_desde_CAD':json.dumps([r[:3] for r in transform[:3]]) if transform else None,
                              'T_camara_desde_CAD_mm':json.dumps(transform),'unidad':'mm','referencia':'Original CAD to OpenCV camera'})
    (target/'resultados_poses.json').write_text(json.dumps({'backend':'pem','translation_unit':'mm','source_ism_sha256':hashlib.sha256((source/'run.json').read_bytes()).hexdigest(),'frames':poses_by_frame},ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save_csv(target/'poses_pem.csv',pose_rows,['imagen','cad','deteccion','score_ism','score_pem','tx_mm','ty_mm','tz_mm','R_camara_desde_CAD','T_camara_desde_CAD_mm','unidad','referencia'])
    diagnostic_path = source / "all_cad_primary_region.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else None
    dense_by_name = {}
    if diagnostic and diagnostic["status"] == "complete":
        if diagnostic["source_run_sha256"] != hashlib.sha256((source / "run.json").read_bytes()).hexdigest():
            raise ValueError("The diagnostic does not belong to this original run.")
        dense_by_name = {f["name"]: f for f in diagnostic["frames"] if f["status"] == "scored"}
        shutil.copy2(diagnostic_path, target / diagnostic_path.name)
        dense_csv = [{"imagen": f["name"], "cad": s["part_key"], "nombre": s["part_name"],
                      "score_hipotesis": s["score"], "semantic": s["semantic"], "appearance": s["appearance"],
                      "geometry": s["geometry"], "visible_ratio": s["visible_ratio"],
                      "alcance": "all CADs on the SAME primary region; additional unfiltered evaluation"}
                     for f in dense_by_name.values() for s in f["scores"]]
        save_csv(target / "scores_todos_cad_region_principal.csv", dense_csv,
                 ["imagen", "cad", "nombre", "score_hipotesis", "semantic", "appearance", "geometry", "visible_ratio", "alcance"])
    inference_catalog = copy.deepcopy(run["catalog"])
    corrections = {"names": display_names()}
    names = corrections["names"]
    # Keep the original JSON exports untouched; only presentation names are corrected.
    run = copy.deepcopy(run)
    for part in run["catalog"]:
        part["name"] = names.get(part["key"], part["name"])
    for frame in run["frames"]:
        if frame["candidate"]:
            candidate = frame["candidate"]
            candidate["part_name"] = names.get(candidate["part_key"], candidate["part_name"])
        for item in frame["comparisons"]:
            item["part_name"] = names.get(item["part_key"], item["part_name"])
    for row in final_per_cad:
        row["nombre_catalogo"] = names.get(row["cad_predicho"], row["nombre_catalogo"])
    if names:
        (target / "correccion_nombres_cad.json").write_text(json.dumps(corrections, ensure_ascii=False, indent=2) + "\n")
    coverage = comparison_coverage(run)
    (target / "cobertura_comparaciones.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
    view_path = source / 'ism_view_details.json'
    view_frames = {}
    view_rows = []
    audit_note = "<p>This suite has no per-view audit yet.</p>"
    if view_path.is_file():
        view_data = json.loads(view_path.read_text())
        if view_data.get('source_run_sha256') != hashlib.sha256((source/'run.json').read_bytes()).hexdigest():
            raise ValueError('The per-view audit does not belong to this suite')
        if view_data.get('status') != 'complete':
            raise ValueError('The per-view audit is incomplete; check ism_view_details.json')
        shutil.copy2(view_path, target/view_path.name)
        view_frames = {f['name']: f for f in view_data['frames']}
        audit_note = ('<p>Additional audit: device '+html.escape(view_data['device'])+
                      ' · semantic aggregation: '+html.escape(str(view_data.get('aggregation')))+
                      '. Original ISM timings exclude this audit.</p>'
                      '<p><a href="ism_vistas.csv">Per-view score CSV</a> · <a href="ism_view_details.json">Audit JSON and additional timings</a></p>')
        for frame in view_data['frames']:
            for item in frame['scores']:
                for view, (semantic, appearance) in enumerate(zip(item['semantic_by_view'], item['appearance_by_view'])):
                    view_rows.append({'imagen':frame['name'], 'cad':item['part_key'], 'vista':view,
                                      'dinov2_semantic':semantic, 'appearance':appearance,
                                      'ganadora_semantica':view == item['semantic_winner'],
                                      'ganadora_apariencia':view == item['appearance_winner'],
                                      'agregacion_semantica':view_data.get('aggregation')})
    save_csv(target/'ism_vistas.csv', view_rows, ['imagen','cad','vista','dinov2_semantic','appearance','ganadora_semantica','ganadora_apariencia','agregacion_semantica'])
    cad_previews = {}
    manifest_parts = {p["key"]: p for p in dataset["parts"]}
    for part in run["catalog"]:
        folder = base / manifest_parts[part["key"]]["sam_template_dir"]
        if (folder / "rgb_2.png").is_file():
            relative = Path("referencias_cad") / (part["key"] + ".png")
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            reference_preview(folder, 2, target / relative)
            reference_preview(folder, 19, target / relative.with_stem(relative.stem + "_frontal"))
            cad_previews[part["key"]] = relative.as_posix()
    all_reference_views = reference_gallery(run["catalog"], manifest_parts, base, target)
    label_provenance = {
        "nombre_captura": "Original RGB filename; only identifies the capture.",
        "cad_predicho": "category_id and part_key selected by SAM-6D by comparing masks against CAD references.",
        "nombre_prediccion": "CAD ID selected by the model; catalog name with explicit user corrections in correccion_nombres_cad.json.",
        "origen_inicial_nombres": "PDDL/DINOPDDL/catalogs/printed_assembly_catalog.yaml, read during inventory preparation.",
        "nombre_archivo_usado_para_clasificar": run["settings"].get("filename_used_for_classification"),
        "etiquetas_reales_verificadas": False,
        "score_mostrado": "detection_ism.json:score (final SAM-6D ISM score, not semantic_score)",
        "catalogo_utilizado_en_inferencia": inference_catalog,
        "catalogo_mostrado": run["catalog"],
        "correcciones_usuario": corrections,
    }
    (target / "origen_etiquetas.json").write_text(json.dumps(label_provenance, ensure_ascii=False, indent=2) + "\n")
    save_csv(target / "confianza_sam6d_por_cad.csv", final_per_cad,
             ["imagen", "cad_predicho", "nombre_catalogo", "confianza_final_sam6d", "detecciones_conservadas",
              "indice_deteccion", "supera_umbral", "origen_score", "alcance", "expected_cad_key", "matches_expected_cad_key", *SCORE_FIELDS])
    # Export calibration together with lossless depth for each real capture.
    for frame in run["frames"]:
        original = frame_lookup[frame["name"]]
        saved_camera = source / "cases" / frame["name"] / "camera.json"
        camera_source = saved_camera if saved_camera.is_file() else base / original["camera"]
        shutil.copy2(camera_source, target / "cases" / frame["name"] / "camera.json")
    audit = base / "photo_naming_audit.json"
    if audit.is_file():
        audit_data = json.loads(audit.read_text())
        if names:
            audit_data["historical_note_es"] = audit_data.get("note_es")
            audit_data["status"] = "superseded_by_user_correction"
            audit_data["note_es"] = "User correction: 03a is sensor_module and 03b is communications_module. These names were swapped in the input catalog; this discrepancy is not attributed to the photographs."
        (target / audit.name).write_text(json.dumps(audit_data, ensure_ascii=False, indent=2) + "\n")
    for filename in ("gpu_verification.json", "sam6d_environment_gpu.json", "runtime_checkpoints.json",
                     "sam6d_suite_validation.json"):
        if (base / filename).is_file():
            shutil.copy2(base / filename, target / filename)
    rows, comparisons = [], []
    for frame in run["frames"]:
        candidate = frame["candidate"] or {}
        rows.append({"imagen": frame["name"], "estado": frame["status"], "dispositivo": frame["device"],
                     "identificacion_aceptada": frame["recognized"],
                     "componente_principal": candidate.get("part_key") if frame["recognized"] else "no reliable identification",
                     "candidato": candidate.get("part_key"), "score_ism": candidate.get("score"),
                     "componentes_visibles": ";".join(frame["visible_components"]), **frame["timing"]})
        for item in frame["comparisons"]:
            comparisons.append({"imagen": frame["name"], "dispositivo": frame["device"],
                                "region_propuesta": frame["comparison_region_proposal_id"],
                                "alcance_region": frame["comparison_scope"], **item,
                                "semantic_comparison_ms": item["semantic_comparison_s"] * 1000})
    time_fields = list(dict.fromkeys(key for row in rows for key in row if key.endswith("_s")))
    save_csv(target / "tiempos_por_imagen.csv", rows,
             ["imagen", "estado", "dispositivo", "identificacion_aceptada", "componente_principal", "candidato",
              "score_ism", "componentes_visibles", *time_fields])
    confidence_rows = [{"archivo_original": Path(frame_lookup[f["name"]]["source_rgb"]).name,
                        "cad_predicho": f["candidate"]["part_key"] if f["candidate"] else None,
                        "nombre_catalogo": f["candidate"]["part_name"] if f["candidate"] else None,
                        "confianza_final_sam6d": f["candidate"]["score"] if f["candidate"] else None,
                        "identificacion_aceptada": f["recognized"], "umbral": run["settings"]["confidence"],
                        "tiempo_proceso_s": f["timing"]["processing_s"], "origen_score": "detection_ism.json:score",
                        "expected_cad_key": expected_cad_by_frame.get(f["name"]),
                        "matches_expected_cad_key": bool(expected_cad_by_frame.get(f["name"])) and bool(f["candidate"]) and f["candidate"]["part_key"] == expected_cad_by_frame.get(f["name"]),
                        **score_components(f['candidate'] or {}), "etiqueta_real_verificada": False} for f in run["frames"]]
    save_csv(target / "confianza_sam6d_por_imagen.csv", confidence_rows,
             ["archivo_original", "cad_predicho", "nombre_catalogo", "confianza_final_sam6d", "identificacion_aceptada",
              "umbral", "tiempo_proceso_s", "origen_score", "expected_cad_key", "matches_expected_cad_key",
              "etiqueta_real_verificada", *SCORE_FIELDS])
    save_csv(target / "comparacion_todos_los_cad.csv", comparisons,
             ["imagen", "dispositivo", "region_propuesta", "alcance_region", "part_key", "part_name", "category_id",
              "semantic_score", "best_template", "compared_proposals", "semantic_comparison_ms", "semantic_comparison_s"])
    chart(run["frames"], target / "tiempos_por_imagen.png")
    durations = [f["timing"]["processing_s"] for f in run["frames"]]
    expected_matches = sum(1 for frame in run["frames"] if bool(expected_cad_by_frame.get(frame["name"])) and bool(frame.get("candidate")) and frame["candidate"]["part_key"] == expected_cad_by_frame.get(frame["name"]))
    expected_accuracy = (expected_matches / expected_total) if expected_total else None
    stats = {"processed": len(durations), "selected": len(run["selected_frames"]), "device": run["device"]["used"],
             "recognized": sum(f["recognized"] for f in run["frames"]),
             "expected_cad_matches": expected_matches,
             "expected_cad_total": expected_total,
             "expected_cad_accuracy": expected_accuracy,
             "mean_processing_s": statistics.mean(durations) if durations else None,
             "median_processing_s": statistics.median(durations) if durations else None,
             "min_processing_s": min(durations) if durations else None,
             "max_processing_s": max(durations) if durations else None,
             "setup_timing": run["setup_timing"], "wall_s": run.get("wall_s")}
    (target / "resumen.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    expected_cad_metric = None
    if expected_total:
        expected_cad_metric = f"{expected_matches}/{expected_total} = {(expected_matches / expected_total):.1%}"
    def e(value):
        return html.escape(str(value))

    def fmt(value):
        return f"{value:.3f}" if value is not None else "—"

    tables, cards, coverage_rows, final_matrix_rows = [], [], [], []
    for number, frame in enumerate(run["frames"], 1):
        c, t = frame["candidate"], frame["timing"]
        label = c["part_key"] if c else "No retained detection"
        score = c["score"] if c else None
        filename = Path(frame_lookup[frame["name"]]["source_rgb"]).name
        status = "Above threshold" if frame["recognized"] else "No accepted identification"
        anchor = f"frame{frame['frame_id']}"
        checked = coverage["frames"][number - 1]
        chosen = c["part_key"] if c and frame["recognized"] else None
        cells = "".join(f'<td class="{score_style(item["semantic_score"], item["part_key"] == chosen)}" data-cad="{e(item["part_key"])}" data-compared="{str(item["compared"]).lower()}">'
                        f'{fmt(item["semantic_score"]) if item["compared"] else "—"}</td>'
                        for item in checked["candidates"])
        coverage_rows.append(f'<tr data-image="{e(frame["name"])}"><td><a href="#{anchor}">{e(filename)}</a></td>{cells}</tr>')
        components=score_components(c or {})
        pose=poses_by_frame.get(frame['name'])
        candidates=(pose or {}).get('poses',[])
        associated=[item for item in candidates if c and item.get('part_key')==c['part_key']]
        selected_pose=max(associated or candidates,key=lambda item:item.get('pose_score',0),default=None)
        pose_caption='Not calculated'
        if selected_pose:
            transform=selected_pose.get('T_camera_from_input_cad_mm',selected_pose.get('T_camera_from_model_mm'))
            pose_caption=selected_pose.get('part_key','CAD')+': t = ('+', '.join(f'{row[3]:.2f}' for row in transform[:3])+') mm'
        tables.append(f'<tr><td><a href="#{anchor}">{e(filename)}</a></td><td>{e(label)}</td>'
                      f'<td data-score-source="score"><b>{fmt(score)}</b></td>'
                      f'<td>{fmt(components["semantic_score"])}</td><td>{fmt(components["appearance_score"])}</td><td>{fmt(components["geometry_score"])}</td><td>{fmt(components["visible_ratio"])}</td><td>{e(status)}</td>'
                      f'<td>{fmt(t["processing_s"])}</td><td><a href="#{anchor}-pose">{e(pose_caption)}</a></td></tr>')
        ordered = sorted(frame["comparisons"], key=lambda item: item["semantic_score"] if item["semantic_score"] is not None else -1, reverse=True)
        ranking = "".join(f'<tr><td>{rank}</td><td>{e(item["part_key"])}</td><td>{e(item["part_name"])}</td>'
                          f'<td>{fmt(item["semantic_score"])}</td><td>{item["semantic_comparison_s"] * 1000:.3f}</td></tr>'
                          for rank, item in enumerate(ordered, 1))
        rel = "cases/" + quote(frame["name"], safe="")
        details = " · ".join(f'{e(k)}: {fmt(v)} s' for k, v in t.items())
        visible = ", ".join(frame["visible_components"]) or "none above threshold"
        final_rows = [r for r in final_per_cad if r["imagen"] == frame["name"]]
        dense = dense_by_name.get(frame["name"])
        display_rows = final_rows
        if dense:
            lookup = {s["part_key"]: s for s in dense["scores"]}
            display_rows = [{"cad_predicho": p["key"], "confianza_final_sam6d": lookup[p["key"]]["score"]} for p in run["catalog"]]
        matrix_cells = "".join(f'<td class="{score_style(r["confianza_final_sam6d"], r["cad_predicho"] == chosen)}" '
                               f'data-cad="{e(r["cad_predicho"])}" data-score-source="{"diagnostic_hypothesis" if dense else "score"}" '
                               f'title="{e(r["cad_predicho"])}: {fmt(r["confianza_final_sam6d"])}">{fmt(r["confianza_final_sam6d"])}</td>'
                               for r in display_rows)
        roi_thumbnail = f'<br><img class="roi-thumb" src="{rel}/primary_region.png" alt="Region used to compare all ten CADs" loading="lazy">' if dense else ""
        final_matrix_rows.append(f'<tr data-image="{e(frame["name"])}"><td><a href="#{anchor}">{e(filename)}</a>{roi_thumbnail}</td>{matrix_cells}</tr>')
        decision_overlay(target / "cases" / frame["name"], final_rows, chosen)
        final_rows.sort(key=lambda r: r["confianza_final_sam6d"] if r["confianza_final_sam6d"] is not None else -1, reverse=True)
        final_table = "".join(f'<tr class="{score_style(r["confianza_final_sam6d"], r["cad_predicho"] == chosen)}"><td>{e(r["cad_predicho"])}</td>'
                              f'<td data-score-source="score"><b>{fmt(r["confianza_final_sam6d"])}</b></td>'
                              f'<td>{fmt(r.get("semantic_score"))}</td><td>{fmt(r.get("appearance_score"))}</td><td>{fmt(r.get("geometry_score"))}</td><td>{fmt(r.get("visible_ratio"))}</td>'
                              f'<td>{"Selected" if r["cad_predicho"] == chosen else "Alternative > 0.40" if r["confianza_final_sam6d"] > 0.4 else "Low score"}</td>'
                              f'<td><a href="{rel}/mask_{r["indice_deteccion"]:03d}.png">View region</a></td></tr>'
                              for r in final_rows if r["confianza_final_sam6d"] is not None)
        saved = detections_by_frame[frame["name"]]
        detection_rows = "".join(f'<tr><td>{i + 1}</td><td>{e(d["part_key"])}</td>'
                                 f'<td data-score-source="score">{fmt(d["score"])}</td>'
                                 f'<td>{fmt(d.get("semantic_score"))}</td><td>{fmt(d.get("appearance_score"))}</td><td>{fmt(d.get("geometry_score"))}</td><td>{fmt(d.get("visible_ratio"))}</td>'
                                 f'<td>{"Above threshold" if d["score"] >= run["settings"]["confidence"] else "Below threshold"}</td>'
                                 f'<td><a href="{rel}/mask_{i:03d}.png">View mask</a></td></tr>' for i, d in enumerate(saved))
        meter = f'<meter min="0" max="1" value="{score}" aria-label="Final SAM-6D score"></meter>' if score is not None and frame["recognized"] else ""
        confidence_color = "#116b5a" if frame["recognized"] else "#6b7580"
        dense_block = ""
        if dense:
            dense_rows = "".join(f'<tr class="{score_style(s["score"], s["part_key"] == chosen)}"><td>{e(s["part_key"])}</td>'
                                  f'<td>{e(names.get(s["part_key"], s["part_name"]))}</td><td>{fmt(s["score"])}</td><td>{fmt(s.get("semantic"))}</td><td>{fmt(s.get("appearance"))}</td><td>{fmt(s.get("geometry"))}</td><td>{fmt(s.get("visible_ratio"))}</td></tr>'
                                  for s in sorted(dense["scores"], key=lambda s: s["score"], reverse=True))
            dense_block = ('<details class="diagnostic"><summary>All-CAD scores on the same region</summary>'
                           f'<img class="roi-detail" src="{rel}/primary_region.png" alt="Crop of the evaluated primary region" loading="lazy">'
                           '<p>Additional evaluation without dropping low scores: each of the ten CAD identities is hypothesized for this region in turn. '
                           'The complete SAM-6D formula includes appearance and geometry. These are hypotheses on one crop, '
                           'not ten detected objects or probabilities. The original prediction is preserved.</p>'
                           '<table><thead><tr><th>Evaluated CAD</th><th>Corrected name</th><th>Hypothesis score</th><th>Semantic</th><th>Appearance</th><th>Geometric</th><th>Visibility</th></tr></thead>'
                           f'<tbody>{dense_rows}</tbody></table></details>')
        cards.append(f'<details class="capture" id="{anchor}"><summary>Image {number:02d} · {e(filename)} · {e(label)} · ISM {fmt(score)}</summary><article><h2>Image {number:02d} · {e(filename)}</h2>'
                     f'<div class="prediction"><div><small>CAD predicted by SAM-6D</small><p><b>{e(label)}</b></p>'
                     f'<small>Catalog name: {e(c["part_name"] if c else "—")}</small></div>'
                     f'<div class="confidence"><small>Final SAM-6D score · ISM</small><strong style="color:{confidence_color}" data-score-source="score">{fmt(score)}</strong>{meter}'
                     f'<small>{e(status)} · threshold {run["settings"]["confidence"]:.2f}</small></div></div>'
                     f'<p class="ism-components">Semantic: <b>{fmt(components["semantic_score"])}</b> · Appearance: <b>{fmt(components["appearance_score"])}</b> · Geometric: <b>{fmt(components["geometry_score"])}</b> · Visibility: {fmt(components["visible_ratio"])} · Geometry contribution: {fmt(components["geometry_contribution"])}</p>'
                     f'<p>ISM = (semantic + appearance + geometry × visibility) / (2 + visibility). Reconstructed: {fmt(components["score_reconstructed"])}.</p>'
                     f'<p>Processing: {fmt(t["processing_s"])} s · Device: {e(frame["device"])}. '
                     f'Accepted visible components: {e(visible)}.</p>'
                     f'<div class="pictures"><img src="{rel}/rgb.png" alt="RGB original" loading="lazy">'
                     f'<img src="{rel}/overlay_decision.png" alt="Selected candidate in green and alternatives above 0.40 in orange" loading="lazy"></div>'
                     f'<details id="{anchor}-pose"><summary>6D pose: coordinate axes and matrices</summary>{pose_section(pose,rel)}</details>'
                     f'{view_audit_section(view_frames.get(frame["name"]),rel)}'
                     f'{dense_block}'
                     '<p class="provenance">The capture name comes from its RGB filename. The predicted label comes from the selected CAD '
                     'and its catalog name; it was not derived from the photograph filename.</p>'
                     '<details><summary>Original detections retained in this image</summary>'
                     '<p>Green: selected part. Orange: another class with final score &gt; 0.40, which may indicate another part or a confusion. '
                     'Each row shows only the highest-scoring region for that class; multiple masks do not imply multiple objects.</p>'
                     '<table><thead><tr><th>Candidate CAD</th><th>Final ISM score</th><th>Semantic</th><th>Appearance</th><th>Geometric</th><th>Visibility</th><th>Interpretation</th><th>Evaluated region</th></tr></thead>'
                     f'<tbody>{final_table}</tbody></table>'
                     f'<p>All ten CADs were compared. The others have no retained detection in this image. '
                     f'Automatic acceptance remains at {run["settings"]["confidence"]:.2f}; orange is a visual aid, not a new acceptance decision.</p>'
                     '</details>'
                     f'<details><summary>All saved detections and their masks ({len(saved)})</summary>'
                     '<table><thead><tr><th>Detection</th><th>Predicted CAD</th><th>Final ISM score</th><th>Semantic</th><th>Appearance</th><th>Geometric</th><th>Visibility</th><th>Status</th><th>Mask</th></tr></thead>'
                     f'<tbody>{detection_rows}</tbody></table></details>'
                     '<details><summary>Internal technical detail: DINOv2 similarity and timings</summary>'
                     f'<p>SAM generated {checked["valid_proposals"]} valid regions in total. Each was compared with all ten CADs. '
                     'This number does not indicate how many objects were detected or how often each CAD appeared.</p>'
                     '<p>DINOv2 is the internal descriptor of SAM-6D, separate from DINOPDDL. The following table is '
                     'an internal semantic comparison on one region, not the final SAM-6D score. '
                     'The original algorithm refines appearance and geometry for the assigned class; it does not compute ten final scores on every mask.</p>'
                     '<table><thead><tr><th>Rank</th><th>CAD</th><th>Catalog name</th><th>Internal DINOv2 similarity</th>'
                     f'<th>CAD comparison (ms)</th></tr></thead><tbody>{ranking}</tbody></table><p>{details}</p>'
                     f'<pre>{e(json.dumps(frame["proposal_counts"], ensure_ascii=False, indent=2))}</pre>'
                     f'<a href="{rel}/case.json">Case JSON</a> · <a href="{rel}/detection_ism.json">Detections and RLE masks</a>'
                     '</details><p><a href="#inicio">Back to summary</a></p></article></details>')
    setup = run["setup_timing"]
    settings=run['settings'];sam=settings.get('segmentor','sam')=='sam'
    stability_note=('SAM · YAML: '+str(settings.get('stability_score_yaml','not recorded'))+' · requested: '+str(settings.get('stability_score'))+' · effective in the generator: '+str(settings.get('stability_score_effective','not recorded in this historical run'))) if sam else 'FastSAM: the SAM stability threshold does not apply.'
    note = json.loads((target / audit.name).read_text()).get("note_es", "") if audit.is_file() else ""
    cad_gallery = "".join(f'<div><div class="cad-views"><img src="{e(cad_previews[p["key"]])}" alt="Oblique CAD view {e(p["key"])}" loading="lazy">'
                          f'<img src="{e(str(Path(cad_previews[p["key"]]).with_stem(p["key"] + "_frontal")))}" alt="Second CAD view {e(p["key"])}" loading="lazy"></div>'
                          f'<strong>{e(p["key"])}</strong><small>{e(p["name"])}</small></div>'
                          for p in run["catalog"] if p["key"] in cad_previews)
    matrix_headers = "".join(f'<th title="{e(p["key"])}">{e(p["key"].split("_")[0])}<br><small>{e(p["name"])}</small></th>' for p in run["catalog"])
    matrix_title = "All-CAD scores on the same region" if dense_by_name else "Which part SAM-6D selected in each photograph"
    matrix_description = (
        'Every cell now contains an additional measurement. Each photograph reuses the same primary region shown in its crop, '
        'and evaluates all ten CADs with the complete SAM-6D formula without score filtering. '
        'This is an additional hypothesis evaluation; it does not replace original predictions or detections.'
        if dense_by_name else 'Each row is a photograph and each column a CAD. Values are final scores of retained detections.')
    matrix_missing_note = (
        'Gaps in the previous table were scores that were not computed or were removed by filtering, rather than zeros. '
        'This additional view replaces them with scores computed on the same crop. '
        '<a href="scores_todos_cad_region_principal.csv">Download all 220 unfiltered scores</a>.'
        if dense_by_name else 'All ten CADs were compared against the 22 images. A dash does not mean an omitted CAD or zero score: no detection was retained for that class.')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM-6D · Final score per image</title><style>
body{{font:16px system-ui,sans-serif;background:#eef2f6;color:#16293a;max-width:1300px;margin:auto;padding:24px;line-height:1.5}}
article,.panel{{background:white;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:8px;border-bottom:1px solid #ccd5dd;text-align:left}}th{{background:#e9eff5}}a{{color:#075a9f}}
.pictures{{display:flex;gap:12px;flex-wrap:wrap}}.pictures img{{width:min(100%,600px);height:auto}}img.chart{{max-width:100%}}
pre{{white-space:pre-wrap}}
summary{{cursor:pointer;font-weight:600;padding:12px;border-radius:6px;background:#e9eff5}}details[open]>summary{{margin-bottom:12px}}.capture{{background:white;border-radius:12px;padding:12px;margin:16px 0}}.capture article{{margin:0;padding:12px}}.view-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}.view-grid figure{{margin:0}}.view-grid img{{width:100%;height:160px;object-fit:contain}}.view-thumb{{width:90px;height:72px;object-fit:contain;vertical-align:middle}}table{{display:block;overflow-x:auto}}
.pose-gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}.pose-card{{display:flex;flex-direction:column;gap:5px;padding:12px;background:#edf5f3;border-radius:8px}}.pose-card img{{width:100%;height:auto}}details{{margin-top:14px}}.device{{font-size:24px;font-weight:bold}}h1{{margin-bottom:0}}
.prediction{{display:flex;justify-content:space-between;gap:20px;padding:18px;background:#edf5f3;border-radius:10px;flex-wrap:wrap}}
.prediction p{{margin:5px 0}}.prediction small{{display:block;color:#526979}}.confidence strong{{display:block;font-size:42px;color:#116b5a}}
meter{{width:180px;height:16px}}.provenance{{padding:12px;background:#f0f3f7;border-left:3px solid #8aa9ba;font-size:14px}}
.checked{{color:#136b52;background:#edf8f2;white-space:nowrap}}.cad-gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin:18px 0}}
.score-winner{{background:#ccebd9;color:#115b36;font-weight:bold;box-shadow:inset 0 0 0 2px #228453}}
.score-alternative{{background:#ffe1b2;color:#7e4300;font-weight:bold}}.score-low{{color:#6b7580;background:#f7f8fa}}.score-empty{{color:#a4aab1;background:#fff}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}.legend span{{padding:7px 12px;border-radius:5px}}
.cad-gallery{{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}}.cad-views{{display:flex;gap:4px}}.cad-gallery img{{width:49%;height:150px;object-fit:contain;border-radius:6px}}.cad-gallery strong,.cad-gallery small{{display:block;font-size:12px;overflow-wrap:anywhere}}
.matrix-scroll{{overflow:auto}}#matriz-comparaciones{{font-size:12px}}#matriz-comparaciones th{{min-width:84px}}#matriz-comparaciones td:first-child{{white-space:nowrap}}
.roi-thumb{{width:75px;height:55px;object-fit:contain}}.roi-detail{{max-width:240px;max-height:220px;object-fit:contain}}
.diagnostic{{background:#f5f7fb;border:1px solid #d8e2ed;border-radius:8px;padding:16px;margin-top:16px}}
</style><body id="inicio"><h1>SAM-6D · Final score per image</h1>
<p class="device">Actual device: {e(run["device"]["used"])} {e(run["device"].get("gpu_name") or "")}</p>
<p><a href="#metodo-ism">How ISM works</a> · <a href="#referencias">All CAD views</a> · <a href="#poses">View 6D pose matrices and axes</a> · <a href="#comparacion-cads">View selected candidates and alternatives per image · {coverage["verified_image_cad_pairs"]}/{coverage["expected_image_cad_pairs"]} verified comparisons</a></p>
<div class="panel"><p>Status: {e(run["status"])} · {stats["processed"]}/{stats["selected"]} captures processed ·
{len(run["catalog"])} CAD × 42 views · {stats["recognized"]} images with accepted identification.</p>
<p><b>Match against the manifest:</b> {expected_cad_metric if expected_cad_metric is not None else 'no expected_cad_key in the manifest'}.</p>
<p>Mean processing time: <b>{fmt(stats["mean_processing_s"])} s</b> · median: {fmt(stats["median_processing_s"])} s.
Model loading: {fmt(setup.get("model_load_s"))} s · Reference preparation/loading: {fmt(setup.get("references_s"))} s.
These setup costs are paid once per run and reported separately from per-image latency.</p>
<p>PEM: {len(pose_rows)} poses estimated in {sum(bool((pose or {}).get("poses")) for pose in poses_by_frame.values())} captures. Pose values and errors are reported separately from ISM scores.</p>
<p><b>Effective stability:</b> {e(stability_note)}</p>
<p><b>The displayed score is the final SAM-6D detection and segmentation score (ISM), on a 0–1 scale.</b>
It is not a calibrated probability or 6D pose confidence. An image may contain multiple parts; the primary candidate is the highest-scoring detection.
The configured acceptance and drawing threshold is {run["settings"]["confidence"]:.2f}; detections are also retained from {run["settings"]["min_score"]:.2f}.</p>
<div class="provenance"><b>Label provenance:</b> Original file is the photograph filename; Predicted CAD is the model decision.
The semantic CAD name comes from the prepared part catalog, including user-confirmed name corrections. No independently verified per-photo labels are available in this historical run; filenames are not used for classification.
<a href="origen_etiquetas.json">View label mapping</a>. {e(note)}</div>
<p><a href="confianza_sam6d_por_imagen.csv">Final per-image score CSV</a> · <a href="confianza_sam6d_por_cad.csv">Final per-CAD score CSV</a> ·
<a href="poses_pem.csv">PEM pose CSV</a> · <a href="resultados_poses.json">Pose JSON</a> · <a href="resultados_completos.json">Full ISM JSON</a></p>
<table id="resumen-confianza"><thead><tr><th>Original capture file</th><th>Predicted CAD / candidate</th><th>Final ISM score</th><th>Semantic</th><th>Appearance</th><th>Geometric</th><th>Visibility</th><th>Status</th><th>ISM processing (s)</th><th>PEM pose</th>
</tr></thead><tbody>{''.join(tables)}</tbody></table></div>
{methodology(audit_note)}
{all_reference_views}
<details class="panel"><summary>6D pose gallery</summary>{pose_gallery(run["frames"], poses_by_frame)}</details>
<details class="panel" id="comparacion-cads"><summary>{matrix_title}</summary>
<p>{matrix_description} Click the photograph to inspect details.</p>
<div class="legend"><span class="score-winner">Green: accepted original prediction</span><span class="score-alternative">Orange: another hypothesis &gt; 0.40</span>
<span class="score-low">Grey: score ≤ 0.40</span><span>—: no retained final score</span></div>
<p>A row without green has no accepted candidate at threshold {run["settings"]["confidence"]:.2f}. Orange does not change the threshold or confirm object presence.</p>
<div class="matrix-scroll"><table id="matriz-comparaciones"><thead><tr><th>Image file</th>{matrix_headers}</tr></thead>
<tbody>{''.join(final_matrix_rows)}</tbody></table></div>
<p>{matrix_missing_note}</p>
<details><summary>View all ten CADs with recognizable views</summary><p>Two enlarged views of each reference, with unchanged geometry.</p><div class="cad-gallery">{cad_gallery}</div></details>
<details><summary>Audit: internal comparison of all CADs ({coverage["verified_image_cad_pairs"]}/{coverage["expected_image_cad_pairs"]})</summary>
<p>This second matrix shows internal similarity on the primary region. It is not the final score and should not be compared numerically with the previous table.</p>
<div class="matrix-scroll"><table id="matriz-interna"><thead><tr><th>Image</th>{matrix_headers}</tr></thead><tbody>{''.join(coverage_rows)}</tbody></table></div>
<p><a href="cobertura_comparaciones.json">View coverage</a> · <a href="comparacion_todos_los_cad.csv">Download internal comparisons</a></p></details></details>
<details class="panel"><summary>Stage timings and internal comparisons</summary>
<p>DINOv2 is part of the internal SAM-6D descriptor; these values do not come from the separate DINOPDDL classifier.
The final score combines semantic similarity, appearance and geometric consistency. Internal scores are shown separately within each capture.</p>
<p>One pass per image, with no discarded warm-up. Cached references:
{sum(p["cache_hit"] for p in run.get("references", {}).get("parts", []))}/{len(run["catalog"])} CAD.
PyTorch: {e(run["device"]["torch"])} · CUDA: {e(run["device"].get("torch_cuda_build"))}.</p>
<p><a href="tiempos_por_imagen.csv">Timing CSV</a> · <a href="comparacion_todos_los_cad.csv">Internal DINOv2 similarity CSV (not final scores)</a> ·
<a href="resumen.json">Numerical summary</a></p><img class="chart" src="tiempos_por_imagen.png" alt="Stage timings"></details>{''.join(cards)}
<p>Open index.html in a browser. The report works offline; use <code>suite report --zip</code> to create a ZIP when needed. It contains RGB, original 16-bit depth,
calibration, masks, scores and measurements. It does not include model weights.</p><script>
function revealAnchor() {{ const node=document.getElementById(decodeURIComponent(location.hash.slice(1))); if(!node)return; let p=node; while(p){{if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}} node.scrollIntoView(); }}
window.addEventListener('hashchange',revealAnchor);window.addEventListener('DOMContentLoaded',revealAnchor);
</script></body></html>'''
    (target / "index.html").write_text(page, encoding="utf-8")
    # Remove the previous landing page when regenerating an export owned by this tool.
    legacy = target / "ABRIR_EN_WINDOWS.html"
    if legacy.is_file():
        legacy.unlink()
    if not create_archive: return None
    bundle = target.with_suffix(".zip")
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, Path(target.name) / path.relative_to(target))
    temporary.replace(bundle)
    return bundle
