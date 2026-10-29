"""Portable report of measured SAM-6D runs; no model execution or inferred ground truth."""
from __future__ import annotations

import csv
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
from urllib.parse import quote
import zipfile


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
                raise ValueError("La etiqueta guardada no coincide con el catálogo utilizado en la ejecución.")
            if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
                raise ValueError("Score final SAM-6D no válido.")
        for part in run["catalog"]:
            assigned = [(i, d) for i, d in enumerate(detections) if d["part_key"] == part["key"]]
            best = max(assigned, key=lambda item: item[1]["score"]) if assigned else None
            per_cad.append({"imagen": frame["name"], "cad_predicho": part["key"], "nombre_catalogo": part["name"],
                            "confianza_final_sam6d": best[1]["score"] if best else None,
                            "detecciones_conservadas": len(assigned), "indice_deteccion": best[0] if best else None,
                            "supera_umbral": best[1]["score"] >= run["settings"]["confidence"] if best else False,
                            "origen_score": "detection_ism.json:score",
                            "alcance": "máximo de detecciones asignadas a este CAD; no es una comparación de todos los CAD sobre la misma máscara"})
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
            "scope": "Comparación semántica de todas las propuestas válidas frente a todos los CAD; no implica score final para cada clase.",
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
    groups = [("sam_s", "SAM"), ("dino_s", "DINOv2"), ("semantic_s", "Comparación de todos los CAD"),
              ("refinement_s", "Apariencia y geometría"), ("other_s", "Lectura y posprocesado")]
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
    ax.set_xlabel("Segundos por imagen · preparación de modelos y referencias excluida")
    ax.set_title("SAM-6D: tiempos medidos de procesamiento")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def export_report(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("La exportación debe estar separada de la ejecución original.")
    run = json.loads((source / "run.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".sam6d_suite_export"
    if not marker.exists() and any(target.iterdir()):
        raise ValueError("La carpeta de exportación contiene archivos ajenos a esta herramienta.")
    marker.touch()
    shutil.copy2(source / "run.json", target / "resultados_completos.json")
    if (source / "execution.log").is_file():
        shutil.copy2(source / "execution.log", target / "execution.log")
    if (source / "cases").is_dir():
        shutil.copytree(source / "cases", target / "cases", dirs_exist_ok=True)
    manifest_path = Path(run["manifest"])
    if not manifest_path.is_file():
        manifest_path = Path(__file__).parent / "prepared/manifest_at_execution.json"
    base = manifest_path.parent
    dataset = json.loads(manifest_path.read_text())
    frame_lookup = {f["name"]: f for f in dataset["frames"]}
    detections_by_frame, final_per_cad = final_confidences(run, source)
    diagnostic_path = source / "all_cad_primary_region.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else None
    dense_by_name = {}
    if diagnostic and diagnostic["status"] == "complete":
        if diagnostic["source_run_sha256"] != hashlib.sha256((source / "run.json").read_bytes()).hexdigest():
            raise ValueError("El diagnóstico no corresponde a esta ejecución original.")
        dense_by_name = {f["name"]: f for f in diagnostic["frames"] if f["status"] == "scored"}
        shutil.copy2(diagnostic_path, target / diagnostic_path.name)
        dense_csv = [{"imagen": f["name"], "cad": s["part_key"], "nombre": s["part_name"],
                      "score_hipotesis": s["score"], "semantic": s["semantic"], "appearance": s["appearance"],
                      "geometry": s["geometry"], "visible_ratio": s["visible_ratio"],
                      "alcance": "todos los CAD sobre la MISMA región primaria; evaluación adicional sin filtrado"}
                     for f in dense_by_name.values() for s in f["scores"]]
        save_csv(target / "scores_todos_cad_region_principal.csv", dense_csv,
                 ["imagen", "cad", "nombre", "score_hipotesis", "semantic", "appearance", "geometry", "visible_ratio", "alcance"])
    inference_catalog = copy.deepcopy(run["catalog"])
    correction_file = Path(__file__).with_name("cad_name_corrections.json")
    corrections = json.loads(correction_file.read_text()) if correction_file.is_file() else {"names": {}}
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
    label_provenance = {
        "nombre_captura": "Nombre del archivo RGB original; solo identifica la captura.",
        "cad_predicho": "category_id y part_key elegidos por SAM-6D al comparar las máscaras con las referencias CAD.",
        "nombre_prediccion": "ID CAD elegido por el modelo; nombre del catálogo con correcciones explícitas del usuario en correccion_nombres_cad.json.",
        "origen_inicial_nombres": "PDDL/DINOPDDL/catalogs/printed_assembly_catalog.yaml, leído durante la preparación del inventario.",
        "nombre_archivo_usado_para_clasificar": run["settings"].get("filename_used_for_classification"),
        "etiquetas_reales_verificadas": False,
        "score_mostrado": "detection_ism.json:score (puntuación final ISM de SAM-6D, no semantic_score)",
        "catalogo_utilizado_en_inferencia": inference_catalog,
        "catalogo_mostrado": run["catalog"],
        "correcciones_usuario": corrections,
    }
    (target / "origen_etiquetas.json").write_text(json.dumps(label_provenance, ensure_ascii=False, indent=2) + "\n")
    save_csv(target / "confianza_sam6d_por_cad.csv", final_per_cad,
             ["imagen", "cad_predicho", "nombre_catalogo", "confianza_final_sam6d", "detecciones_conservadas",
              "indice_deteccion", "supera_umbral", "origen_score", "alcance"])
    # Export calibration together with lossless depth for each real capture.
    for frame in run["frames"]:
        original = frame_lookup[frame["name"]]
        shutil.copy2(base / original["camera"], target / "cases" / frame["name"] / "camera.json")
    audit = base / "photo_naming_audit.json"
    if audit.is_file():
        audit_data = json.loads(audit.read_text())
        if names:
            audit_data["historical_note_es"] = audit_data.get("note_es")
            audit_data["status"] = "superseded_by_user_correction"
            audit_data["note_es"] = "Corrección del usuario: 03a es sensor_module y 03b es communications_module. El catálogo de entrada tenía estos dos nombres intercambiados; no se atribuye la discrepancia a las fotos."
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
                     "componente_principal": candidate.get("part_key") if frame["recognized"] else "sin identificación fiable",
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
                        "etiqueta_real_verificada": False} for f in run["frames"]]
    save_csv(target / "confianza_sam6d_por_imagen.csv", confidence_rows,
             ["archivo_original", "cad_predicho", "nombre_catalogo", "confianza_final_sam6d", "identificacion_aceptada",
              "umbral", "tiempo_proceso_s", "origen_score", "etiqueta_real_verificada"])
    save_csv(target / "comparacion_todos_los_cad.csv", comparisons,
             ["imagen", "dispositivo", "region_propuesta", "alcance_region", "part_key", "part_name", "category_id",
              "semantic_score", "best_template", "compared_proposals", "semantic_comparison_ms", "semantic_comparison_s"])
    chart(run["frames"], target / "tiempos_por_imagen.png")
    durations = [f["timing"]["processing_s"] for f in run["frames"]]
    stats = {"processed": len(durations), "selected": len(run["selected_frames"]), "device": run["device"]["used"],
             "recognized": sum(f["recognized"] for f in run["frames"]),
             "mean_processing_s": statistics.mean(durations) if durations else None,
             "median_processing_s": statistics.median(durations) if durations else None,
             "min_processing_s": min(durations) if durations else None,
             "max_processing_s": max(durations) if durations else None,
             "setup_timing": run["setup_timing"], "wall_s": run.get("wall_s")}
    (target / "resumen.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    def e(value):
        return html.escape(str(value))

    def fmt(value):
        return f"{value:.3f}" if value is not None else "—"

    tables, cards, coverage_rows, final_matrix_rows = [], [], [], []
    for number, frame in enumerate(run["frames"], 1):
        c, t = frame["candidate"], frame["timing"]
        label = c["part_key"] if c else "Sin detección conservada"
        score = c["score"] if c else None
        filename = Path(frame_lookup[frame["name"]]["source_rgb"]).name
        status = "Supera el umbral" if frame["recognized"] else "Sin identificación aceptada"
        anchor = f"frame{frame['frame_id']}"
        checked = coverage["frames"][number - 1]
        chosen = c["part_key"] if c and frame["recognized"] else None
        cells = "".join(f'<td class="{score_style(item["semantic_score"], item["part_key"] == chosen)}" data-cad="{e(item["part_key"])}" data-compared="{str(item["compared"]).lower()}">'
                        f'{fmt(item["semantic_score"]) if item["compared"] else "—"}</td>'
                        for item in checked["candidates"])
        coverage_rows.append(f'<tr data-image="{e(frame["name"])}"><td><a href="#{anchor}">{e(filename)}</a></td>{cells}</tr>')
        tables.append(f'<tr><td><a href="#{anchor}">{e(filename)}</a></td><td>{e(label)}</td>'
                      f'<td data-score-source="score"><b>{fmt(score)}</b></td><td>{e(status)}</td>'
                      f'<td>{fmt(t["processing_s"])}</td></tr>')
        ordered = sorted(frame["comparisons"], key=lambda item: item["semantic_score"] if item["semantic_score"] is not None else -1, reverse=True)
        ranking = "".join(f'<tr><td>{rank}</td><td>{e(item["part_key"])}</td><td>{e(item["part_name"])}</td>'
                          f'<td>{fmt(item["semantic_score"])}</td><td>{item["semantic_comparison_s"] * 1000:.3f}</td></tr>'
                          for rank, item in enumerate(ordered, 1))
        rel = "cases/" + quote(frame["name"], safe="")
        details = " · ".join(f'{e(k)}: {fmt(v)} s' for k, v in t.items())
        visible = ", ".join(frame["visible_components"]) or "ninguno por encima del umbral"
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
        roi_thumbnail = f'<br><img class="roi-thumb" src="{rel}/primary_region.png" alt="Región utilizada para comparar los diez CAD" loading="lazy">' if dense else ""
        final_matrix_rows.append(f'<tr data-image="{e(frame["name"])}"><td><a href="#{anchor}">{e(filename)}</a>{roi_thumbnail}</td>{matrix_cells}</tr>')
        decision_overlay(target / "cases" / frame["name"], final_rows, chosen)
        final_rows.sort(key=lambda r: r["confianza_final_sam6d"] if r["confianza_final_sam6d"] is not None else -1, reverse=True)
        final_table = "".join(f'<tr class="{score_style(r["confianza_final_sam6d"], r["cad_predicho"] == chosen)}"><td>{e(r["cad_predicho"])}</td>'
                              f'<td data-score-source="score"><b>{fmt(r["confianza_final_sam6d"])}</b></td>'
                              f'<td>{"Elegida" if r["cad_predicho"] == chosen else "Alternativa > 0,40" if r["confianza_final_sam6d"] > 0.4 else "Puntuación baja"}</td>'
                              f'<td><a href="{rel}/mask_{r["indice_deteccion"]:03d}.png">Ver región</a></td></tr>'
                              for r in final_rows if r["confianza_final_sam6d"] is not None)
        saved = detections_by_frame[frame["name"]]
        detection_rows = "".join(f'<tr><td>{i + 1}</td><td>{e(d["part_key"])}</td>'
                                 f'<td data-score-source="score">{fmt(d["score"])}</td>'
                                 f'<td>{"Supera umbral" if d["score"] >= run["settings"]["confidence"] else "Bajo umbral"}</td>'
                                 f'<td><a href="{rel}/mask_{i:03d}.png">Ver máscara</a></td></tr>' for i, d in enumerate(saved))
        meter = f'<meter min="0" max="1" value="{score}" aria-label="Score final SAM-6D"></meter>' if score is not None and frame["recognized"] else ""
        confidence_color = "#116b5a" if frame["recognized"] else "#6b7580"
        dense_block = ""
        if dense:
            dense_rows = "".join(f'<tr class="{score_style(s["score"], s["part_key"] == chosen)}"><td>{e(s["part_key"])}</td>'
                                  f'<td>{e(names.get(s["part_key"], s["part_name"]))}</td><td>{fmt(s["score"])}</td></tr>'
                                  for s in sorted(dense["scores"], key=lambda s: s["score"], reverse=True))
            dense_block = ('<section class="diagnostic"><h3>Scores de los diez CAD sobre esta misma región</h3>'
                           f'<img class="roi-detail" src="{rel}/primary_region.png" alt="Recorte de la región principal evaluada" loading="lazy">'
                           '<p>Evaluación adicional sin descartar scores bajos: se supone, por turno, que esta región es cada uno de los diez CAD. '
                           'Se calcula la fórmula completa de SAM-6D con apariencia y geometría. Son hipótesis sobre este recorte, '
                           'no diez piezas detectadas ni probabilidades. La predicción original se conserva.</p>'
                           '<table><thead><tr><th>CAD evaluado</th><th>Nombre corregido</th><th>Score de la hipótesis</th></tr></thead>'
                           f'<tbody>{dense_rows}</tbody></table></section>')
        cards.append(f'<article id="{anchor}"><h2>Imagen {number:02d} · {e(filename)}</h2>'
                     f'<div class="prediction"><div><small>CAD predicho por SAM-6D</small><p><b>{e(label)}</b></p>'
                     f'<small>Nombre asociado en el catálogo: {e(c["part_name"] if c else "—")}</small></div>'
                     f'<div class="confidence"><small>Confianza final SAM-6D · ISM</small><strong style="color:{confidence_color}" data-score-source="score">{fmt(score)}</strong>{meter}'
                     f'<small>{e(status)} · umbral {run["settings"]["confidence"]:.2f}</small></div></div>'
                     f'<p>Proceso: {fmt(t["processing_s"])} s · Dispositivo: {e(frame["device"])}. '
                     f'Componentes visibles aceptados: {e(visible)}.</p>'
                     f'<div class="pictures"><img src="{rel}/rgb.png" alt="RGB original" loading="lazy">'
                     f'<img src="{rel}/overlay_decision.png" alt="Elegida en verde y alternativas mayores de 0,40 en naranja" loading="lazy"></div>'
                     f'{dense_block}'
                     '<p class="provenance">El nombre de la captura procede del archivo RGB. La etiqueta predicha procede del CAD elegido '
                     'y de su nombre en el catálogo; no se ha tomado del nombre de esta foto.</p>'
                     '<h3>Detecciones originales conservadas en esta imagen</h3>'
                     '<p>Verde: pieza elegida. Naranja: otra clase con score final &gt; 0,40, que puede corresponder a otra pieza o a una confusión. '
                     'Cada fila muestra solo la región de mayor score de esa clase; varias máscaras no equivalen a varias piezas.</p>'
                     '<table><thead><tr><th>CAD candidato</th><th>Score final SAM-6D</th><th>Interpretación</th><th>Región evaluada</th></tr></thead>'
                     f'<tbody>{final_table}</tbody></table>'
                     f'<p>Se compararon los diez CAD. Los demás no tienen una detección conservada en esta imagen. '
                     f'La aceptación automática sigue en {run["settings"]["confidence"]:.2f}; el naranja es una ayuda visual, no una aceptación nueva.</p>'
                     f'<details><summary>Todas las detecciones guardadas y sus máscaras ({len(saved)})</summary>'
                     '<table><thead><tr><th>Detección</th><th>CAD predicho</th><th>Confianza final SAM-6D</th><th>Estado</th><th>Máscara</th></tr></thead>'
                     f'<tbody>{detection_rows}</tbody></table></details>'
                     '<details><summary>Detalle técnico interno: similitud DINOv2 y tiempos</summary>'
                     f'<p>SAM propuso {checked["valid_proposals"]} regiones válidas en total. Cada una se comparó con los diez CAD. '
                     'Ese número no indica cuántas piezas se detectaron ni cuántas veces apareció cada CAD.</p>'
                     '<p>DINOv2 es el descriptor interno de SAM-6D, distinto del proyecto DINOPDDL. La tabla siguiente es '
                     'una comparación semántica interna sobre una misma región, no el score final de SAM-6D. '
                     'El algoritmo original refina apariencia y geometría de la clase asignada; no calcula diez scores finales sobre cada máscara.</p>'
                     '<table><thead><tr><th>Orden</th><th>CAD</th><th>Nombre del catálogo</th><th>Similitud interna DINOv2</th>'
                     f'<th>Comparación CAD (ms)</th></tr></thead><tbody>{ranking}</tbody></table><p>{details}</p>'
                     f'<pre>{e(json.dumps(frame["proposal_counts"], ensure_ascii=False, indent=2))}</pre>'
                     f'<a href="{rel}/case.json">JSON del caso</a> · <a href="{rel}/detection_ism.json">Detecciones y máscaras RLE</a>'
                     '</details><p><a href="#inicio">Volver al resumen</a></p></article>')
    setup = run["setup_timing"]
    note = json.loads((target / audit.name).read_text()).get("note_es", "") if audit.is_file() else ""
    cad_gallery = "".join(f'<div><div class="cad-views"><img src="{e(cad_previews[p["key"]])}" alt="Vista oblicua CAD {e(p["key"])}" loading="lazy">'
                          f'<img src="{e(str(Path(cad_previews[p["key"]]).with_stem(p["key"] + "_frontal")))}" alt="Segunda vista CAD {e(p["key"])}" loading="lazy"></div>'
                          f'<strong>{e(p["key"])}</strong><small>{e(p["name"])}</small></div>'
                          for p in run["catalog"] if p["key"] in cad_previews)
    matrix_headers = "".join(f'<th title="{e(p["key"])}">{e(p["key"].split("_")[0])}<br><small>{e(p["name"])}</small></th>' for p in run["catalog"])
    matrix_title = "Scores de todos los CAD sobre la misma región" if dense_by_name else "Qué pieza eligió SAM-6D en cada foto"
    matrix_description = (
        'Ahora todas las celdas contienen una medición nueva. En cada foto se reutiliza la misma región principal, mostrada en el recorte, '
        'y se evalúan los diez CAD con la fórmula completa de SAM-6D, sin aplicar el filtro de score. '
        'Es una evaluación adicional de hipótesis; no reemplaza la predicción ni las detecciones originales.'
        if dense_by_name else 'Cada fila es una foto y cada columna un CAD. Los números son scores finales de las detecciones conservadas.')
    matrix_missing_note = (
        'Los huecos de la tabla anterior eran scores no calculados o descartados por el filtrado, no ceros. '
        'Esta vista adicional los sustituye por scores realmente calculados sobre el mismo recorte. '
        '<a href="scores_todos_cad_region_principal.csv">Descargar los 220 scores sin filtrar</a>.'
        if dense_by_name else 'Los diez CAD se compararon con las 22 imágenes. «—» no significa CAD omitido ni score cero: no quedó una detección guardada para esa clase.')
    page = f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM-6D · Confianza final por imagen</title><style>
body{{font:16px system-ui,sans-serif;background:#eef2f6;color:#16293a;max-width:1300px;margin:auto;padding:24px;line-height:1.5}}
article,.panel{{background:white;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:8px;border-bottom:1px solid #ccd5dd;text-align:left}}th{{background:#e9eff5}}a{{color:#075a9f}}
.pictures{{display:flex;gap:12px;flex-wrap:wrap}}.pictures img{{width:min(100%,600px);height:auto}}img.chart{{max-width:100%}}
pre{{white-space:pre-wrap}}details{{margin-top:14px}}.device{{font-size:24px;font-weight:bold}}h1{{margin-bottom:0}}
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
</style><body id="inicio"><h1>SAM-6D · Confianza final por imagen</h1>
<p class="device">Dispositivo real: {e(run["device"]["used"])} {e(run["device"].get("gpu_name") or "")}</p>
<p><a href="#comparacion-cads">Ver elegidas y alternativas por imagen · {coverage["verified_image_cad_pairs"]}/{coverage["expected_image_cad_pairs"]} comparaciones verificadas</a></p>
<div class="panel"><p>Estado: {e(run["status"])} · {stats["processed"]}/{stats["selected"]} capturas procesadas ·
{len(run["catalog"])} CAD × 42 vistas · {stats["recognized"]} imágenes con identificación aceptada.</p>
<p>Tiempo de proceso medio: <b>{fmt(stats["mean_processing_s"])} s</b> · mediana: {fmt(stats["median_processing_s"])} s.
Carga de modelos: {fmt(setup.get("model_load_s"))} s · preparación/carga de referencias: {fmt(setup.get("references_s"))} s.
Estos dos costes se pagan una vez por ejecución y se muestran aparte del tiempo por imagen.</p>
<p><b>La confianza mostrada es el score final de detección y segmentación de SAM-6D (ISM), en escala 0–1.</b>
No es una probabilidad calibrada ni una confianza de pose 6D. Una imagen puede contener varias piezas; el candidato principal es la detección de mayor score.
El umbral configurado para aceptar y dibujar es {run["settings"]["confidence"]:.2f}; se conservan también detecciones desde {run["settings"]["min_score"]:.2f}.</p>
<div class="provenance"><b>Origen de las etiquetas:</b> «Archivo original» es el nombre de la foto; «CAD predicho» es la elección del modelo.
El nombre semántico asociado al CAD procede del catálogo de piezas preparado, con las correcciones de nombres confirmadas por el usuario. No hay una etiqueta real verificada por foto y no se usa el nombre del archivo para clasificar.
<a href="origen_etiquetas.json">Consultar el mapeo de etiquetas</a>. {e(note)}</div>
<p><a href="confianza_sam6d_por_imagen.csv">CSV de confianza final por imagen</a> · <a href="confianza_sam6d_por_cad.csv">CSV de scores finales por CAD</a> ·
<a href="resultados_completos.json">JSON completo</a></p>
<table id="resumen-confianza"><thead><tr><th>Archivo original de la captura</th><th>CAD predicho / candidato</th><th>Confianza final SAM-6D</th><th>Estado</th><th>Proceso (s)</th>
</tr></thead><tbody>{''.join(tables)}</tbody></table></div>
<section class="panel" id="comparacion-cads"><h2>{matrix_title}</h2>
<p>{matrix_description} Pulsa la foto para ver el detalle.</p>
<div class="legend"><span class="score-winner">Verde: predicción original aceptada</span><span class="score-alternative">Naranja: otra hipótesis &gt; 0,40</span>
<span class="score-low">Gris: score ≤ 0,40</span><span>—: sin score final conservado</span></div>
<p>Una fila sin verde no tiene elección aceptada al umbral {run["settings"]["confidence"]:.2f}. El naranja no cambia ese umbral ni confirma que la pieza esté presente.</p>
<div class="matrix-scroll"><table id="matriz-comparaciones"><thead><tr><th>Archivo de la imagen</th>{matrix_headers}</tr></thead>
<tbody>{''.join(final_matrix_rows)}</tbody></table></div>
<p>{matrix_missing_note}</p>
<details><summary>Ver los diez CAD con vistas reconocibles</summary><p>Dos vistas de cada referencia, ampliadas y sin modificar su geometría.</p><div class="cad-gallery">{cad_gallery}</div></details>
<details><summary>Auditoría: comparación interna de todos los CAD ({coverage["verified_image_cad_pairs"]}/{coverage["expected_image_cad_pairs"]})</summary>
<p>Esta segunda matriz muestra la similitud interna sobre la región principal. No es el score final y no debe compararse numéricamente con la tabla anterior.</p>
<div class="matrix-scroll"><table id="matriz-interna"><thead><tr><th>Imagen</th>{matrix_headers}</tr></thead><tbody>{''.join(coverage_rows)}</tbody></table></div>
<p><a href="cobertura_comparaciones.json">Ver cobertura</a> · <a href="comparacion_todos_los_cad.csv">Descargar comparaciones internas</a></p></details></section>
<details class="panel"><summary>Tiempos por etapa y comparaciones internas</summary>
<p>DINOv2 forma parte del descriptor interno de SAM-6D; estos valores no proceden del clasificador independiente de DINOPDDL.
La confianza final combina similitud semántica, apariencia y coherencia geométrica. Las puntuaciones internas se muestran por separado dentro de cada captura.</p>
<p>Una pasada por imagen, sin calentamiento descartado. Referencias recuperadas de caché:
{sum(p["cache_hit"] for p in run.get("references", {}).get("parts", []))}/{len(run["catalog"])} CAD.
PyTorch: {e(run["device"]["torch"])} · CUDA: {e(run["device"].get("torch_cuda_build"))}.</p>
<p><a href="tiempos_por_imagen.csv">CSV de tiempos</a> · <a href="comparacion_todos_los_cad.csv">CSV de similitudes internas DINOv2 (no score final)</a> ·
<a href="resumen.json">Resumen numérico</a></p><img class="chart" src="tiempos_por_imagen.png" alt="Tiempos por etapa"></details>{''.join(cards)}
<p>Extrae el ZIP completo y abre index.html en tu navegador. El informe funciona sin conexión y contiene RGB, profundidad original de 16 bits,
calibración, máscaras, scores y mediciones. No incluye los pesos de los modelos.</p></body></html>'''
    (target / "index.html").write_text(page, encoding="utf-8")
    # Retirar la entrada anterior al regenerar una exportación de esta herramienta.
    legacy = target / "ABRIR_EN_WINDOWS.html"
    if legacy.is_file():
        legacy.unlink()
    bundle = target.with_suffix(".zip")
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, Path(target.name) / path.relative_to(target))
    temporary.replace(bundle)
    return bundle
