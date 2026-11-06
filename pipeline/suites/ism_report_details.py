"""Offline reference gallery and measured ISM audit, separate from original predictions."""
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote


def reference_gallery(catalog, parts, base, target):
    blocks = []
    for part in catalog:
        key = part['key']
        folder = base / parts[key]['sam_template_dir']
        figures = []
        for source in sorted(folder.glob('rgb_*.png'), key=lambda p: int(p.stem.split('_')[-1])):
            view = int(source.stem.split('_')[-1])
            relative = Path('referencias_cad') / key / source.name
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / relative)
            url = quote(relative.as_posix(), safe='/')
            figures.append(f'<figure id="ref-{html.escape(key)}-{view}"><a href="{url}"><img src="{url}" loading="lazy" alt="CAD {html.escape(key)}, vista {view}"></a><figcaption>Vista {view}</figcaption></figure>')
        count = len(figures)
        if not figures:
            figures.append('<p>Referencias renderizadas no disponibles en el dataset de esta ejecución.</p>')
        blocks.append(f'<details><summary>{html.escape(key)} · {html.escape(part["name"])} · {count} vistas</summary><div class="view-grid">'+''.join(figures)+'</div></details>')
    return '<details class="panel" id="referencias"><summary>Vistas renderizadas de todos los CAD</summary><p>Referencias disponibles en el dataset al exportar, usadas en la nueva auditoría DINOv2. Índices desde 0. Pulsa una imagen para ampliarla.</p>'+''.join(blocks)+'</details>'


def methodology(audit_note=""):
    return '''<details class="panel" id="metodo-ism"><summary>Cómo funciona ISM: semántica, apariencia y geometría</summary>
<details><summary>1. DINOv2 y score semántico por vista</summary><p>SAM genera máscaras. DINOv2 extrae un descriptor global (token CLS) de cada recorte y de cada vista renderizada del CAD. Se compara su similitud coseno, limitada a [0, 1]. La mayor similitud selecciona la vista; el score semántico del CAD se obtiene con la agregación configurada (avg_5: media de las cinco mejores vistas). Por tanto, el score de la vista ganadora puede diferir del score semántico agregado. La clase se asigna usando ese score agregado.</p></details>
<details><summary>2. Apariencia por vista y vista utilizada por ISM</summary><p>DINOv2 también produce descriptores locales de patches. Para cada patch válido del recorte se busca el patch de referencia con mayor producto escalar; se promedian esos máximos y se limita el resultado a [0, 1]. No se exige que los patches tengan la misma posición. ISM calcula apariencia sobre la vista ganadora semántica. La ganadora por apariencia que muestra la auditoría es una comparación adicional entre todas las vistas: no sustituye la selección original.</p></details>
<details><summary>3. Geometría paso a paso: proyección y cajas</summary><ol>
<li>Se toman los puntos muestreados del CAD en metros (2048 en esta integración) y la rotación de la vista ganadora semántica.</li>
<li>Dentro de la máscara SAM se usan los píxeles con profundidad positiva. Se convierte la profundidad a metros: Z = depth × depth_scale / 1000; X = (u − cx) Z / fx; Y = (v − cy) Z / fy. La media de esos puntos proporciona la traslación aproximada t.</li>
<li>Los puntos del CAD pasan a cámara con R × punto + t. Se proyectan con la calibración K: u = fx X/Z + cx, v = fy Y/Z + cy. La implementación convierte a píxeles enteros y limita las coordenadas a los bordes de la imagen.</li>
<li>Los mínimos y máximos de los puntos proyectados forman una caja rectangular. Se compara con la caja de la propuesta SAM: geometría = área de intersección / área de unión (IoU). Las cajas y áreas medidas se muestran por CAD en el detalle de cada captura.</li></ol>
<p>Este score mide coincidencia de cajas 2D. No compara distancias entre dos superficies 3D, no calcula IoU de siluetas y no ajusta una pose mediante ICP. Dos formas distintas pueden tener cajas parecidas. Una profundidad o escala incorrecta puede alterar la proyección. La rotación procede de una vista discreta; la pose 6D PEM se calcula después y por separado.</p>
<p>La rutina original compute_iou evalúa la condición de intersección sobre todo el lote: si alguna caja no solapa en ambos ejes puede devolver cero para el lote completo. La auditoría conserva el resultado del modelo y expone las áreas para poder revisar esta limitación.</p></details>
<details><summary>4. Visibilidad y fórmula final</summary><p>La visibilidad se estima mediante apariencia: para cada patch de referencia se busca su mayor similitud con el recorte; se cuenta qué fracción supera el umbral visible_thred entre los que tienen similitud no nula. No es una comprobación de oclusiones mediante profundidad. El umbral efectivo aparece en la auditoría.</p>
<p><code>ISM = (semántica + apariencia + geometría × visibilidad) / (2 + visibilidad)</code>. Se muestran el producto geometría × visibilidad y el score reconstruido. Una visibilidad baja reduce el peso geométrico. Ninguno de estos scores es una probabilidad calibrada.</p></details>
<p>Las tablas por vista son una auditoría adicional sobre la misma máscara primaria guardada (mask_000.png) de cada captura, para todos los CAD. No representan todas las propuestas SAM ni cambian las detecciones originales. Si faltan mediciones de una ejecución antigua, se indica explícitamente. </p>''' + audit_note + '</details>'


def view_audit_section(frame, relative=""):
    if not frame or frame.get('status') != 'scored':
        return '<details><summary>ISM por vista: mediciones no disponibles</summary><p>Esta captura no tiene auditoría por vista. Genera las mediciones con <code>bash run.sh scores --run CARPETA_SUITE --view-details</code> y regenera el informe. Se reutilizan las máscaras guardadas.</p></details>'
    blocks = []
    for item in frame['scores']:
        key = html.escape(item['part_key'])
        rows = []
        sw, aw = item['semantic_winner'], item['appearance_winner']
        for i, (s, a) in enumerate(zip(item['semantic_by_view'], item['appearance_by_view'])):
            url = quote(f'referencias_cad/{item["part_key"]}/rgb_{i}.png', safe='/')
            rows.append(f'<tr><td><a href="{url}"><img class="view-thumb" src="{url}" loading="lazy" alt="Vista {i}"></a> {i}</td><td>{s:.6f}{" · Ganadora semántica / usada por ISM" if i == sw else ""}</td><td>{a:.6f}{" · Ganadora apariencia" if i == aw else ""}</td></tr>')
        g = item['geometry_details']
        overlay = ('<figure><img style="max-width:100%" loading="lazy" src="'+relative+'/'+quote(g['overlay'], safe='/')+'" alt="Comparación de cajas geométricas"><figcaption>Azul: caja del CAD proyectado. Naranja: caja de la propuesta SAM.</figcaption></figure>') if g.get('overlay') else ''
        f = lambda value: html.escape(json.dumps(value, ensure_ascii=False))
        blocks.append(f'''<details><summary>{key} · vista ISM {sw} · ganadora apariencia {aw}</summary>
<p>Semántica agregada: {item['semantic']:.6f} · Apariencia usada: {item['appearance']:.6f} · Geometría: {item['geometry']:.6f} · Visibilidad: {item['visible_ratio']:.6f} · Aporte geométrico: {item['geometry']*item['visible_ratio']:.6f} · ISM: {item['score']:.6f}.</p>
<details><summary>Desglose DINOv2 y apariencia de cada vista</summary><table><thead><tr><th>Vista renderizada</th><th>Similitud DINOv2 semántica</th><th>Score apariencia</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details>
<details><summary>Detalle geométrico medido de la vista ISM {sw}</summary>
{overlay}<p>Cajas [x1, y1, x2, y2] en píxeles. Proyectada: {f(g['projected_box_xyxy_px'])}; propuesta SAM: {f(g['proposal_box_xyxy_px'])}.</p>
<p>Intersección: {g['intersection_px2']:.3f} px² · Unión: {g['union_px2']:.3f} px² · IoU devuelta por el modelo: {item['geometry']:.6f}.</p>
<p>Traslación aproximada [X, Y, Z]: {f(g['translation_m'])} m · Puntos proyectados: {g['point_count']} · Umbral de visibilidad: {g['visible_threshold']}.</p></details></details>''')
    delta = frame.get('original_score_delta')
    comparison = (f'<p>Score original guardado: {frame["original_score"]:.6f} · Diferencia absoluta del score recalculado para esa clase: {delta:.6f}. '+('Coincide con tolerancia 0,001.' if delta <= .001 else 'Los resultados recalculados difieren del original; consulta dispositivo, versión y referencias en el JSON. Estas mediciones se muestran como auditoría adicional y no reemplazan los scores originales.')+'</p>') if delta is not None else ''
    return '<details class="diagnostic"><summary>Auditoría ISM por vista y geometría · misma región primaria</summary><p>Resultados adicionales sobre mask_000.png; ganadoras independientes para semántica y apariencia. La fórmula original usa la vista semántica.</p>'+comparison+''.join(blocks)+'</details>'
