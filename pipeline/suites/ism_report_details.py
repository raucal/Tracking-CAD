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
            figures.append(f'<figure id="ref-{html.escape(key)}-{view}"><a href="{url}"><img src="{url}" loading="lazy" alt="CAD {html.escape(key)}, view {view}"></a><figcaption>View {view}</figcaption></figure>')
        count = len(figures)
        if not figures:
            figures.append('<p>Rendered references are unavailable in the dataset for this run.</p>')
        blocks.append(f'<details><summary>{html.escape(key)} · {html.escape(part["name"])} · {count} views</summary><div class="view-grid">'+''.join(figures)+'</div></details>')
    return '<details class="panel" id="referencias"><summary>Rendered views of all CADs</summary><p>References available in the dataset at export time, used by the additional DINOv2 audit. Indices start at 0. Click an image to enlarge it.</p>'+''.join(blocks)+'</details>'


def methodology(audit_note=""):
    return '''<details class="panel" id="metodo-ism"><summary>How ISM works: semantics, appearance and geometry</summary>
<details><summary>1. DINOv2 and per-view semantic scores</summary><p>SAM generates masks. DINOv2 extracts a global CLS descriptor from each crop and rendered CAD view. Cosine similarities are clipped to [0,1]. The highest similarity selects the view; the CAD semantic score uses the configured aggregation (avg_5: mean of the five highest scores). The winning-view score can therefore differ from the aggregated semantic score used for class assignment.</p></details>
<details><summary>2. Per-view appearance and the view used by ISM</summary><p>DINOv2 also produces local patch descriptors. Each valid query patch is matched to the reference patch with the highest dot product; these maxima are averaged and clipped to [0,1]. Patches need not occupy the same position. ISM computes appearance on the semantic winner. The appearance winner shown by the audit is an additional comparison across all views and does not replace the original selection.</p></details>
<details><summary>3. Geometry step by step: projection and bounding boxes</summary><ol>
<li>Take CAD points sampled in metres (2048 in this integration) and the rotation of the best semantic view.</li>
<li>Use positive-depth pixels inside the SAM mask. Convert to metres: Z = depth × depth_scale / 1000; X = (u − cx) Z / fx; Y = (v − cy) Z / fy. Their mean gives the approximate translation t.</li>
<li>Transform CAD points into camera coordinates with R × point + t. Project with K: u = fx X/Z + cx, v = fy Y/Z + cy. The implementation uses integer pixels and clips coordinates to image bounds.</li>
<li>Minima and maxima of projected points form a rectangular bounding box. Compare it with the SAM proposal box: geometry = intersection area / union area (IoU). Measured boxes and areas are shown per CAD in each case.</li></ol>
<p>This score measures 2D bounding-box agreement. It does not measure distances between 3D surfaces, silhouette IoU, or an ICP pose fit. Different shapes can have similar boxes. Incorrect depth or scale changes projection. Rotation comes from a discrete view; PEM estimates a 6D pose later and separately.</p>
<p>The original compute_iou checks intersection over the entire batch: if one pair does not overlap on both axes, it can return zero for the full batch. Historical diagnostics preserve that model output and expose areas for inspection. The new identification evaluator uses independent per-pair IoU.</p></details>
<details><summary>4. Visibility and final formula</summary><p>Visibility is estimated through appearance: for each reference patch, find its highest query similarity, then count the fraction above visible_thred among patches with nonzero similarity. This is not a depth-based occlusion test. The effective threshold is included in the audit.</p>
<p><code>ISM = (semantic + appearance + geometry × visibility) / (2 + visibility)</code>. Geometry × visibility and the reconstructed score are shown. Low visibility reduces geometric weight. None of these scores is a calibrated probability.</p></details>
<p>Per-view tables are additional audits of the same saved primary mask (mask_000.png), for all CADs. They do not represent every SAM proposal or change the original detections. Missing historical measurements are explicitly marked.</p>
''' + audit_note + '</details>'


def view_audit_section(frame, relative=""):
    if not frame or frame.get('status') != 'scored':
        return '<details><summary>ISM per view: measurements unavailable</summary><p>This capture has no per-view audit. Generate measurements with <code>bash run.sh scores --run SUITE_DIRECTORY --view-details</code>, then regenerate the report. Saved masks are reused.</p></details>'
    blocks = []
    for item in frame['scores']:
        key = html.escape(item['part_key'])
        rows = []
        sw, aw = item['semantic_winner'], item['appearance_winner']
        for i, (s, a) in enumerate(zip(item['semantic_by_view'], item['appearance_by_view'])):
            url = quote(f'referencias_cad/{item["part_key"]}/rgb_{i}.png', safe='/')
            rows.append(f'<tr><td><a href="{url}"><img class="view-thumb" src="{url}" loading="lazy" alt="View {i}"></a> {i}</td><td>{s:.6f}{" · Semantic winner / used by ISM" if i == sw else ""}</td><td>{a:.6f}{" · Appearance winner" if i == aw else ""}</td></tr>')
        g = item['geometry_details']
        overlay = ('<figure><img style="max-width:100%" loading="lazy" src="'+relative+'/'+quote(g['overlay'], safe='/')+'" alt="Geometric bounding-box comparison"><figcaption>Blue: projected CAD bounding box. Orange: SAM proposal bounding box.</figcaption></figure>') if g.get('overlay') else ''
        f = lambda value: html.escape(json.dumps(value, ensure_ascii=False))
        blocks.append(f'''<details><summary>{key} · ISM view {sw} · appearance winner {aw}</summary>
<p>Aggregated semantic score: {item['semantic']:.6f} · Appearance used: {item['appearance']:.6f} · Geometry: {item['geometry']:.6f} · Visibility: {item['visible_ratio']:.6f} · Geometry contribution: {item['geometry']*item['visible_ratio']:.6f} · ISM: {item['score']:.6f}.</p>
<details><summary>Per-view DINOv2 and appearance breakdown</summary><table><thead><tr><th>Rendered view</th><th>DINOv2 semantic similarity</th><th>Appearance score</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details>
<details><summary>Measured geometry for ISM view {sw}</summary>
{overlay}<p>Bounding boxes [x1, y1, x2, y2] in pixels. Projected: {f(g['projected_box_xyxy_px'])}; SAM proposal: {f(g['proposal_box_xyxy_px'])}.</p>
<p>Intersection: {g['intersection_px2']:.3f} px² · Union: {g['union_px2']:.3f} px² · Model-returned IoU: {item['geometry']:.6f}.</p>
<p>Approximate translation [X, Y, Z]: {f(g['translation_m'])} m · Projected points: {g['point_count']} · Visibility threshold: {g['visible_threshold']}.</p></details></details>''')
    delta = frame.get('original_score_delta')
    comparison = (f'<p>Saved original score: {frame["original_score"]:.6f} · Absolute difference of the recomputed class score: {delta:.6f}. '+('Matches within tolerance 0.001.' if delta <= .001 else 'Recomputed results differ from the original; check the device, version and references in JSON. These measurements are additional diagnostics and do not replace original scores.')+'</p>') if delta is not None else ''
    return '<details class="diagnostic"><summary>Per-view ISM and geometry audit · same primary region</summary><p>Additional results on mask_000.png; semantic and appearance winners are independent. The original formula uses the semantic view.</p>'+comparison+''.join(blocks)+'</details>'
