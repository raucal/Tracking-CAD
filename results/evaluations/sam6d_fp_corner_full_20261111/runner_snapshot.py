"""Fresh, label-blind SAM6D identification followed by FoundationPose registration.

Run using assembly-runtime Python. FoundationPose runs in its configured venv.
Never overwrites a run; --frame selects a smoke case; omit it for the full manifest.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from settings import ISM, PREPARED, ROOT, path
from region_filters import reject_vertical_option_a_corner, rank_candidates, CORNER_FILTER_REASON


def write_json(p, value):
    p = Path(p)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_iou(a, b):
    """Each row is independent, including empty intersections and degenerate boxes."""
    import torch
    wh = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0)
    intersection = wh.prod(dim=1)
    union = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(dim=1) + (b[:, 2:] - b[:, :2]).clamp(min=0).prod(dim=1) - intersection
    return torch.where(union > 0, intersection / union.clamp(min=1e-12), torch.zeros_like(union))


def prediction_input(frame):
    # Deliberate allowlist: no name, expected CAD, filename hint or source path.
    return {k: frame[k] for k in ('frame_id', 'rgb', 'depth', 'camera')}


def validate_dataset(dataset, base):
    import numpy as np
    from PIL import Image
    import trimesh
    parts, frames = dataset['parts'], dataset['frames']
    for group, key in ((parts, 'key'), (parts, 'category_id'), (frames, 'name'), (frames, 'frame_id')):
        if len({x[key] for x in group}) != len(group):
            raise ValueError('Duplicate manifest identifier: ' + key)
    keys = {p['key'] for p in parts}
    if not keys or not frames:
        raise ValueError('Empty catalog or frame list')
    part_checks = []
    for p in parts:
        mesh_path = base / p['sam_ply']
        mesh = trimesh.load_mesh(mesh_path, process=False)
        T = np.asarray(p['T_local_from_sam_mm'], dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]) or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1):
            raise ValueError('Invalid rigid CAD transform: ' + p['key'])
        placed = mesh.copy(); placed.apply_transform(T)
        if not np.allclose(placed.bounds, p['bounds_local_mm'], atol=.05):
            raise ValueError('CAD units/origin disagree with manifest: ' + p['key'])
        templates = base / p['sam_template_dir']
        template_files = [templates / f'{prefix}_{i}.png' for i in range(42) for prefix in ('rgb', 'mask')]
        if not all(f.is_file() for f in template_files):
            raise ValueError('Missing templates: ' + p['key'])
        part_checks.append({'key':p['key'], 'category_id':p['category_id'], 'mesh_sha256':digest(mesh_path),
                            'mesh_bounds_mm':mesh.bounds.tolist(), 'rigid_transform_and_local_bounds':'passed',
                            'templates_checked':len(template_files)})
    checks = []
    for f in frames:
        row = {'frame_id':f['frame_id'], 'name':f['name'], 'status':'valid', 'errors':[]}
        try:
            if f.get('expected_cad_key') not in keys:
                raise ValueError('Missing/unknown expected_cad_key; cannot evaluate identification')
            with Image.open(base / f['rgb']) as im: rgb = np.asarray(im.convert('RGB'))
            with Image.open(base / f['depth']) as im: depth = np.asarray(im)
            if rgb.shape[:2] != depth.shape or depth.ndim != 2 or depth.dtype.kind not in 'ui' or not (depth>0).any():
                raise ValueError('RGB/depth shape, dtype or valid-depth mismatch')
            cam = json.loads((base / f['camera']).read_text()); K = np.asarray(cam['cam_K']).reshape(3,3)
            if not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]) or not 0 < float(cam['depth_scale']) < 1000:
                raise ValueError('Invalid intrinsics/depth_scale')
            row.update(shape=list(depth.shape), depth_unit='raw * depth_scale millimetres',
                       valid_depth_fraction=float((depth>0).mean()), hashes={k:digest(base/f[k]) for k in ('rgb','depth','camera')})
            if f.get('prepared_rgb_sha256') and row['hashes']['rgb'] != f['prepared_rgb_sha256']:
                raise ValueError('Prepared RGB hash differs from manifest')
        except Exception as e:
            row.update(status='invalid', errors=[str(e)])
        checks.append(row)
    return {'parts':part_checks, 'frames':checks, 'total':len(frames), 'valid':sum(c['status']=='valid' for c in checks)}


def command_record(command, cwd=ROOT):
    p = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {'command':shlex.join(map(str,command)), 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}


def scores_on_mask(model, rgb, depth, camera, mask, bbox, parts, device, corner_filter=True):
    import numpy as np
    import torch
    from model.utils import Detections
    from run_suite_pddl import compare_all, measured
    from tools.score_all_cads import score_hypotheses
    timings = {}
    detection = Detections({'boxes':torch.tensor([bbox], device=device, dtype=torch.float32),
                            'masks':torch.tensor(mask.astype(np.float32), device=device)[None]})
    batch = {'depth':torch.tensor(depth.astype(np.int32), device=device)[None],
             'cam_intrinsic':torch.tensor(camera['cam_K'], device=device,dtype=torch.float32).reshape(1,3,3),
             'depth_scale':torch.tensor([camera['depth_scale']], device=device)}
    with measured(timings, 'ranking_query_features_s', device):
        query, patches = model.descriptor_model(np.asarray(rgb), detection)
    with measured(timings, 'ranking_all_cads_s', device):
        semantic, views, _ = compare_all(model, query, device)
        values = score_hypotheses(model, detection, patches, batch, semantic, views)
        values = {k:v.cpu().tolist() for k,v in values.items()}
    rows = [{'cad_key':p['key'], 'category_id':p['category_id'], 'status':'scored', 'error':None,
             'best_template':int(views[0,i]), **{k:v[i] for k,v in values.items()}} for i,p in enumerate(parts)]
    x1,y1,x2,y2 = bbox
    rows=rank_candidates(rows,[x1,y1,x2-x1,y2-y1],rgb.size,corner_filter)
    for row in rows:
        row.update(foundationpose_score=None, foundationpose_status='not_evaluated',
                   foundationpose_reason=row['filter_reason'] or 'Only ISM winner registered; FP does not select CAD')
    return rows, timings


def mesh_overlay(folder, mesh_path, pose):
    """Actual mesh edges projected on the input, in addition to FP's box/axes view."""
    import cv2
    import numpy as np
    from PIL import Image
    import trimesh
    mesh = trimesh.load_mesh(mesh_path, process=False)
    T = np.asarray(pose['T_camera_from_model_mm'])
    K = np.asarray(json.loads((folder/'camera.json').read_text())['cam_K']).reshape(3,3)
    points = np.asarray(mesh.vertices) @ T[:3,:3].T + T[:3,3]
    projected = points @ K.T
    uv = projected[:,:2] / projected[:,2:]
    image = np.array(Image.open(folder/'input_rgb.png').convert('RGB'))
    drawn = image.copy()
    for a,b in mesh.edges_unique:
        if min(points[a,2],points[b,2])>1 and np.isfinite(uv[[a,b]]).all() and np.abs(uv[[a,b]]).max()<1e6:
            cv2.line(drawn,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),(20,240,130),1,cv2.LINE_AA)
    Image.fromarray(cv2.addWeighted(image,.35,drawn,.65,0)).save(folder/'cad_mesh_overlay.png')


def predict(model, item, parts, base, folder, device, worker, confidence, corner_filter=True):
    import numpy as np
    import torch
    from PIL import Image
    from run_suite_pddl import process_frame
    from sam6d_ui.backends.foundationpose import run as fp_run
    started = time.perf_counter()
    # item contains only allowlisted inference fields. Ground truth is joined later.
    import run_suite_pddl as ism_worker
    rejected=[]
    def reject(part_key,bbox,image_size):
        excluded=corner_filter and reject_vertical_option_a_corner(part_key,bbox,image_size)
        if excluded:rejected.append({'cad_key':part_key,'bbox':bbox,'reason':CORNER_FILTER_REASON})
        return excluded
    ism_worker.reject_repeated_region=reject
    case, rgb, detections, masks, _ = process_frame(model,dict(item,name=str(item['frame_id'])),parts,base,device,0.,confidence)
    case['region_filter']={'enabled':corner_filter,'rejected_detections':rejected}
    folder.mkdir(parents=True)
    rgb.save(folder/'input_rgb.png')
    shutil.copy2(base/item['depth'],folder/'depth.png'); shutil.copy2(base/item['camera'],folder/'camera.json')
    write_json(folder/'ism_case.json',case)
    write_json(folder/'detections.json',detections)
    prediction = {'status':'failed', 'error':None, 'timing':dict(case['timing']), 'scores':[], 'pose':None,
                  'original_assignment_winner':case['candidate']['part_key'] if case['candidate'] else None,
                  'predicted_cad':None, 'semantic_winner':None, 'margin':None, 'region_filter':case['region_filter']}
    if not masks:
        prediction.update(error='No retained SAM6D proposal; no mask available for ranking',timing={**prediction['timing'],'total_s':time.perf_counter()-started})
        return prediction
    mask = masks[0]
    Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask_000.png')
    overlay=np.array(rgb).copy();overlay[mask]=(overlay[mask]*.6 + np.array([40,210,100])*.4).astype(np.uint8)
    Image.fromarray(overlay).save(folder/'mask_overlay.png')
    with Image.open(folder/'depth.png') as im:depth=np.asarray(im)
    camera=json.loads((folder/'camera.json').read_text())
    x,y,w,h=case['candidate']['bbox']
    rows,timings=scores_on_mask(model,rgb,depth,camera,mask,[x,y,x+w,y+h],parts,device,corner_filter)
    prediction['timing'].update(timings)
    eligible=[r for r in rows if r['eligible']]
    winner=eligible[0];part=next(p for p in parts if p['key']==winner['cad_key'])
    prediction.update(status='predicted',scores=rows,predicted_cad=winner['cad_key'],
                      semantic_winner=min(eligible,key=lambda r:r['semantic_rank'])['cad_key'],
                      margin=winner['score']-eligible[1]['score'] if len(eligible)>1 else None,
                      primary_mask_sha256=digest(folder/'mask_000.png'))
    write_json(folder/'prediction_before_pose.json',prediction)
    pose_started=time.perf_counter()
    try:
        pose=fp_run(folder,base/part['sam_ply'],0,np.linalg.inv(np.asarray(part['T_local_from_sam_mm'])))
        if not pose.get('poses'):raise RuntimeError('FoundationPose produced no pose')
        prediction['pose']=pose
        winner.update(foundationpose_status='estimated',foundationpose_reason=None,foundationpose_score=pose['poses'][0]['pose_score'])
        mesh_overlay(folder,base/part['sam_ply'],pose['poses'][0])
    except Exception as e:
        winner.update(foundationpose_status='failed',foundationpose_reason=str(e))
        prediction.update(status='pose_failed',error=str(e))
        write_json(folder/'pose_error.json',{'error':str(e),'traceback':traceback.format_exc()})
    prediction['timing']['foundationpose_pipeline_s']=time.perf_counter()-pose_started
    prediction['timing']['total_s']=time.perf_counter()-started
    return prediction


def save_csv(p, rows, fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


def stats(values):
    if not values:return {'count':0,'mean':None,'median':None,'p95':None,'sum':None}
    import numpy as np
    return {'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'p95':float(np.percentile(values,95)),'sum':sum(values)}


def export(run, output):
    frames=run['frames'];keys=[p['key'] for p in run['catalog']]
    predicted=[f for f in frames if f.get('predicted_cad') is not None]
    matches=sum(f['correct'] is True for f in predicted)
    summary={'total_manifest':run['validation']['total'],'valid_manifest':run['validation']['valid'],
             'selected':len(frames),'processed_identification':len(predicted),'complete_with_pose':sum(f['status']=='predicted' for f in frames),
             'failed_identification':sum(f['status']=='failed' for f in frames), 'failed_pose':sum(f['status']=='pose_failed' for f in frames),
             'invalid':sum(f['status']=='invalid' for f in frames),'omitted':run['validation']['total']-len(frames),
             'matches':matches,'top1_on_predictions':matches/len(predicted) if predicted else None,
             'top1_on_selected':matches/len(frames) if frames else None,
             'failure_reasons':[{'name':f['name'],'status':f['status'],'error':f.get('error')} for f in frames if f['status']!='predicted'],
             'topk':{},'latency':{},'pose_accuracy_measured':False,
             'pose_accuracy_limitation':'No pose ground truth or symmetry annotations in manifest; registration success is not accuracy.',
             'wall_s':run.get('wall_s')}
    for k in sorted({1,min(3,len(keys)),min(5,len(keys))}):
        hit=sum(f['expected_cad'] in [r['cad_key'] for r in f['scores'] if r.get('eligible',True)][:k] for f in predicted)
        summary['topk'][str(k)]={'matches':hit,'denominator_predictions':len(predicted),'accuracy':hit/len(predicted) if predicted else None}
    for stage in sorted({s for f in frames for s in f.get('timing',{})}):
        summary['latency'][stage]=stats([f['timing'][stage] for f in frames if stage in f.get('timing',{})])
    summary['semantic_to_final_changed']=sum(f['semantic_winner']!=f['predicted_cad'] for f in predicted)
    summary['assignment_to_final_changed']=sum(f['original_assignment_winner']!=f['predicted_cad'] for f in predicted)
    summary['semantic_correct']=sum(f['semantic_winner']==f['expected_cad'] for f in predicted)
    summary['per_cad']=[]
    confusion=[]
    for key in keys:
        support=[f for f in frames if f['expected_cad']==key];selected=[f for f in predicted if f['predicted_cad']==key]
        tp=sum(f['correct'] is True for f in selected)
        summary['per_cad'].append({'cad_key':key,'support':len(support),'predictions':len(selected),'correct':tp,
                                   'recall_on_selected':tp/len(support) if support else None,'precision':tp/len(selected) if selected else None})
        confusion.append({'expected_cad':key,**{p:sum(f['expected_cad']==key and f.get('predicted_cad')==p for f in frames) for p in keys},
                          'NO_PREDICTION':sum(f['expected_cad']==key and not f.get('predicted_cad') for f in frames)})
    summary['confused_pairs']=sorted([{'expected':r['expected_cad'],'predicted':k,'count':r[k]} for r in confusion for k in keys if k!=r['expected_cad'] and r[k]],key=lambda r:-r['count'])
    summary['wrong_high_scores']=sorted([{'name':f['name'],'expected':f['expected_cad'],'predicted':f['predicted_cad'],
                                        'score':f['scores'][0]['score'],'margin':f['margin']} for f in predicted if not f['correct']],key=lambda r:-r['score'])
    summary['margins_correct']=stats([f['margin'] for f in predicted if f['correct'] and f['margin'] is not None])
    summary['margins_wrong']=stats([f['margin'] for f in predicted if not f['correct'] and f['margin'] is not None])
    wide=[];long=[]
    for f in frames:
        row={k:f.get(k) for k in ('frame_id','name','rgb_path','expected_cad','predicted_cad','correct','status','error','margin','semantic_winner','original_assignment_winner')}
        row['total_s']=f.get('timing',{}).get('total_s')
        by={r['cad_key']:r for r in f['scores']}
        for key in keys:
            r=by.get(key,{})
            row[key+'__eligible']=r.get('eligible')
            row[key+'__filter_reason']=r.get('filter_reason')
            for score in ('score','semantic','appearance','geometry','visible_ratio','foundationpose_score'):
                row[key+'__'+score]=r.get(score)
            item={'frame_id':f['frame_id'],'name':f['name'],'rgb_path':f['rgb_path'],'expected_cad':f['expected_cad'],
                  'predicted_cad':f.get('predicted_cad'),'correct':f.get('correct'),'cad_key':key,
                  **r,'error':r.get('error',f.get('error'))}
            if not r:item.update(status='not_evaluated',foundationpose_status='not_evaluated',foundationpose_reason=f.get('error'))
            long.append(item)
        wide.append(row)
    save_csv(output/'scores_wide.csv',wide);save_csv(output/'scores_long.csv',long)
    save_csv(output/'confusion_matrix.csv',confusion);save_csv(output/'per_cad.csv',summary['per_cad'])
    save_csv(output/'latency.csv',[{'name':f['name'],**f.get('timing',{})} for f in frames])
    write_json(output/'scores_wide.json',wide);write_json(output/'scores_long.json',long);write_json(output/'summary.json',summary)
    run['summary']=summary;write_json(output/'run.json',run)
    report=[f"# Evaluación SAM6D + FoundationPose\n\n{matches} de {len(predicted)} predicciones coinciden con el CAD del manifest.",
            f"Manifest: {summary['total_manifest']}; válidos: {summary['valid_manifest']}; seleccionados: {len(frames)}; identificación: {len(predicted)}; pose completada: {summary['complete_with_pose']}.",
            'Protocolo: máscaras SAM nuevas; se aplica el filtro documentado en settings.region_filter al seleccionar la región y al decidir el CAD final. Todos los CAD conservan sus puntuaciones medidas en la misma región; los excluidos tienen eligible=false, filter_reason y rank=null. Gana el mayor score elegible (desempate: category_id). El margen y top-k usan sólo candidatos elegibles. FoundationPose registra únicamente al ganador. El CAD esperado se une después de predecir.',
            'Score ISM = (semantic + appearance + geometry * visible_ratio) / (2 + visible_ratio). Semantic: media de los 5 cosenos mayores entre 42 vistas; appearance: similitud de patches en la vista de mayor coseno; geometry: IoU de cajas 2D (no ajuste 6D); visible_ratio: fracción de patches de referencia compatibles. Semantic y appearance son cosenos/similitudes recortados a [0,1]; IoU y visible_ratio [0,1]; no son probabilidades. FoundationPose: salida de su red de ranking de hipótesis, sin calibración probabilística ni escala comparable con ISM.',
            'Pose: T_camera_from_model_mm y T_camera_from_input_cad_mm; cámara OpenCV X derecha, Y abajo, Z hacia delante; traslación mm. FP recibe malla y profundidad en metros y convierte la salida a mm. T_camera_from_input = T_camera_from_model @ inv(T_local_from_sam_mm). No hay ground truth de pose ni simetrías anotadas; no se mide precisión de pose.',
            'Caché: sólo referencias CAD con firmas/hash verificadas; hits y claves en run.json. SAM, ranking y pose se ejecutan de nuevo. Semilla 0; muestreo por category_id. La primera imagen incluye costes fríos. Otros procesos GPU pueden influir en tiempos. Reproducibilidad numérica exacta en CUDA no garantizada.',
            'Primera suite: results/reports/sam6d_suite_20261102T150131Z/confianza_sam6d_por_cad.csv mostraba máximos por CAD entre detecciones asignadas, con huecos para clases no asignadas. No es un ranking completo en una máscara. Su pose es PEM, y el IoU y el protocolo difieren: no se atribuye una mejora causal ni se comparan directamente sus latencias.',
            'Artefactos: scores_wide.csv/json (tabla completa), scores_long.csv/json (imagen × CAD), run.json (poses, tiempos, rankings y procedencia), summary.json, confusion_matrix.csv, per_cad.csv, latency.csv, cases/<frame_id>/mask_overlay.png, pose_overlay.png y cad_mesh_overlay.png.',
            '## Métricas\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```',
            '## Repetición\n\n```bash\n'+run['command']+'\n```']
    (output/'report.md').write_text('\n\n'.join(report)+'\n', encoding='utf-8')
    h=['<!doctype html><meta charset="utf-8"><title>Evaluación CAD</title><style>body{font:14px sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:5px}table{border-collapse:collapse}img{width:32%}.wrong{background:#ffe2e2}</style>',f'<h1>{matches} de {len(predicted)} predicciones coinciden con el manifest</h1>', '<p>Los candidatos filtrados se marcan EXCLUIDO; sus puntuaciones medidas se conservan. Consulta settings.region_filter en run.json.</p><p><a href="report.md">Informe y protocolo</a> · <a href="scores_wide.csv">CSV completo</a> · <a href="scores_long.csv">CSV largo</a> · <a href="run.json">JSON con poses</a></p><table><tr><th>Imagen</th><th>Esperado</th><th>Predicho</th><th>Margen</th>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>']
    for f in frames:
        by={r['cad_key']:r for r in f['scores']}
        h.append('<tr'+(' class="wrong"' if f.get('correct') is not True else '')+'><td><a href="#case'+str(f['frame_id'])+'">'+html.escape(f['name'])+'</a></td>'+''.join('<td>'+html.escape(str(f.get(k)))+'</td>' for k in ('expected_cad','predicted_cad','margin'))+''.join('<td>'+ ((f"{by[k]['score']:.6f}" + (" EXCLUIDO" if not by[k].get("eligible",True) else "")) if k in by else 'N/E')+'</td>' for k in keys)+'</tr>')
    h.append('</table>')
    for f in frames:
        h.append('<h2 id="case'+str(f['frame_id'])+'">'+html.escape(f['name'])+'</h2>')
        for name in ('mask_overlay.png','pose_overlay.png','cad_mesh_overlay.png'):
            rel=Path('cases')/str(f['frame_id'])/name
            if (output/rel).is_file():h.append(f'<a href="{rel}"><img loading="lazy" src="{rel}"></a>')
    (output/'index.html').write_text('\n'.join(h), encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=PREPARED/'manifest.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frame',action='append')
    parser.add_argument('--confidence',type=float,default=.3,help='Logged for original acceptance only; identification always selects top-1')
    parser.add_argument('--corner-filter',action=argparse.BooleanOptionalAction,default=True,help='Exclude only vertical_option_a in lower-right region; --no-corner-filter reproduces the unfiltered protocol')
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=args.manifest.resolve();base=manifest.parent;dataset=json.loads(manifest.read_text())
    if args.frame and set(args.frame)-{f['name'] for f in dataset['frames']}:parser.error('Unknown frame')
    selected=[f for f in dataset['frames'] if not args.frame or f['name'] in args.frame]
    started=time.perf_counter()
    run={'schema_version':1,'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
         'command':shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),
         'manifest':str(manifest),'manifest_sha256':digest(manifest),'catalog':dataset['parts'],'frames':[],
         'settings':{'seed':0,'threads':args.threads,'sam':'vit_h','stability':.97,'confidence_logged_only':args.confidence,
                     'final_selection':'max full ISM score on primary mask; FP winner only, no rerank','hardcoded_region_or_cad_filter':args.corner_filter,
                     'region_filter':{'enabled':args.corner_filter,'cad_key':'03a_vertical_option_a','bbox_format':'xywh','centre_x_gt':.72,'centre_y_gt':.65,'applies_to':['proposal_selection','final_CAD_selection'],'requested_after_review_of_baseline':True},
                     'iou':'independent per row; fixes upstream batch-wide all()','warmup_cases':0},
         'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                        'nvidia':command_record(['nvidia-smi']), 'git_head':command_record(['git','rev-parse','HEAD']),
                        'git_status':command_record(['git','status','--short']),
                        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
                        'runner_sha256':digest(__file__)}, 'setup_timing':{}}
    shutil.copy2(manifest,output/'dataset_manifest.json')
    shutil.copy2(__file__,output/'runner_snapshot.py')
    source_files = [PIPELINE/'region_filters.py', PIPELINE/'tools/evaluate_identification.py', ISM/'run_suite_pddl.py', ISM/'model/detector.py', ISM/'model/loss.py', ISM/'utils/bbox_utils.py', PIPELINE/'tools/score_all_cads.py', PIPELINE/'sam6d_ui/backends/foundationpose_worker.py']
    run['environment']['sources'] = {str(p):digest(p) for p in source_files}
    run['environment']['model_revisions'] = {name:command_record(['git','-C',str(ROOT/'models'/name),'rev-parse','HEAD']) for name in ('SAM-6D','FoundationPoseOfficial')}
    weights = [p for p in (ROOT/'checkpoints').rglob('*.pth') if p.is_file() and not p.is_symlink()]
    run['environment']['checkpoint_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in weights}
    run['setup_timing']['provenance_s'] = time.perf_counter()-started
    run['validation']=validate_dataset(dataset,base);write_json(output/'validation.json',run['validation'])
    write_json(output/'run.json',run)
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    os.environ.setdefault('MPLCONFIGDIR',str(output/'runtime/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(output/'runtime/cache'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(path('runtime_cache')/'torch_extensions'))
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    os.environ.setdefault('FOUNDATIONPOSE_ROOT',str(path('foundationpose')))
    os.environ.setdefault('FOUNDATIONPOSE_NATIVE',str(path('native')/'foundationpose'))
    os.environ.setdefault('FOUNDATIONPOSE_PYTHON',str(path('foundationpose_python')))
    sys.path.insert(0,str(ISM));os.chdir(ISM)
    import numpy as np
    import torch
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import run_suite_pddl as ism_worker
    import model.detector as detector
    from segmentor_runtime import configure,move_segmentor
    from sam6d_ui.backends import foundationpose as fp
    validation={c['frame_id']:c for c in run['validation']['frames']}
    worker=None;setup_error=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: run outside restricted sandbox on GPU host')
        device=torch.device('cuda')
        run['environment'].update(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        probe=command_record([str(path('foundationpose_python')),'-B',str(PIPELINE/'sam6d_ui/backends/foundationpose_worker.py'),'--probe'])
        run['environment']['foundationpose_probe']=probe
        if probe['returncode'] or not json.loads(probe['stdout'].strip().splitlines()[-1])['available']:raise RuntimeError('FoundationPose probe failed')
        setup=time.perf_counter()
        with initialize_config_dir(version_base=None,config_dir=str(ISM/'configs')):cfg=compose(config_name='run_inference.yaml')
        configure(cfg,'sam','vit_h',.97)
        model=instantiate(cfg.model);model.eval();model.descriptor_model.model.to(device);model.descriptor_model.model.device=device
        move_segmentor(model,device,'sam')
        run['setup_timing']['sam6d_model_load_s']=time.perf_counter()-setup
        # Override only in this evaluation process; preserve existing application/local edits.
        ism_worker.reject_repeated_region=reject_vertical_option_a_corner if args.corner_filter else lambda *a:False
        detector.compute_iou=paired_iou
        setup=time.perf_counter()
        with torch.inference_mode():run['references']=ism_worker.load_references(model,dataset['parts'],base,device,path('reference_cache'))
        run['setup_timing']['references_s']=time.perf_counter()-setup
        setup=time.perf_counter();worker=fp.resident(output/'foundationpose_worker.log');worker.request('warmup',timeout=300)
        run['setup_timing']['foundationpose_model_load_s']=time.perf_counter()-setup
        run['model_config']=__import__('omegaconf').OmegaConf.to_container(cfg,resolve=True)
    except Exception as e:
        setup_error=str(e);run['setup_error']={'error':str(e),'traceback':traceback.format_exc()}
    try:
        for n,f in enumerate(selected,1):
            print(f"[{n}/{len(selected)}] frame_id={f['frame_id']}",flush=True)
            folder=output/'cases'/str(f['frame_id'])
            if validation[f['frame_id']]['status']!='valid':
                result={'status':'invalid','error':'; '.join(validation[f['frame_id']]['errors']),'scores':[],'predicted_cad':None,'timing':{}}
            elif setup_error:
                result={'status':'failed','error':'Setup: '+setup_error,'scores':[],'predicted_cad':None,'timing':{}}
            else:
                try:
                    with torch.inference_mode():result=predict(model,prediction_input(f),dataset['parts'],base,folder,device,worker,args.confidence,args.corner_filter)
                except Exception as e:
                    result={'status':'failed','error':str(e),'scores':[],'predicted_cad':None,'timing':{},'traceback':traceback.format_exc()}
            # Evaluation-only join, strictly after prediction and pose.
            result.update(frame_id=f['frame_id'],name=f['name'],rgb_path=str((base/f['rgb']).resolve()),
                          expected_cad=f['expected_cad_key'],correct=result.get('predicted_cad')==f['expected_cad_key'] if result.get('predicted_cad') else None)
            folder.mkdir(parents=True,exist_ok=True);write_json(folder/'case.json',result)
            run['frames'].append(result);write_json(output/'run.json',run)
            print(json.dumps({k:result.get(k) for k in ('name','status','predicted_cad','expected_cad','correct','margin','error')}),flush=True)
    finally:
        if worker is not None:worker.close()
    # Check input correspondence and immutability at end, not merely file existence.
    unchanged=digest(manifest)==run['manifest_sha256']
    for f in selected:
        check=validation[f['frame_id']]
        if check['status']=='valid':unchanged=unchanged and all(digest(base/f[k])==v for k,v in check['hashes'].items())
    run['inputs_unchanged']=unchanged
    run['status']='complete' if all(f['status']=='predicted' for f in run['frames']) and unchanged else 'completed_with_failures'
    run['wall_s']=time.perf_counter()-started;run['finished_utc']=datetime.now(timezone.utc).isoformat()
    summary=export(run,output)
    print(json.dumps(summary,ensure_ascii=True),flush=True)
    return 0 if run['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
