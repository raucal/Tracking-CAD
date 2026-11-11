"""Verify a completed fresh evaluation, compare historical bytes, and plot results."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluate_identification import digest, write_json, save_csv, command_record, ROOT
from settings import path
from region_filters import reject_vertical_option_a_corner


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--initial-audit',type=Path);args=p.parse_args();root=args.run.resolve()
    r=json.loads((root/'run.json').read_text(encoding='utf-8'));m=json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
    by_id={f['frame_id']:f for f in m['frames']};keys={p['key'] for p in m['parts']};base=Path(r['manifest']).parent
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
    check('manifest_snapshot_hash',digest(root/'dataset_manifest.json')==r['manifest_sha256'])
    check('original_manifest_unchanged',digest(Path(r['manifest']))==r['manifest_sha256'])
    for part in r['validation']['parts']:
        source=next(p for p in m['parts'] if p['key']==part['key'])
        check('mesh_unchanged:'+part['key'],digest(base/source['sam_ply'])==part['mesh_sha256'])
    for name, value in r['environment'].get('checkpoint_sha256',{}).items():
        check('checkpoint_unchanged:'+name,digest(ROOT/name)==value)
    fp_env=command_record([str(path('foundationpose_python')),'-m','pip','list','--format=json'])
    write_json(root/'foundationpose_environment.json',fp_env)
    check('unique_result_ids',len({f['frame_id'] for f in r['frames']})==len(r['frames']))
    for f in r['frames']:
        i=f['frame_id'];source=by_id[i];folder=root/'cases'/str(i);rows=f['scores']
        check(f'{i}:manifest_join',f['name']==source['name'] and f['expected_cad']==source['expected_cad_key'] and f['rgb_path']==str((base/source['rgb']).resolve()))
        if not rows:continue
        check(f'{i}:all_CADs_once',len(rows)==len(keys) and {x['cad_key'] for x in rows}==keys)
        eligible=[x for x in rows if x.get('eligible',True)]
        check(f'{i}:sorted_eligible_ranking',all(eligible[n]['score']>=eligible[n+1]['score'] for n in range(len(eligible)-1)) and [x['rank'] for x in eligible]==list(range(1,len(eligible)+1)))
        if r['settings'].get('region_filter',{}).get('enabled'):
            case=json.loads((folder/'ism_case.json').read_text(encoding='utf-8'))
            shape=next(v['shape'] for v in r['validation']['frames'] if v['frame_id']==i)
            bbox=case['candidate']['bbox']
            check(f'{i}:corner_filter_eligibility',all(x['eligible']==(not reject_vertical_option_a_corner(x['cad_key'],bbox,(shape[1],shape[0]))) for x in rows))
            check(f'{i}:excluded_not_ranked',all(x['rank'] is None and x['filter_reason'] for x in rows if not x['eligible']))
            raw=sorted(rows,key=lambda x:x['raw_rank'])
            check(f'{i}:raw_scores_preserved',all(raw[n]['score']>=raw[n+1]['score'] for n in range(len(raw)-1)))
        check(f'{i}:winner_and_correctness',f['predicted_cad']==eligible[0]['cad_key'] and f['correct']==(f['predicted_cad']==f['expected_cad']))
        check(f'{i}:margin',np.isclose(f['margin'],eligible[0]['score']-eligible[1]['score'],atol=1e-9))
        check(f'{i}:primary_mask_hash',digest(folder/'mask_000.png')==f['primary_mask_sha256'])
        check(f'{i}:score_formula',all(np.isclose(x['score'],(x['semantic']+x['appearance']+x['geometry']*x['visible_ratio'])/(2+x['visible_ratio']),atol=2e-7) for x in rows))
        check(f'{i}:score_scales_finite',all(np.isfinite(x[k]) and -1e-6<=x[k]<=1+1e-6 for x in rows for k in ('score','semantic','appearance','geometry','visible_ratio')))
        check(f'{i}:FP_only_winner',all(x['foundationpose_score'] is None and x['foundationpose_status']=='not_evaluated' for x in rows[1:]))
        for x in rows:
            part=next(p for p in m['parts'] if p['key']==x['cad_key'])
            check(f'{i}:{x["cad_key"]}:category_mapping',x['category_id']==part['category_id'])
        if f.get('pose'):
            pose=f['pose']['poses'][0];T=np.asarray(pose['T_camera_from_model_mm']);P=np.asarray(pose['T_camera_from_input_cad_mm'])
            part=next(p for p in m['parts'] if p['key']==f['predicted_cad']);local=np.asarray(part['T_local_from_sam_mm'])
            check(f'{i}:pose_transform',np.allclose(P,T@np.linalg.inv(local),atol=1e-5))
            check(f'{i}:rigid_pose',np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3) and np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-3) and T[2,3]>0)
            check(f'{i}:pose_score',np.isfinite(rows[0]['foundationpose_score']) and rows[0]['foundationpose_score']==pose['pose_score'])
            check(f'{i}:visualizations',all((folder/n).is_file() for n in ('mask_overlay.png','pose_overlay.png','cad_mesh_overlay.png')))
    with (root/'scores_long.csv').open(encoding='utf-8-sig') as fp:long=list(csv.DictReader(fp))
    with (root/'scores_wide.csv').open(encoding='utf-8-sig') as fp:wide=list(csv.DictReader(fp))
    check('CSV_long_cartesian_size',len(long)==len(r['frames'])*len(keys))
    check('CSV_long_unique_pairs',len({(x['frame_id'],x['cad_key']) for x in long})==len(long))
    check('CSV_wide_size',len(wide)==len(r['frames']))
    pose_rows=[]
    for f in r['frames']:
        pose=(f.get('pose') or {}).get('poses',[None])[0]
        row={'frame_id':f['frame_id'],'name':f['name'],'rgb_path':f['rgb_path'],'expected_cad':f['expected_cad'],'predicted_cad':f.get('predicted_cad'),'correct':f.get('correct'),'pose_status':(f.get('pose') or {}).get('status'),'translation_unit':'mm','camera_convention':'OpenCV: X right, Y down, Z forward','pose_score':pose.get('pose_score') if pose else None}
        for key in ('T_camera_from_model_mm','T_camera_from_input_cad_mm'):
            for i in range(4):
                for j in range(4):row[f'{key}_{i}{j}']=pose[key][i][j] if pose else None
        pose_rows.append(row)
    save_csv(root/'poses.csv',pose_rows);write_json(root/'poses.json',pose_rows)
    check('pose_CSV_rows',len(pose_rows)==len(r['frames']))
    historical=ROOT/'results/reports/sam6d_suite_20261102T150131Z'
    old=json.loads((historical/'resultados_completos.json').read_text(encoding='utf-8'))
    oldm=json.loads((historical/'dataset_manifest.json').read_text(encoding='utf-8'))
    common=[];new_hash={digest(base/f['rgb']):f for f in m['frames']}
    for f in old['frames']:
        rgb=historical/'cases'/f['name']/'rgb.png'
        if rgb.is_file() and digest(rgb) in new_hash:
            match=new_hash[digest(rgb)]
            new=next((x for x in r['frames'] if x['frame_id']==match['frame_id']),None)
            common.append({'old_name':f['name'],'new_name':match['name'],'rgb_sha256':digest(rgb),
                           'expected_from_current_manifest':match['expected_cad_key'],
                           'old_prediction':f['candidate']['part_key'] if f['candidate'] else None,
                           'new_prediction':new.get('predicted_cad') if new else None})
    baseline={'first_suite':str(historical),'old_cases':len(old['frames']),'new_cases':len(r['frames']),
              'old_expected_labels':sum('expected_cad_key' in f for f in oldm['frames']),
              'common_exact_rgb_bytes':len(common),'common':common,
              'direct_protocol_comparison_valid':False,
              'reasons':['First suite uses PEM, new uses FoundationPose','First table contains assigned-detection maxima; new all-CAD hypotheses share one mask','New IoU is per row; historical IoU can zero whole batch','Different capture sets; no original ground truth labels']}
    write_json(root/'baseline_comparison.json',baseline)
    missing_sources=[]
    for f in m['frames']:
        for key in ('source_rgb','source_depth','source_metadata'):
            if key in f and not (base/f[key]).is_file():missing_sources.append({'name':f['name'],'field':key,'recorded_path':f[key]})
    initial=json.loads(args.initial_audit.read_text(encoding='utf-8')) if args.initial_audit else None
    if initial:
        for sub in ('','models/SAM-6D','models/FoundationPoseOfficial'):
            before=initial['commands'][sub+' diff']['stdout']
            current=command_record(['git','-C',str(ROOT/sub),'diff'])
            check('preserved_preexisting_tracked_diff:'+sub,current['stdout']==before)
    history={'initial_audit':str(args.initial_audit) if args.initial_audit else None,
             'captured_utc':initial['captured_utc'] if initial else None,
             'window_start_utc':initial['window_start_utc'] if initial else None,
             'recent_mtimes':initial['recent_mtimes'] if initial else None,
             'certainty':'Git timestamps show recorded commits, not proof of actual edit time. Current diff cannot be dated to the last two hours. mtime cannot prove authorship or exact edit history.'}
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'checks_passed':sum(c['passed'] for c in checks),
           'checks_total':len(checks),'all_passed':all(c['passed'] for c in checks), 'baseline':baseline,'history':history,
           'missing_original_source_paths':missing_sources,
           'source_limitation':'Prepared RGB-D/camera hashes verified; original paths are broken or unavailable. Cannot certify historical alignment or calibration accuracy from arrays alone.'}
    write_json(root/'verification.json',audit)
    plots(root,r)
    shutil.copy2(__file__,root/'audit_runner_snapshot.py')
    summary=r['summary']
    wrong=[f for f in r['frames'] if f.get('correct') is False]
    corners=[]
    for f in wrong:
        case_folder=root/'cases'/str(f['frame_id'])
        case_path=case_folder/'ism_case.json'
        if not case_path.is_file():case_path=case_folder/'unfiltered_ism_case.json'
        case=json.loads(case_path.read_text(encoding='utf-8'))
        x,y,w,h=case['candidate']['bbox']
        shape=next(v['shape'] for v in r['validation']['frames'] if v['frame_id']==f['frame_id'])
        if (x+w/2)/shape[1]>.72 and (y+h/2)/shape[0]>.65:corners.append(f['name'])
    medc=f"{summary['margins_correct']['median']:.6f}" if summary['margins_correct']['median'] is not None else 'N/E'
    medw=f"{summary['margins_wrong']['median']:.6f}" if summary['margins_wrong']['median'] is not None else 'N/E'
    wrong_ranked=summary['wrong_high_scores']
    confusions={k:sum(f.get('predicted_cad')==k for f in wrong) for k in sorted(keys)}
    switches=[f for f in r['frames'] if f.get('predicted_cad') and f.get('semantic_winner')!=f['predicted_cad']]
    rescued=[f['name'] for f in switches if f['correct'] and f['semantic_winner']!=f['expected_cad']]
    lost=[f['name'] for f in switches if not f['correct'] and f['semantic_winner']==f['expected_cad']]
    fp_scores=[f['scores'][0]['foundationpose_score'] for f in r['frames'] if f['scores'] and f['scores'][0]['foundationpose_score'] is not None]
    latency_table='| Etapa | Media s | Mediana s | P95 s |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {stage} | {v['mean']:.4f} | {v['median']:.4f} | {v['p95']:.4f} |" for stage,v in summary['latency'].items() if v['count'])
    analysis=[f"{summary['matches']} de {summary['processed_identification']} predicciones coinciden con el CAD del manifest. Top-3: {summary['topk'].get('3',{}).get('matches')}/{summary['processed_identification']}; top-5: {summary['topk'].get('5',{}).get('matches')}/{summary['processed_identification']}. Identificaciones completas: {summary['processed_identification']}; poses estimadas: {summary['complete_with_pose']}; fallos de ejecución: {summary['failed_identification']+summary['failed_pose']}; omitidos: {summary['omitted']}.",
              f"En {len(corners)} de los {len(wrong)} errores, la región primaria tiene su centro en x>0.72, y>0.65 de la imagen: {', '.join(corners)}. En el contact sheet se puede revisar si corresponde al objeto esperado. Sin anotaciones de máscara no se mide su IoU real.",
              'Distribución de las identidades predichas incorrectamente: '+json.dumps({k:v for k,v in confusions.items() if v},ensure_ascii=False)+'. El recall de clases sin casos positivos queda sin definir.',
              f"Margen mediano en aciertos: {medc}; en errores: {medw}. Los errores ordenados por puntuación se guardan en summary.json: wrong_high_scores. Son puntuaciones sin calibración probabilística; un score o margen alto no certifica una identificación.",
              f"El refinamiento de semantic elegible a ISM completo elegible cambia {len(switches)} decisiones. Rescata: {rescued}. Pierde: {lost}. Aciertos del ranking semántico: {summary['semantic_correct']}; del ranking final: {summary['matches']}. FoundationPose no cambia la identidad por diseño.",
              'Errores con las puntuaciones mayores: '+json.dumps(wrong_ranked[:3],ensure_ascii=False),
              f"Tiempo global, incluida carga y procedencia: {r['wall_s']:.3f} s. La tabla contiene etapas y subtotales solapados; no deben sumarse sus filas.",latency_table,
              'Se guardan las poses en poses.csv/json con las dos matrices, la unidad y el score FP. Rango observado de scores FP: '+str((min(fp_scores),max(fp_scores)) if fp_scores else None)+'. Es una escala diferente de ISM. Sin ground truth ni simetrías anotadas no procede calcular ADD, ADD-S, VSD o error angular.']
    table='| CAD | Casos | Aciertos | Recall |\n|---|---:|---:|---:|\n'+'\n'.join('| '+p['cad_key']+' | '+str(p['support'])+' | '+str(p['correct'])+' | '+(f"{p['recall_on_selected']:.1%}" if p['recall_on_selected'] is not None else 'No definido')+' |' for p in summary['per_cad'])
    lines=['# Auditoría y hallazgos',*analysis,table,'Filtro y limitación de comparación: '+json.dumps(r['settings'].get('region_filter',{'enabled':False}),ensure_ascii=False)+'. Esta heurística se solicitó tras revisar errores del conjunto actual; los resultados filtrados no son una validación independiente sobre datos nuevos.',f"{audit['checks_passed']}/{audit['checks_total']} comprobaciones de correspondencia, fórmulas, rankings, poses y exportaciones pasan.",
           '## Errores y limitaciones con evidencia',
           '- `pipeline/region_filters.py`: regla compartida para excluir sólo vertical_option_a cuando el centro de su bbox está en x>0.72, y>0.65. La interfaz y el worker SAM6D la usan; la suite también impide seleccionar ese candidato durante el ranking final. `--no-corner-filter` desactiva la regla en la suite.',
           '- `models/SAM-6D/SAM-6D/Instance_Segmentation_Model/utils/bbox_utils.py:214`: `(wh_inter > 0).all()` aplica una condición a todo el lote; una pareja disjunta anula IoU de todas. La suite instala `paired_iou` por pareja; prueba con una pareja idéntica y otra disjunta: [1,0].',
           '- `pipeline/tools/process_saved_demos.py:152`: CAD seleccionado mediante `DEMO_PARTS[demo.name]` y pasado en `request.cad` (línea 166); es pose con identidad conocida, no identificación automática. No forma parte de estas predicciones.',
           '- `pipeline/suites/sam6d_suite.py:44`: la opción pose llama a PEM, no a FoundationPose. La suite nueva usa explícitamente el trabajador FP.',
           '- `pipeline/suites/sam6d_suite_report.py:279`: denominador esperado cuenta todo el manifest incluso cuando se selecciona un subconjunto. La nueva suite separa manifest, seleccionados, predicciones y poses.',
           f"- Manifest actual: {len(missing_sources)} rutas source_* no existen resueltas respecto a su directorio. Ejemplo: `{m['frames'][0].get('source_rgb')}`. La inferencia utiliza los archivos preparados existentes y verificados; no reconstruye ni inventa los originales.",
           '- Escala/cámara/transformaciones: checks numéricos y límites CAD pasan. `depth_scale` está en mm por unidad raw; SAM6D y FP convierten a metros. La correspondencia física RGB-profundidad y la exactitud de calibración no pueden demostrarse sin datos externos. No se afirma precisión métrica de pose.',
           '## Cambios conservados y realizados',
           'La regla del filtro se comparte en `pipeline/region_filters.py`; la usan la interfaz, el worker SAM6D y la suite. El IoU corregido sigue limitado al proceso de evaluación. Los resultados anteriores se conservan en sus directorios originales.',
           '## Historial de las dos horas previas',
           json.dumps(history,ensure_ascii=False,indent=2),
           'La instantánea `initial_audit.json` conserva diff completo, staged diff, estado, commits, reflog y mtimes antes de trabajar. Consulte la instantánea inicial cuando se haya indicado --initial-audit. Sin ella no se reconstruye el historial de esta ejecución. No se puede fechar el diff local con certeza.',
           '## Primera suite',json.dumps(baseline,ensure_ascii=False,indent=2),
           'La codificación del informe del smoke falló al finalizar la inferencia y se corrigió explícitamente a UTF-8. Se reexportaron los cálculos guardados; no se contaron como otra ejecución. La suite completa se ejecutó después con la corrección.',
           'La tabla original es `confianza_sam6d_por_cad.csv`; `comparacion_todos_los_cad.csv` almacena cosenos semánticos y el diagnóstico adicional `ism_view_details.json` usa máscaras guardadas. No se reutilizaron máscaras, predicciones ni poses históricas para simular una ejecución nueva.',
           '## Repetir', '```bash',r['command'],f"PYTHONUTF8=1 {sys.executable} {Path(__file__).resolve()} --run {root} --initial-audit {args.initial_audit}",'```',
           '## Visualizaciones','`confusion_matrix.png`, `score_heatmap.png`, `margins.png`, `latency.png`, `mask_contact_sheet.jpg` y los tres overlays de cada caso. El wireframe del CAD incluye aristas ocultas: es una ayuda de inspección, no una métrica de precisión.']
    (root/'audit.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    index=(root/'index.html').read_text(encoding='utf-8')
    if 'audit.md' not in index:
        index += '\n<p><a href="audit.md">Auditoría y errores</a> · <a href="verification.json">Comprobaciones</a></p>\n' + ''.join(f'<a href="{f}"><img src="{f}"></a>' for f in ('confusion_matrix.png','score_heatmap.png','margins.png','latency.png','mask_contact_sheet.jpg'))
        (root/'index.html').write_text(index,encoding='utf-8')
    print(json.dumps({'checks_passed':audit['checks_passed'],'checks_total':len(checks),'baseline_common':len(common),'missing_sources':len(missing_sources)}))
    return 0 if audit['all_passed'] else 1


def plots(root,r):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'runtime/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    keys=[p['key'] for p in r['catalog']];frames=r['frames'];pred=[f for f in frames if f.get('predicted_cad')]
    mat=np.array([[sum(f['expected_cad']==a and f.get('predicted_cad')==b for f in frames) for b in keys] for a in keys])
    fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(mat,cmap='Blues',vmin=0)
    ax.set(xticks=range(len(keys)),xticklabels=keys,yticks=range(len(keys)),yticklabels=keys,xlabel='Predicted CAD',ylabel='Manifest CAD',title='Confusion matrix (counts)')
    plt.setp(ax.get_xticklabels(),rotation=65,ha='right')
    for i in range(len(keys)):
        for j in range(len(keys)):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>mat.max()/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/'confusion_matrix.png',dpi=150);plt.close(fig)
    heat=np.array([[next((x['score'] for x in f['scores'] if x['cad_key']==key),np.nan) for key in keys] for f in frames])
    fig,ax=plt.subplots(figsize=(12,16));im=ax.imshow(heat,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set(xticks=range(len(keys)),xticklabels=keys,yticks=range(len(frames)),yticklabels=[f['name'] for f in frames],title='Full ISM score for each image and CAD (same mask)')
    plt.setp(ax.get_xticklabels(),rotation=65,ha='right');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/'score_heatmap.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for correct,color,label in ((True,'#168540','Correct'),(False,'#d84343','Incorrect')):
        fs=[f for f in pred if f['correct']==correct];ax.scatter([f['margin'] for f in fs],[f['scores'][0]['score'] for f in fs],c=color,label=label,alpha=.7)
    ax.set(xlabel='Top-1 minus top-2 ISM score',ylabel='Top-1 ISM score',title='Scores are not probabilities');ax.legend();fig.tight_layout();fig.savefig(root/'margins.png',dpi=150);plt.close(fig)
    stages=('sam_s','dino_s','ranking_query_features_s','ranking_all_cads_s','foundationpose_pipeline_s')
    fig,ax=plt.subplots(figsize=(13,6));bottom=np.zeros(len(frames))
    for stage in stages:
        values=np.array([f.get('timing',{}).get(stage,0) for f in frames]);ax.bar(range(len(frames)),values,bottom=bottom,label=stage);bottom+=values
    ax.set(xlabel='Case index',ylabel='Seconds',title='Selected inference stages (remaining I/O and setup excluded)');ax.legend();fig.tight_layout();fig.savefig(root/'latency.png',dpi=150);plt.close(fig)
    cols=5;w,h=256,212;sheet=Image.new('RGB',(cols*w,((len(frames)+cols-1)//cols)*h),'#fff');draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        p=root/'cases'/str(f['frame_id'])/'mask_overlay.png';x=(i%cols)*w;y=(i//cols)*h
        if p.is_file():
            with Image.open(p) as im:im=im.convert('RGB');im.thumbnail((256,192));sheet.paste(im,(x,y))
        draw.text((x+3,y+192),f"{f['frame_id']} {f['name']} {'OK' if f.get('correct') else 'ERROR'}",fill='green' if f.get('correct') else 'red')
    sheet.save(root/'mask_contact_sheet.jpg',quality=90)


if __name__=='__main__':raise SystemExit(main())
