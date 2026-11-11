"""Compare this filtered execution with the completed unfiltered baseline."""
import csv
import hashlib
import json
from pathlib import Path

root=Path('/home/raca/projects/CADTracking/results/evaluations')
base=root/'sam6d_fp_full_20261111';out=root/'sam6d_fp_corner_full_20261111'
a=json.loads((base/'run.json').read_text());b=json.loads((out/'run.json').read_text())
assert a['status']=='complete' and b['status']=='complete'
assert a['manifest_sha256']==b['manifest_sha256']
assert a['environment']['checkpoint_sha256']==b['environment']['checkpoint_sha256']
assert {p['part_key']:p['cache_key'] for p in a['references']['parts']}=={p['part_key']:p['cache_key'] for p in b['references']['parts']}
frames={f['frame_id']:f for f in a['frames']};assert len(frames)==len(b['frames'])==45
rows=[]
for after in b['frames']:
 before=frames[after['frame_id']]
 assert before['name']==after['name'] and before['expected_cad']==after['expected_cad']
 v_a=next(v for v in a['validation']['frames'] if v['frame_id']==after['frame_id'])
 v_b=next(v for v in b['validation']['frames'] if v['frame_id']==after['frame_id'])
 assert v_a['hashes']==v_b['hashes']
 rows.append({'frame_id':after['frame_id'],'name':after['name'],'expected_cad':after['expected_cad'],
              'before_cad':before['predicted_cad'],'after_cad':after['predicted_cad'],
              'before_correct':before['correct'],'after_correct':after['correct'],
              'mask_changed':before['primary_mask_sha256']!=after['primary_mask_sha256'],
              'rejected_proposals':len(after['region_filter']['rejected_detections']),
              'excluded_ranking_candidates':sum(not r['eligible'] for r in after['scores'])})
summary={'baseline':str(base),'filtered_run':str(out),'matched_input_cases':45,
         'same_input_hashes':True,'same_checkpoint_hashes':True,'same_reference_cache_keys':True,
         'before_matches':a['summary']['matches'],'after_matches':b['summary']['matches'],
         'recovered_cases':[r['name'] for r in rows if not r['before_correct'] and r['after_correct']],
         'lost_cases':[r['name'] for r in rows if r['before_correct'] and not r['after_correct']],
         'mask_changed_cases':sum(r['mask_changed'] for r in rows),
         'rejected_proposals':sum(r['rejected_proposals'] for r in rows),
         'excluded_ranking_candidates':sum(r['excluded_ranking_candidates'] for r in rows),
         'protocol_note':'Same data and weights; corner heuristic requested after inspecting these cases. This is not an independent held-out validation.',
         'cases':rows}
(out/'filter_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (out/'filter_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Comparación con y sin filtro',
       f"Sin filtro: {summary['before_matches']}/45. Con filtro: {summary['after_matches']}/45.",
       f"Propuestas de 03a descartadas en esquina: {summary['rejected_proposals']}. Máscaras primarias cambiadas: {summary['mask_changed_cases']}.",
       'Recuperados: '+', '.join(summary['recovered_cases']), 'Aciertos perdidos: '+(', '.join(summary['lost_cases']) or 'ninguno'),
       'Se verificaron hashes iguales para RGB, profundidad, cámara, pesos y claves de las referencias en los 45 casos. Las dos ejecuciones hicieron inferencia nueva; sólo se reutilizaron referencias CAD.',
       summary['protocol_note'],'| Imagen | Esperado | Antes | Después | Antes correcto | Después correcto |',
       '|---|---|---|---|---:|---:|']
for row in rows:
 lines.append('| '+' | '.join(str(row[k]) for k in ('name','expected_cad','before_cad','after_cad','before_correct','after_correct'))+' |')
(out/'comparison.md').write_text('\n\n'.join(lines[:7])+'\n\n'+'\n'.join(lines[7:])+'\n',encoding='utf-8')
index=(out/'index.html').read_text(encoding='utf-8')
index += '<p><a href="comparison.md">Comparación con y sin filtro</a> · <a href="filter_comparison.csv">CSV de comparación</a></p>'
(out/'index.html').write_text(index,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False))
