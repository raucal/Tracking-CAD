"""Image + CAD detection, optionally followed by official RGB-D PEM pose estimation."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))
from pathlib import Path
import argparse
import base64
from datetime import datetime,timezone
import json
import sys

from sam6d_ui.inputs import prepare_upload

from settings import ROOT, PIPELINE, PREPARED, RESULTS

def file(path):
    path=Path(path).resolve()
    return {'name':path.name,'data':base64.b64encode(path.read_bytes()).decode()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb',type=Path);p.add_argument('--cad',type=Path)
    p.add_argument('--depth',type=Path);p.add_argument('--camera',type=Path)
    p.add_argument('--example',help='Capture from the existing assembly, e.g. control_board')
    p.add_argument('--part',default='05_control_board',help='CAD key for an assembly example')
    p.add_argument('--segmentor',choices=('sam','fastsam'),default='sam')
    p.add_argument('--backbone',choices=('vit_h','vit_l','vit_b'),default='vit_h')
    p.add_argument('--mode',choices=('rgb','rgbd'))
    p.add_argument('--pose',action='store_true')
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    p.add_argument('--cad-units',choices=('mm','cm','m'),default='mm')
    p.add_argument('--confidence',type=float,default=0.5)
    p.add_argument('--min-score',type=float,default=0.3)
    p.add_argument('--fastsam-conf',type=float,default=0.05)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    mode=a.mode or ('rgbd' if a.example or a.depth or a.pose else 'rgb')
    config={'mode':mode,'device':a.device,'segmentor':a.segmentor,'backbone':a.backbone,'pose':a.pose,
            'cad_units':a.cad_units,'confidence':a.confidence,'min_score':a.min_score,'fastsam_conf':a.fastsam_conf}
    if a.example:
        if a.rgb or a.cad or a.depth or a.camera:p.error('Use --example o archivos propios, no ambos.')
        base=PREPARED;data=json.loads((base/'manifest.json').read_text())
        frame=next((f for f in data['frames'] if f['name']==a.example),None)
        part=next((v for v in data['parts'] if v['key']==a.part),None)
        if not frame or not part:p.error('Captura o CAD desconocidos.')
        request={k:file(base/frame[src]) for k,src in (('rgb','rgb'),('depth','depth'),('camera','camera'))}
        request['cad']=file(base/part['source_stl'])
    else:
        if not a.rgb or not a.cad:p.error('Se requieren --rgb y --cad, o --example.')
        request={'rgb':file(a.rgb),'cad':file(a.cad)}
        if a.depth:request['depth']=file(a.depth)
        if a.camera:request['camera']=file(a.camera)
    request['options']=config
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=(a.output or RESULTS/'runs'/stamp).resolve()
    if output.exists():p.error('La salida ya existe; elige una carpeta nueva para conservar los resultados.')
    inputs=prepare_upload(request,output)
    from sam6d_ui.engine import Engine
    result=Engine(RESULTS/'interface').run(output,inputs,lambda text:print(text,flush=True))
    print(json.dumps({'output':str(output),'device':result['device'],'segmentor':a.segmentor,'best_score':result['best_score'],
                      'recognized':result['recognized'],'pose_6d_estimated':result['pose_6d_estimated'],'timing':result['timing']},indent=2),flush=True)
    return 0
if __name__=='__main__':sys.exit(main())
