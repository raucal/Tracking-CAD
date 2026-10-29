"""FoundationPose with a manually selected rectangle; no SAM, DINO or CAD rendering."""
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .foundationpose import run as foundationpose_run


def validate_box(value,size):
    if not isinstance(value,list) or len(value)!=4:
        raise ValueError('La caja manual debe contener [x, y, ancho, alto].')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or int(v)!=v for v in value):
        raise ValueError('La caja manual requiere coordenadas enteras en píxeles.')
    x,y,width,height=map(int,value)
    if min(x,y)<0 or min(width,height)<4 or x+width>size[0] or y+height>size[1]:
        raise ValueError('La caja manual debe estar dentro de la imagen y medir al menos 4×4 píxeles.')
    return [x,y,width,height]


def run(folder,inputs,update):
    started=time.perf_counter();folder=Path(folder)
    config=inputs['options']
    if config['pose_backend']!='foundationpose' or config['mode']!='rgbd' or not config['pose']:
        raise ValueError('La inicialización manual requiere FoundationPose y RGB-D.')
    with Image.open(folder/'input_rgb.png') as source:rgb=source.convert('RGB')
    x,y,width,height=validate_box(inputs['manual_bbox'],rgb.size)
    mask=np.zeros((rgb.height,rgb.width),np.uint8);mask[y:y+height,x:x+width]=255
    Image.fromarray(mask).save(folder/'mask_000.png')
    overlay=rgb.copy();draw=ImageDraw.Draw(overlay);draw.rectangle((x,y,x+width-1,y+height-1),outline=(40,220,150),width=3)
    overlay.save(folder/'overlay.png')
    (folder/'manual_selection.json').write_text(json.dumps({'type':'rectangle_mask','bbox':[x,y,width,height],'automatic_detection':False,'background_excluded':False},indent=2))
    (folder/'detections.json').write_text('[]')
    transform=np.eye(4);transform[:3,3]=-np.asarray(inputs['input_center_mm'])
    update('FoundationPose: estimando pose desde la caja manual, sin SAM6D')
    pose_started=time.perf_counter()
    pose=foundationpose_run(folder,folder/'object_mm.obj',0,transform)
    result={'backend':'foundationpose','initialization':'manual_bbox','mode':'rgbd','device':'cuda',
            'display_name':inputs['cad_name'],'cad_name':inputs['cad_name'],'cad_key':'manual_cad',
            'recognized':False,'accepted_detections':0,'valid_proposals':0,'best_score':None,
            'formula':'manual rectangular mask; no automatic detection','score_is_probability':False,
            'pose_6d_estimated':bool(pose.get('poses')),'pose':pose,
            'timing':{'inference_s':0,'pose_pipeline_s':time.perf_counter()-pose_started,'total_s':time.perf_counter()-started},
            'inference_stages':{},'model_reused':False,'reference_cache_hit':False,'renderer':'not_used',
            'options':config,'detections':[]}
    (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
