"""SAM refresh on a current RGB-D pair; transient images stay in shared memory."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


class Redetector:
    def __init__(self, engine_factory, seed_folder):
        self.engine_factory = engine_factory
        self.seed_folder = Path(seed_folder)
        self.inputs = deepcopy(json.loads((self.seed_folder / 'inputs.json').read_text()))
        for key in ('manual_bbox', 'capture_id', 'recording_path'):
            self.inputs.pop(key, None)
        self.inputs['options'].update(mode='rgbd', device='cuda', pose=False)

    def __call__(self, frame, scratch, cancelled):
        folder = scratch / 'detection'
        folder.mkdir(exist_ok=True)
        (scratch / 'tracking_model.json').unlink(missing_ok=True)
        mask_path = scratch / 'mask.npy'
        mask_path.unlink(missing_ok=True)
        for name in (self.inputs['cad_filename'], 'object_mm.obj'):
            if not (folder / name).exists():
                shutil.copy2(self.seed_folder / name, folder / name)
        Image.fromarray(frame['rgb']).save(folder / 'input_rgb.png')
        Image.fromarray(frame['depth']).save(folder / 'depth.png')
        (folder / 'camera.json').write_text(json.dumps(frame['camera']))
        if cancelled.is_set():
            return None
        result = self.engine_factory().run(folder, self.inputs, lambda message: None, save_outputs=False)
        if cancelled.is_set():
            return None
        accepted = [(i, d) for i, d in enumerate(result['detections'])
                    if d['score'] >= self.inputs['options']['confidence']]
        if not accepted:
            return None
        index, detection = max(accepted, key=lambda pair: pair[1]['score'])
        mask=np.asarray(result['_masks'][index])>0
        if mask.shape != frame['depth'].shape or not mask.any():
            raise ValueError('La máscara de refresco no corresponde al RGB-D actual.')
        np.save(mask_path, mask, allow_pickle=False)
        if self.inputs.get('auto_assembly'):
            (scratch / 'tracking_model.json').write_text(json.dumps(result['tracking_model']))
        y,x=np.nonzero(mask)
        value = {'score':float(detection['score']),'segmentor':self.inputs['options']['segmentor'],'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]}
        if self.inputs.get('auto_assembly'):
            value.update(cad_key=result['cad_key'], display_name=result['display_name'],
                         assembly=self.inputs['auto_assembly'])
        return value
