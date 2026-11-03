"""Official FastSAM-x and SAM-6D PEM weights; one shared copy per checkpoint."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / 'settings.py').is_file())))
from pathlib import Path
import os
import argparse
import hashlib
import json
import gdown

from settings import ROOT, CORE, PREPARED, path
ASSETS=[('FastSAM-x.pt','1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv','Instance_Segmentation_Model/checkpoints/FastSAM'),
        ('sam-6d-pem-base.pth','1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7','Pose_Estimation_Model/checkpoints')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=path('checkpoints'));a=p.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
 report=[]
 for name,google_id,group in ASSETS:
  target=a.cache/name
  if not target.is_file():
   temp=target.with_suffix('.download')
   result=gdown.download(id=google_id,output=str(temp),quiet=False,use_cookies=False)
   if not result:raise RuntimeError('Download failed: '+name)
   if temp.stat().st_size<10_000_000:raise RuntimeError('Checkpoint too small: '+name)
   temp.rename(target)
  link=CORE/group/name;link.parent.mkdir(parents=True,exist_ok=True)
  if link.is_symlink():
   if link.resolve()!=target.resolve():raise ValueError('Unexpected checkpoint link: '+str(link))
  elif not link.exists():link.symlink_to(os.path.relpath(target, link.parent))
  report.append({'name':name,'official_google_drive_id':google_id,'cache':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
  print('Ready:',name,flush=True)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
