"""Official FastSAM-x and SAM-6D PEM weights; one shared copy per checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import gdown

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'SAM6D/SAM-6D/SAM-6D'
ASSETS=[('FastSAM-x.pt','1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv','Instance_Segmentation_Model/checkpoints/FastSAM'),
        ('sam-6d-pem-base.pth','1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7','Pose_Estimation_Model/checkpoints')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path('/dev/shm/pddl-assembly-runtime/checkpoints'));a=p.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
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
  elif not link.exists():link.symlink_to(target)
  report.append({'name':name,'official_google_drive_id':google_id,'cache':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
  print('Ready:',name,flush=True)
 (ROOT/'SAM6D/pipeline/prepared/pose_checkpoints.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
