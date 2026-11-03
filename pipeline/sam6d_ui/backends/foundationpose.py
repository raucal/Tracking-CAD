"""Adaptador de FoundationPose en un Python aislado del motor SAM-6D."""
import json
import os
from pathlib import Path
import subprocess
import time


from settings import RESULTS, path


def python_path():
    return Path(os.environ.get('FOUNDATIONPOSE_PYTHON', path('foundationpose_python')))


def availability():
    python = python_path()
    if not python.is_file():
        return {'available': False, 'reason': 'Falta instalar el entorno de FoundationPose.', 'python': str(python)}
    try:
        result = subprocess.run([str(python), '-B', str(Path(__file__).with_name('foundationpose_worker.py')), '--probe'],
                                capture_output=True, text=True, timeout=25)
        if result.returncode:
            return {'available': False, 'reason': result.stderr[-1500:] or 'No se pudo comprobar FoundationPose.'}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'reason': str(error)}


_resident=None
_resident_lock=__import__('threading').Lock()


def resident(log_path=None):
    global _resident
    with _resident_lock:
        if _resident is None:
            import atexit
            from .resident import ResidentPose
            _resident=ResidentPose(python_path(),Path(__file__).with_name('resident_worker.py'),
                                   log_path or RESULTS/'interface/logs/foundationpose-worker.log')
            atexit.register(_resident.close)
        return _resident


def run(folder, mesh, mask_index, model_from_input, recording=None):
    folder=Path(folder);transform=folder/'model_from_input_mm.json'
    transform.write_text(json.dumps(model_from_input.tolist()))
    worker=resident()
    args={'folder':str(folder),'mesh':str(Path(mesh).resolve()),'mask':str(folder/f'mask_{mask_index:03d}.png'),
          'transform':str(transform),'output':str(folder/'pose.json'),'recording':str(recording) if recording else None}
    started=time.perf_counter()
    try:
        result=worker.request('estimate',timeout=1800 if recording else 300,args=args)
    finally:
        (folder/'pose.log').write_text(json.dumps({'resident_worker':worker.status(),'log':str(worker.log_path),'elapsed_s':time.perf_counter()-started},indent=2))
    return result
