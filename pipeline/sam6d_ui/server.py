"""Loopback-only HTTP UI with one serialized GPU worker and bounded uploads."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import shutil
from pathlib import Path
import threading
import traceback
from urllib.parse import unquote, urlparse, parse_qs
import uuid

from .inputs import options, prepare_upload
from .server_files import ServerFiles
from .camera import CameraService
from .camera.tracking import LivePoseTracker, validate_interval

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent


class App:
    def __init__(self, data_root, file_root=None, capture_root=None):
        self.files = ServerFiles(file_root or PIPELINE.parents[1])
        self.data_root = data_root.resolve()
        self.jobs_root = self.data_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.legacy_jobs_root = PIPELINE / 'prepared/sam6d_ui/jobs'
        self.camera = CameraService(capture_root if capture_root is not None else PIPELINE.parent / 'captures')
        from .backends.foundationpose import resident
        self.pose_worker = lambda: resident(self.data_root/'logs/foundationpose-worker.log')
        self.tracker = LivePoseTracker(self.camera, self.data_root/'live', resident_factory=self.pose_worker)
        self.foundationpose_status = None
        self.manifest_path = PIPELINE / "prepared/manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.jobs = {}
        self.lock = threading.Lock()
        self.submission_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.engine = None
        self.startup = {'state':'idle','message':'Precarga no solicitada','segmentor':None}

    def preload(self, segmentor='sam', backbone='vit_h'):
        with self.lock:
            self.startup={'state':'loading','message':'Precargando modelos en GPU…','segmentor':segmentor,'backbone':backbone}
        self.pool.submit(self._preload,segmentor,backbone)

    def _preload(self,segmentor,backbone):
        import time
        started=time.perf_counter();errors=[];timing={}
        def update(message):
            with self.lock:self.startup['message']=message
        try:
            update('Cargando FoundationPose en GPU…')
            worker=self.pose_worker();worker.warmup();timing['foundationpose_load_s']=worker.status()['load_s']
            self.foundationpose_status={'available':True,'cuda':True,'python':str(worker.python)}
        except Exception as error:errors.append('FoundationPose: '+str(error))
        try:
            from .engine import Engine
            self.engine=Engine(self.data_root)
            timing.update(self.engine.warmup(segmentor,backbone,update))
        except Exception as error:errors.append('SAM/DINOv2: '+str(error))
        with self.lock:
            self.startup.update(state='error' if errors else 'ready',message='; '.join(errors) if errors else 'Modelos cargados en GPU',timing={**timing,'total_s':time.perf_counter()-started})

    def viewer_status(self):
        with self.lock:
            jobs=list(self.jobs.values());startup=dict(self.startup)
            latest=next((j for j in reversed(jobs) if j['status']=='complete'),None)
            result=latest.get('result',{}) if latest else {}
            job={'id':latest['id'],'pose':bool(result.get('pose_6d_estimated')),'timing':result.get('timing',{})} if latest else None
            current=jobs[-1] if jobs else None
            message=current['message'] if current else startup['message']
        return {'startup':startup,'job':job,'message':message,'tracking':self.tracker.status(),'camera':self.camera.status()}

    def status(self):
        from .engine import environment
        from .backends.foundationpose import availability
        if self.foundationpose_status is None:
            self.foundationpose_status = availability()
        env = environment()
        names = json.loads((PIPELINE / "cad_name_corrections.json").read_text())["names"]
        return {"environment": env, "foundationpose": self.foundationpose_status,
                "startup":dict(self.startup), "mode": "local_gpu_service", "pose_available": env["pose_available"],
                "examples": [{"name": f["name"], "label": f["name"]} for f in self.manifest["frames"]],
                "catalog": [{"key": p["key"], "name": names.get(p["key"], p["name"])} for p in self.manifest["parts"]],
                "file_root": str(self.files.root),
                "max_image_pixels": 4_000_000, "cad_formats": ["STL", "OBJ", "PLY"]}

    def start_tracking(self, request):
        if not isinstance(request,dict):raise ValueError('Se requiere una petición JSON.')
        interval=validate_interval(request.get('redetect_interval_s',0))
        with self.submission_lock:
            with self.lock:
                if any(j['status'] in ('queued','running') for j in self.jobs.values()):
                    raise ValueError('Espera a que terminen los análisis pendientes.')
            job_id = request.get('job_id','')
            job = self.job(job_id)
            result = job.get('result',{})
            folder = self.job_folder(job_id)
            inputs = json.loads((folder/'inputs.json').read_text())
            if job['status'] != 'complete' or result.get('backend') != 'foundationpose' or not result.get('pose_6d_estimated') or not inputs.get('capture_id'):
                raise ValueError('Primero localiza la pieza con FoundationPose sobre la cámara en directo.')
            captured = json.loads((self.camera.capture_folder(inputs['capture_id'])/'capture.json').read_text())
            import time
            if time.time()-captured['timestamp']>120:
                raise ValueError('La pose inicial es antigua. Vuelve a localizar la pieza.')
            if interval and 'manual_bbox' in inputs:
                raise ValueError('Selecciona inicialización automática para activar los refrescos SAM.')
            from .camera.redetection import Redetector
            def engine():
                if self.engine is None:
                    from .engine import Engine
                    self.engine=Engine(self.data_root)
                return self.engine
            detector=Redetector(engine,folder) if 'manual_bbox' not in inputs else None
            return self.tracker.start(folder,redetect_interval_s=interval,redetect=detector)

    def stop_camera(self):
        self.tracker.stop()
        return self.camera.stop()

    def submit(self, request):
        with self.submission_lock:
            return self._submit(request)

    def _submit(self, request):
        if self.tracker.busy():
            raise ValueError('Detén el seguimiento rápido antes de iniciar otro análisis.')
        if not isinstance(request, dict):
            raise ValueError('Se requiere una petición JSON.')
        if request.get('continuous') and min(shutil.disk_usage(folder).free for folder in (self.data_root, self.camera.root)) < 256 * 1024**2:
            raise ValueError('Detección en directo detenida: se necesitan al menos 256 MiB libres para otro análisis.')
        manual_bbox=request.get('manual_bbox')
        if manual_bbox is not None and request.get('continuous'):
            raise ValueError('La caja manual usa una imagen congelada. Utiliza el seguimiento rápido para continuar en directo.')
        config = options(request.get('options', {}))
        if manual_bbox is not None and config['pose_backend']!='foundationpose':
            raise ValueError('La caja manual está disponible con FoundationPose.')
        request = {**request, 'options': config}
        if config['pose_backend'] == 'foundationpose':
            from .backends.foundationpose import availability
            if self.foundationpose_status is None:
                self.foundationpose_status = availability()
            if not self.foundationpose_status['available']:
                raise ValueError(self.foundationpose_status.get('reason') or 'FoundationPose no está preparado.')
        with self.lock:
            if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= 4:
                raise ValueError("Ya hay cuatro trabajos en cola. Espera a que termine uno.")
        if request.get("example"):
            frame = next((f for f in self.manifest["frames"] if f["name"] == request["example"]), None)
            part = next((p for p in self.manifest["parts"] if p["key"] == request.get("example_cad")), None)
            if not frame or not part:
                raise ValueError("Ejemplo o CAD desconocido.")
            base = self.manifest_path.parent
            def file(path):
                return {"name": Path(path).name, "data": base64.b64encode((base / path).read_bytes()).decode()}
            cad_path = (PIPELINE.parents[1] / "PDDL/Assembly CAD/Assembly parts" / (part["key"] + ".stl")).resolve()
            rgb = file(frame["rgb"])
            rgb["name"] = frame["name"] + ".png"
            request = {"options": request.get("options", {}), "rgb": rgb,
                       "cad": {"name": cad_path.name, "data": base64.b64encode(cad_path.read_bytes()).decode()},
                       "depth": file(frame["depth"]), "camera": file(frame["camera"])}
        server_paths = None
        recording_path = None
        capture_id = None
        if request.get('live_camera'):
            recording_id = request.get('recording_id')
            if recording_id:
                if config['pose_backend'] != 'foundationpose':
                    raise ValueError('El seguimiento de una grabación requiere FoundationPose.')
                recording_path = self.camera.recording_folder(recording_id)
                first = json.loads((recording_path / 'frames.jsonl').read_text().splitlines()[0])
                rgb_path, depth_path = recording_path / first['rgb'], recording_path / first['depth']
                camera_path = recording_path / 'camera.json'
            else:
                capture_id = request.get('capture_id')
                if capture_id:
                    if manual_bbox is None:
                        raise ValueError('Una captura seleccionada requiere una región manual explícita.')
                    folder = self.camera.capture_folder(capture_id)
                else:
                    capture = self.camera.capture()
                    capture_id = capture['id']
                    folder = self.camera.capture_folder(capture_id)
                rgb_path, depth_path, camera_path = folder / 'rgb.png', folder / 'depth.png', folder / 'camera.json'
            def upload(path):
                return {'name': path.name, 'data': base64.b64encode(path.read_bytes()).decode()}
            if 'server_files' in request:
                cad_path = self.files.resolve(request['server_files'].get('cad', ''))
                if not cad_path.is_file() or cad_path.suffix.lower() not in ('.stl', '.obj', '.ply') or cad_path.stat().st_size > 50 * 1024**2:
                    raise ValueError('Selecciona una malla CAD válida de hasta 50 MiB.')
                cad = upload(cad_path)
                server_paths = {'cad': str(cad_path)}
            else:
                cad = request.get('cad')
            request = {'options': config, 'cad': cad, 'rgb': upload(rgb_path),
                       'depth': upload(depth_path), 'camera': upload(camera_path)}
        if 'server_files' in request:
            files, server_paths = self.files.read_request(request['server_files'], request.get('options', {}).get('mode', 'rgb'))
            request = {**files, 'options': request.get('options', {})}
        job_id = uuid.uuid4().hex
        backend = 'foundationpose' if config['pose_backend'] == 'foundationpose' else 'sam6d'
        parent = self.jobs_root / backend
        parent.mkdir(parents=True, exist_ok=True)
        folder = parent / job_id
        inputs = prepare_upload(request, folder)
        if manual_bbox is not None:
            from .backends.manual_pose import validate_box
            inputs['manual_bbox']=validate_box(manual_bbox,inputs['rgb_size'])
        if recording_path:
            inputs['recording_path'] = str(recording_path)
        if capture_id:
            inputs['capture_id'] = capture_id
        inputs['backend'] = backend
        if server_paths:
            inputs['server_paths'] = server_paths
        (folder / 'inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "status": "queued", "message": "En cola", "inputs": inputs}
        self.pool.submit(self._run, job_id, folder, inputs)
        return {"id": job_id, "status": "queued"}

    def _run(self, job_id, folder, inputs):
        def update(message):
            with self.lock:
                self.jobs[job_id].update(status="running", message=message)
        try:
            update("Preparando el motor de inferencia")
            if 'manual_bbox' in inputs:
                from .backends.manual_pose import run as manual_pose
                result = manual_pose(folder, inputs, update)
            else:
                if self.engine is None:
                    from .engine import Engine
                    self.engine = Engine(self.data_root)
                result = self.engine.run(folder, inputs, update)
            with self.lock:
                self.jobs[job_id].update(status="complete", message="Finalizado", result=result)
        except Exception as error:
            (folder / "error.log").write_text(traceback.format_exc())
            with self.lock:
                self.jobs[job_id].update(status="error", message=str(error))

    def job(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise KeyError(job_id)
        with self.lock:
            if job_id in self.jobs:
                return dict(self.jobs[job_id])
        result = self.job_folder(job_id) / "result.json"
        if result.is_file():
            return {"id": job_id, "status": "complete", "message": "Resultado guardado", "result": json.loads(result.read_text())}
        raise KeyError(job_id)

    def job_folder(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise KeyError(job_id)
        for root in (self.jobs_root / 'sam6d', self.jobs_root / 'foundationpose', self.jobs_root, self.legacy_jobs_root):
            folder = root / job_id
            if folder.is_dir():
                return folder
        raise KeyError(job_id)


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def handle(self):
            try:super().handle()
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass

        def log_request(self,code='-',size='-'):
            if urlparse(getattr(self,'path','')).path in ('/api/camera/frame.jpg','/api/tracking/frame.jpg') and code in (200,204):return
            super().log_request(code,size)

        def send_file(self, target):
            if not target.is_file():
                self.send_error(404)
                return
            size = target.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get('Range')
            if requested:
                import re
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                if not match:
                    self.send_error(416)
                    return
                start, end = int(match[1]), min(int(match[2]) if match[2] else size - 1, size - 1)
                if start > end:
                    self.send_error(416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with target.open('rb') as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_preview(self, stream):
            # Finite responses + one browser request at a time avoid a MJPEG TCP backlog.
            import socket
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.protocol_version='HTTP/1.1'
            self.close_connection=self.request_version!='HTTP/1.1' or self.headers.get('Connection','').lower()=='close'
            try:
                after=int(parse_qs(urlparse(self.path).query).get('after',['-1'])[0])
            except ValueError:
                self.json({'error':'Número de imagen no válido.'},400);return
            frame=stream.latest_preview()
            unchanged=frame is not None and after==frame['sequence']
            self.send_response(204 if frame is None or unchanged else 200)
            self.send_header('Cache-Control','no-store, max-age=0')
            self.send_header('X-Content-Type-Options','nosniff')
            if frame is not None:
                self.send_header('X-Frame-Sequence',str(frame['sequence']))
                self.send_header('X-Frame-Age-Ms',str(round(frame['age_s']*1000,2)))
            if frame is None or unchanged:
                self.send_header('Content-Length','0');self.end_headers();return
            data=frame['jpeg']
            self.send_header('Content-Type','image/jpeg')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            try:self.wfile.write(data);self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):self.close_connection=True

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ('/api/camera/frame.jpg','/api/tracking/frame.jpg'):
                self.send_preview(app.camera if path.startswith('/api/camera/') else app.tracker)
                return
            if path == '/api/viewer/status':
                self.json(app.viewer_status())
                return
            if path == '/api/tracking/status':
                self.json(app.tracker.status())
                return
            if path == '/api/camera/status':
                self.json(app.camera.status())
                return
            if path == '/api/camera/devices':
                try:
                    from .camera.realsense import devices
                    self.json({'devices': devices()})
                except Exception as error:
                    self.json({'devices': [], 'error': str(error)})
                return
            if path == '/api/camera/recordings':
                values = []
                for item in (app.camera.root / 'videos').glob('*/recording.json'):
                    try:
                        value = json.loads(item.read_text())
                        if value.get('frames') and value.get('status') in ('complete', 'interrupted'):
                            values.append(value)
                    except (OSError, ValueError):
                        continue
                self.json({'recordings': sorted(values, key=lambda value: value.get('started_at', 0), reverse=True)[:100]})
                return
            if path in ('/api/camera/stream','/api/tracking/stream'):
                self.close_connection=True
                stream = app.tracker if path == '/api/tracking/stream' else app.camera
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                previous = -1
                try:
                    while True:
                        frame = stream.preview(previous)
                        if frame is None:
                            break
                        previous, data = frame
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path.startswith('/camera-files/'):
                base = app.camera.root
                relative = path.removeprefix('/camera-files/')
                category, separator, remainder = relative.partition('/')
                category = {'captures': 'images', 'recordings': 'videos'}.get(category, category)
                target = (base / (category + separator + remainder)).resolve()
                try:
                    target.relative_to(base)
                except ValueError:
                    self.send_error(404)
                    return
                self.send_file(target)
                return
            if path == '/api/browse':
                query = parse_qs(urlparse(self.path).query)
                try:
                    self.json(app.files.browse(query.get('path', [''])[0], query.get('kind', ['rgb'])[0]))
                except (ValueError, OSError, RuntimeError) as error:
                    self.json({'error': str(error)}, 400)
                return
            if path == "/api/status":
                self.json(app.status())
                return
            if path.startswith("/api/jobs/"):
                try:
                    self.json(app.job(path.rsplit("/", 1)[-1]))
                except KeyError:
                    self.json({"error": "Trabajo no encontrado"}, 404)
                return
            if path.startswith("/files/"):
                parts = path.removeprefix('/files/').split('/', 1)
                try:
                    base = app.job_folder(parts[0]).resolve()
                    target = (base / parts[1]).resolve()
                    target.relative_to(base)
                except (ValueError, KeyError, IndexError):
                    self.send_error(404)
                    return
            elif path in ("/", "/index.html", "/viewer", "/viewer.html", "/app.js", "/video.js", "/camera.js", "/manual.js", "/style.css"):
                target = HERE / "static" / ("index.html" if path == "/" else "viewer.html" if path == "/viewer" else path[1:])
            else:
                self.send_error(404)
                return
            self.send_file(target)

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ('/api/jobs', '/api/camera/start', '/api/camera/stop', '/api/camera/capture',
                             '/api/camera/record/start', '/api/camera/record/stop', '/api/tracking/start', '/api/tracking/stop', '/api/tracking/config'):
                self.json({"error": "Ruta no encontrada"}, 404)
                return
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.json({"error": "Origen no permitido"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json({'error': 'Content-Length no válido.'}, 400)
                return
            if not 0 < length <= 96 * 1024**2 or self.headers.get_content_type() != "application/json":
                self.json({"error": "Se requiere JSON con archivos de tamaño limitado."}, 400)
                return
            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request,dict):raise ValueError('Se requiere un objeto JSON.')
                if route == '/api/jobs':
                    self.json(app.submit(request), 202)
                else:
                    actions = {'/api/tracking/config': lambda: app.tracker.configure(request.get('redetect_interval_s')),
                               '/api/tracking/start': lambda: app.start_tracking(request),
                               '/api/tracking/stop': app.tracker.stop, '/api/camera/start': lambda: app.camera.start(request),
                               '/api/camera/stop': app.stop_camera, '/api/camera/capture': app.camera.capture,
                               '/api/camera/record/start': app.camera.start_recording,
                               '/api/camera/record/stop': app.camera.stop_recording}
                    self.json(actions[route]())
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                self.json({"error": str(error)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--data-dir", type=Path, default=PIPELINE.parent / "results/interface")
    parser.add_argument("--capture-dir", type=Path, default=PIPELINE.parent / "captures", help="Carpeta para imágenes y vídeos de cámara")
    parser.add_argument("--file-root", type=Path, default=PIPELINE.parents[1], help='Carpeta del servidor disponible en el selector')
    parser.add_argument('--preload', choices=('sam','fastsam','none'), default='sam', help='Segmentador a precargar con DINOv2 y FoundationPose; none desactiva la precarga')
    parser.add_argument('--backbone', choices=('vit_h','vit_l','vit_b'), default='vit_h', help='Variante SAM a precargar; se ignora con FastSAM')
    args = parser.parse_args()
    app = App(args.data_dir, args.file_root, args.capture_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    if args.preload!='none':app.preload(args.preload,args.backbone)
    print(f"Interfaz SAM-6D / FoundationPose: http://127.0.0.1:{args.port}/", flush=True)
    print("Servicio local para túnel SSH. Ctrl+C para detenerlo.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.tracker.stop()
        app.camera.stop()
        server.server_close()
        app.pool.shutdown(wait=True)
        from .backends.foundationpose import _resident
        if _resident is not None:_resident.close()
