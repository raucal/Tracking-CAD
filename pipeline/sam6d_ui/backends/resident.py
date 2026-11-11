"""Serialized RPC to a single persistent FoundationPose process."""
import json
from pathlib import Path
from queue import Queue, Empty
import subprocess
import threading
import time
import uuid


class ResidentPose:
    def __init__(self,python,worker,log_path):
        self.python=Path(python);self.worker=Path(worker);self.log_path=Path(log_path)
        self.lock=threading.RLock();self.process=None;self.reader=None;self.log=None;self.messages=None
        self.ready=False;self.load_s=None
    def _receive(self,timeout):
        try:item=self.messages.get(timeout=timeout)
        except Empty:raise RuntimeError('The pose worker did not respond in time.')
        if item is None:raise RuntimeError('The pose worker closed. See '+str(self.log_path))
        return item
    def _start(self):
        if self.process and self.process.poll() is None and self.ready:return
        self.close();self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.log_path.open('a');self.messages=Queue()
        self.process=subprocess.Popen([str(self.python),'-B',str(self.worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        process=self.process;messages=self.messages;log=self.log
        def read():
            try:
                for line in process.stdout:
                    if line.startswith('@POSE@'):
                        try:messages.put(json.loads(line[6:]))
                        except ValueError:pass
                    elif not log.closed:log.write(line);log.flush()
            finally:messages.put(None)
        self.reader=threading.Thread(target=read,name='FoundationPose-RPC',daemon=True);self.reader.start()
        try:
            result=self._receive(180)
            if result.get('status')!='ready':raise RuntimeError(result.get('error','Failed to load FoundationPose.'))
            self.ready=True;self.load_s=result['load_s']
        except BaseException:self.close();raise
    def request(self,command,timeout=300,**payload):
        with self.lock:
            self._start();rid=uuid.uuid4().hex
            started=time.perf_counter()
            try:
                self.process.stdin.write(json.dumps(dict(id=rid,command=command,**payload))+'\n');self.process.stdin.flush()
                response=self._receive(timeout)
                if response.get('request_id')!=rid:raise RuntimeError('Out-of-sequence response from the pose worker.')
            except (OSError,RuntimeError):self.close();raise
            if response['status']=='error':raise RuntimeError(response.get('error','Pose error.'))
            result=response['result'];result['rpc_seconds']=time.perf_counter()-started
            return result
    def warmup(self):return self.request('warmup')
    def start_tracking(self,folder):return self.request('start_tracking',folder=str(folder))
    def track(self,scratch):return self.request('track',timeout=30,scratch=str(scratch))
    def relocalize(self,scratch):return self.request('relocalize',scratch=str(scratch))
    def status(self):return {'ready':bool(self.ready and self.process and self.process.poll() is None),'pid':self.process.pid if self.process else None,'load_s':self.load_s}
    def close(self):
        with self.lock:
            if self.process:
                if self.process.poll() is None:
                    try:self.process.stdin.close()
                    except OSError:pass
                    try:self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                if self.reader:self.reader.join(3)
                for stream in (self.process.stdin,self.process.stdout):
                    if stream:stream.close()
            if self.log:self.log.close()
            self.process=None;self.reader=None;self.log=None;self.ready=False
