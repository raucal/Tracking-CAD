/* La cámara vive en el servidor, compartida por directo, capturas y grabación. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
let live=false,active=false,recording=false,pending=false,fastActive=false,fastPreparing=false,fastGeneration=0,fastPolling=false;
async function api(path,payload){
  const response=await fetch('/api/camera/'+path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'No se pudo completar la operación de cámara.');return data;
}
function note(text,error=false){$('camera-status').textContent=text;$('camera-status').classList.toggle('error',error);}
function selectionChanged(){
  const manual=!!window.VisionManual?.active();
  $('sam-refresh').disabled=manual||fastPreparing;
  $('sam-refresh-note').textContent=manual?'Solo FoundationPose: los refrescos SAM están desactivados. Selecciona inicialización automática para utilizarlos.':'SAM recupera inmediatamente al perder el seguimiento. Este intervalo controla solo el respaldo periódico; 0 lo desactiva y mantiene la recuperación automática. Una sola detección pendiente; modelos cargados en GPU.';
  if($('fast-start'))$('fast-start').disabled=pending||!live||!active||fastActive||fastPreparing||!!window.VisionApp?.isBusy()||!!window.VisionApp?.isLiveDetection()||$('pose-backend').value!=='foundationpose';
  $('track-recording').disabled=fastActive||fastPreparing||pending||!!window.VisionApp?.isBusy()||!!window.VisionApp?.isLiveDetection()||!$('camera-recording').value||$('pose-backend').value!=='foundationpose';
}
function controls(){
  $('live-camera').hidden=!live&&!active;
  $('camera-start').disabled=pending||active;
  $('fast-start').disabled=pending||!live||!active||fastPreparing||fastActive||!!window.VisionApp?.isBusy()||!!window.VisionApp?.isLiveDetection()||$('pose-backend').value!=='foundationpose';
  $('fast-stop').disabled=!fastActive&&!fastPreparing;
  $('run').disabled=$('run-example').disabled=fastActive||fastPreparing||!!window.VisionApp?.isBusy()||!!window.VisionApp?.isLiveDetection();
  $('live-detection-start').disabled=!!window.VisionManual?.active()||fastActive||fastPreparing||pending||!live||!active||!!window.VisionApp?.isBusy()||!!window.VisionApp?.isLiveDetection();
  $('live-detection-stop').disabled=!window.VisionApp?.isLiveDetection();
  $('camera-stop').disabled=pending||!active;
  $('camera-capture').disabled=pending||!active;
  $('camera-export').disabled=pending||!active;
  $('record-start').disabled=pending||!active||recording;
  $('record-stop').disabled=pending||!recording;
  $('record-start').classList.toggle('recording',recording);
  for(const id of ['camera-device','camera-resolution','camera-fps'])$(id).disabled=active||pending;
  selectionChanged();stream();
  if(window.visionCompact){window.VisionVideo.stop($('fast-preview'));$('fast-preview').hidden=true;}
}
function stream(){
  if(active&&!window.visionCompact&&!fastActive){
    window.VisionVideo.play($('camera-preview'),'camera');$('camera-preview').hidden=false;
  }else{window.VisionVideo.stop($('camera-preview'));$('camera-preview').hidden=true;}
}

async function status(){
  try{
    const data=await api('status');active=data.running;recording=!!data.recording;
    if((!active||data.error)&&window.VisionApp?.isLiveDetection())window.VisionApp.stopLiveDetection(data.error||'Detección detenida: cámara cerrada.');
    if(data.error)note(data.error,true);
    else if(recording)note('Grabando RGB-D · '+data.recording.frames+' fotogramas guardados.');
    else if(data.last_recording?.error)note('Grabación detenida: '+data.last_recording.error,true);
    else if(active)note('Directo activo · fotograma '+data.frame_number+' · '+(data.camera?.serial||''));
    else if(!pending)note('Cámara cerrada. Las capturas y grabaciones guardadas permanecen disponibles.');
    stream();controls();
  }catch(error){note(error.message,true);if(window.VisionApp?.isLiveDetection())window.VisionApp.stopLiveDetection('No se puede consultar el estado de la cámara.');}
}
async function devices(){
  try{
    const data=await api('devices');const selected=$('camera-device').value;$('camera-device').replaceChildren();
    for(const item of data.devices){const option=new Option(item.name+' · '+item.serial,item.serial);$('camera-device').add(option);}
    if(!$('camera-device').options.length)$('camera-device').add(new Option('No hay RealSense detectada',''));
    if([...$('camera-device').options].some(option=>option.value===selected))$('camera-device').value=selected;
    if(data.error)note(data.error,true);
  }catch(error){note(error.message,true);}
}
async function recordings(selected){
  try{
    const data=await api('recordings');selected=selected||$('camera-recording').value;
    $('camera-recording').replaceChildren(new Option('Selecciona una grabación',''));
    for(const row of data.recordings){
      const label=new Date(row.started_at*1000).toLocaleString()+' · '+row.frames+' frames · '+row.status;
      $('camera-recording').add(new Option(label,row.id));
    }
    $('camera-recording').value=selected||'';selectionChanged();
  }catch(error){note(error.message,true);}
}
async function action(name,payload={}){
  if(pending)return null;pending=true;controls();let failure=null;
  try{return await api(name,payload);}catch(error){failure=error.message;return null;}
  finally{pending=false;await status();controls();if(failure)note(failure,true);}
}
function links(base,names){
  $('camera-downloads').replaceChildren();
  for(const [name,label]of names){const link=document.createElement('a');link.href=base+name;link.download=name;link.textContent=label;$('camera-downloads').append(link);}
}
$('camera-start').onclick=async()=>{
  const [width,height]=$('camera-resolution').value.split('x').map(Number);
  await action('start',{serial:$('camera-device').value,width,height,fps:Number($('camera-fps').value)});
};
$('live-detection-start').onclick=()=>window.VisionApp.startLiveDetection();
$('live-detection-stop').onclick=()=>window.VisionApp.stopLiveDetection();
$('camera-stop').onclick=async()=>{await stopFast();if(window.VisionApp?.isLiveDetection())window.VisionApp.stopLiveDetection('Detección detenida al cerrar la cámara.');await action('stop');await recordings();};
async function capture(exportImage=false){
  const data=await action('capture');if(!data)return;
  links('/camera-files/captures/'+data.id+'/',[['rgb.png','Descargar imagen PNG'],['depth.png','Profundidad uint16'],['camera.json','Calibración'],['capture.json','Metadatos']]);
  if(exportImage)$('camera-downloads').querySelector('a').click();
  note(exportImage?'Imagen PNG lista para descargar; también se ha conservado su profundidad y calibración.':'Captura RGB-D guardada. Analizar fotograma actual tomará una imagen nueva.');
}
$('camera-capture').onclick=()=>capture();
$('camera-export').onclick=()=>capture(true);
function showRecording(id){
  $('camera-playback').hidden=!id;
  if(!id){$('camera-playback').removeAttribute('src');return;}
  const base='/camera-files/recordings/'+id+'/';
  links(base,[['video.mp4','Descargar vídeo MP4'],['camera.json','Calibración'],['frames.jsonl','Índice de fotogramas'],['recording.json','Metadatos de vídeo']]);
  $('camera-playback').src=base+'video.mp4';
}
$('record-start').onclick=()=>action('record/start');
$('record-stop').onclick=async()=>{
  const data=await action('record/stop');if(!data)return;
  await recordings(data.id);
  showRecording(data.id);
};
$('camera-recording').onchange=()=>{
  selectionChanged();const id=$('camera-recording').value;
  showRecording(id);
};
$('track-recording').onclick=()=>window.VisionApp.analyzeRecording($('camera-recording').value);
async function trackingApi(path,payload){
  const response=await fetch('/api/tracking/'+path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'Error de seguimiento.');return result;
}
async function stopFast(){
  fastGeneration++;const wasActive=fastActive;fastPreparing=false;fastActive=false;controls();
  if(wasActive){try{await trackingApi('stop',{});}catch(error){$('fast-status').textContent=error.message;return;}}
  window.VisionVideo.stop($('fast-preview'));$('fast-preview').hidden=true;
  $('fast-status').textContent='Seguimiento detenido. Puedes volver a localizar la pieza.';
  window.AssemblyIdentification?.error('Seguimiento detenido. La pieza mostrada corresponde a la última identificación.');
}
$('fast-stop').onclick=stopFast;
$('fast-start').onclick=async()=>{
  const interval=Number($('sam-refresh').value);
  if(!window.VisionManual?.active()&&(!Number.isFinite(interval)||(interval!==0&&(interval<1||interval>3600)))){$('fast-status').textContent='El intervalo debe ser 0 o estar entre 1 y 3600 segundos.';return;}
  const generation=++fastGeneration;fastPreparing=true;controls();
  $('fast-status').textContent=window.VisionManual?.active()?'Estimando pose desde la caja manual. Mantén estable la escena.':'Localizando la pieza con SAM6D y FoundationPose. Mantén la escena estable durante el inicio.';
  try{
    const id=await window.VisionApp.initializeFastTracking();
    if(generation!==fastGeneration)return;
    if(!id)throw new Error('No se obtuvo una pose inicial. Consulta el mensaje del resultado.');
    const state=await trackingApi('start',{job_id:id,redetect_interval_s:window.VisionManual?.active()?0:Number($('sam-refresh').value)});
    if(generation!==fastGeneration){await trackingApi('stop',{});return;}
    fastActive=state.running;fastPreparing=false;controls();
    $('fast-status').textContent='Iniciando seguimiento con los modelos ya cargados…';
  }catch(error){if(generation===fastGeneration){fastActive=false;fastPreparing=false;controls();$('fast-status').textContent=error.message;}}
};
async function trackingStatus(){
  if(!fastActive||fastPolling)return;fastPolling=true;
  try{
    const data=await trackingApi('status');
    if(!fastActive)return;
    window.AssemblyIdentification?.tracking(data);
    if(!data.running){fastActive=false;controls();window.VisionVideo.stop($('fast-preview'));$('fast-status').textContent=data.error||'Seguimiento detenido.';return;}
    const interval=data.redetect_interval_s||0;
    const refresh=interval?' · respaldo SAM cada '+interval+' s · '+(data.redetections||0)+' recuperaciones · '+(data.sam_checks||0)+' comprobaciones'+(data.sam_running?' · SAM en segundo plano':''):(data.automatic_recovery?' · SAM al perder seguimiento; respaldo desactivado':' · inicialización manual; sin recuperación SAM');
    let source='/api/tracking/stream';
    if(data.state==='tracking'){
      $('fast-status').textContent=data.fps.toFixed(1)+' FPS de seguimiento · latencia '+Math.round(data.latency_s*1000)+' ms'+refresh+(data.last_detection_score!==undefined?' · último score '+data.last_detection_score.toFixed(3):'')+(data.redetect_error?' · '+data.redetect_error:'');
    }else if(data.state==='checking'){
      source='/api/camera/stream';$('fast-status').textContent='Comprobando profundidad y estabilidad de la pose ('+(data.consecutive_quality_failures||0)+') · '+(data.redetect_error||'');
    }else if(data.state==='redetecting'){
      source='/api/camera/stream';$('fast-status').textContent='Recuperando pose con FoundationPose…'+refresh;
    }else if(data.state==='recovering'){
      source='/api/camera/stream';$('fast-status').textContent=(data.redetect_error||'Pose perdida.')+refresh;
    }else{$('fast-status').textContent='Preparando seguimiento…';return;}
    if(data.identification){
      $('fast-status').textContent+=' · Pieza: '+(data.display_name||'sin identificar')+' · cambios de CAD: '+(data.component_changes||0);
      if(data.cad_key&&$('camera-cad-source').value==='assembly'){
        $('assembly-part').value=data.cad_key;$('assembly-part').onchange?.();
      }
    }
    if(!window.visionCompact)window.VisionVideo.play($('fast-preview'),source.includes('/tracking/')?'tracking':'camera');

  }catch(error){$('fast-status').textContent=error.message;}finally{fastPolling=false;}
}
$('form').addEventListener('change',event=>{if((fastActive||fastPreparing)&&!['camera-recording','sam-refresh'].includes(event.target.id))stopFast();});
$('sam-refresh').onchange=async()=>{
  const value=Number($('sam-refresh').value);
  if(!Number.isFinite(value)||(value!==0&&(value<1||value>3600))){$('fast-status').textContent='El intervalo debe ser 0 o estar entre 1 y 3600 segundos.';return;}
  if(fastActive){try{await trackingApi('config',{redetect_interval_s:value});await trackingStatus();}catch(error){$('fast-status').textContent=error.message;}}
};
for(const [image,status] of [['camera-preview','camera-video-status'],['fast-preview','fast-video-status']]){
  $(image).addEventListener('vision-frame',event=>{const d=event.detail;$(status).textContent='Vídeo recibido: '+(d.fps===null?'…':d.fps.toFixed(1)+' FPS')+' · antigüedad estimada '+Math.round(d.ageMs)+' ms';});
  $(image).addEventListener('vision-video-error',event=>{$(status).textContent=event.detail.message;});
}
setInterval(trackingStatus,300);
window.VisionCamera={sourceChanged:value=>{if(!value&&(fastActive||fastPreparing))stopFast();const entered=value&&!live;live=value;controls();if(entered){devices();recordings();status();}},selectionChanged,busyChanged:controls};
setInterval(()=>{if(live||active)status();},2000);
controls();
})();
