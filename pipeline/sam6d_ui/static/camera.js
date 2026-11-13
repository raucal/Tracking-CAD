/* The camera runs on the server, shared by live view, captures and recording. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
let live=false,active=false,recording=false,pending=false,fastActive=false,fastPreparing=false,fastGeneration=0,fastPolling=false;
async function api(path,payload){
  const response=await fetch('/api/camera/'+path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not complete the camera operation.');return data;
}
function note(text,error=false){$('camera-status').textContent=text;$('camera-status').classList.toggle('error',error);}
function selectionChanged(){
  const manual=!!window.VisionManual?.active();
  $('sam-refresh').disabled=manual||fastPreparing;
  $('sam-refresh-note').textContent=manual?'FoundationPose only: SAM refreshes are disabled. Select automatic initialization to use them.':'SAM recovers immediately after tracking loss. This interval controls only periodic backup; 0 disables it while keeping automatic recovery. One pending detection; models stay loaded on the GPU.';
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
  if(!fastActive||window.visionCompact){window.VisionVideo.stop($('cad-preview'));$('cad-preview').hidden=true;$('cad-view').hidden=true;}
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
    if((!active||data.error)&&window.VisionApp?.isLiveDetection())window.VisionApp.stopLiveDetection(data.error||'Detection stopped: camera closed.');
    if(data.error)note(data.error,true);
    else if(data.frame_warning)note('Camera remains open · '+data.frame_warning,true);
    else if(recording)note('Recording RGB-D · '+data.recording.frames+' frames saved.');
    else if(data.last_recording?.error)note('Recording stopped: '+data.last_recording.error,true);
    else if(active)note('Live camera · frame '+data.frame_number+' · '+(data.camera?.serial||''));
    else if(!pending)note('Camera closed. Saved captures and recordings remain available.');
    stream();controls();
  }catch(error){note(error.message,true);if(window.VisionApp?.isLiveDetection())window.VisionApp.stopLiveDetection('Could not query camera status.');}
}
async function devices(){
  try{
    const data=await api('devices');const selected=$('camera-device').value;$('camera-device').replaceChildren();
    for(const item of data.devices){const option=new Option(item.name+' · '+item.serial,item.serial);$('camera-device').add(option);}
    if(!$('camera-device').options.length)$('camera-device').add(new Option('No RealSense detected',''));
    if([...$('camera-device').options].some(option=>option.value===selected))$('camera-device').value=selected;
    if(data.error)note(data.error,true);
  }catch(error){note(error.message,true);}
}
async function recordings(selected){
  try{
    const data=await api('recordings');selected=selected||$('camera-recording').value;
    $('camera-recording').replaceChildren(new Option('Select a recording',''));
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
$('camera-stop').onclick=async()=>{await stopFast();if(window.VisionApp?.isLiveDetection())window.VisionApp.stopLiveDetection('Detection stopped when closing the camera.');await action('stop');await recordings();};
async function capture(exportImage=false){
  const data=await action('capture');if(!data)return;
  links('/camera-files/captures/'+data.id+'/',[['rgb.png','Download PNG image'],['depth.png','uint16 depth'],['camera.json','Calibration'],['capture.json','Metadata']]);
  if(exportImage)$('camera-downloads').querySelector('a').click();
  note(exportImage?'PNG image ready for download; depth and calibration are also retained.':'RGB-D capture saved. Analyse current frame takes a fresh image.');
}
$('camera-capture').onclick=()=>capture();
$('camera-export').onclick=()=>capture(true);
function showRecording(id){
  $('camera-playback').hidden=!id;
  if(!id){$('camera-playback').removeAttribute('src');return;}
  const base='/camera-files/recordings/'+id+'/';
  links(base,[['video.mp4','Download MP4 video'],['camera.json','Calibration'],['frames.jsonl','Frame index'],['recording.json','Video metadata']]);
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
  const result=await response.json();if(!response.ok)throw new Error(result.error||'Tracking error.');return result;
}
async function stopFast(){
  fastGeneration++;const wasActive=fastActive;fastPreparing=false;fastActive=false;controls();
  if(wasActive){try{await trackingApi('stop',{});}catch(error){$('fast-status').textContent=error.message;return;}}
  window.VisionVideo.stop($('fast-preview'));$('fast-preview').hidden=true;
  $('fast-status').textContent='Tracking stopped. You can localize the part again.';
  window.AssemblyIdentification?.error('Tracking stopped. The displayed part is the last identified one.');
}
$('fast-stop').onclick=stopFast;
$('fast-start').onclick=async()=>{
  const interval=Number($('sam-refresh').value);
  if(!window.VisionManual?.active()&&(!Number.isFinite(interval)||(interval!==0&&(interval<1||interval>3600)))){$('fast-status').textContent='The interval must be 0 or between 1 and 3600 seconds.';return;}
  const generation=++fastGeneration;fastPreparing=true;controls();
  $('fast-status').textContent=window.VisionManual?.active()?'Estimating pose from the manual box. Keep the scene stationary.':'Localizing the part with SAM6D and FoundationPose. Keep the scene stationary during initialization.';
  try{
    const id=await window.VisionApp.initializeFastTracking();
    if(generation!==fastGeneration)return;
    if(!id)throw new Error('No initial pose was obtained. Read the result message.');
    const state=await trackingApi('start',{job_id:id,redetect_interval_s:window.VisionManual?.active()?0:Number($('sam-refresh').value)});
    if(generation!==fastGeneration){await trackingApi('stop',{});return;}
    fastActive=state.running;fastPreparing=false;controls();
    $('fast-status').textContent='Starting tracking with the loaded models…';
  }catch(error){if(generation===fastGeneration){fastActive=false;fastPreparing=false;controls();$('fast-status').textContent=error.message;}}
};
async function trackingStatus(){
  if(!fastActive||fastPolling)return;fastPolling=true;
  try{
    const data=await trackingApi('status');
    if(!fastActive)return;
    window.AssemblyIdentification?.tracking(data);
    $('cad-view').hidden=!!window.visionCompact||!data.running;
    if(data.running&&data.state==='tracking'&&!data.cad_view_error&&!window.visionCompact){
      window.VisionVideo.play($('cad-preview'),'cad');
      $('cad-view-status').textContent='Current CAD pose · X red · Y green · Z blue'+(Number.isFinite(data.cad_render_s)?' · render '+Math.round(data.cad_render_s*1000)+' ms':'');
    }else{
      window.VisionVideo.stop($('cad-preview'));$('cad-preview').hidden=true;
      $('cad-view-status').textContent=data.cad_view_error?'Failed to render the CAD: '+data.cad_view_error:'Waiting for a valid pose; the view updates when tracking recovers.';
    }
    if(!data.running){fastActive=false;controls();window.VisionVideo.stop($('fast-preview'));$('fast-status').textContent=data.error||'Tracking stopped.';return;}
    const interval=data.redetect_interval_s||0;
    const refresh=interval?' · SAM backup every '+interval+' s · '+(data.redetections||0)+' recoveries · '+(data.sam_checks||0)+' checks'+(data.sam_running?' · background SAM':''):(data.automatic_recovery?' · SAM after tracking loss; periodic backup disabled':' · manual initialization; no SAM recovery');
    let source='/api/tracking/stream';
    if(data.state==='tracking'){
      $('fast-status').textContent=data.fps.toFixed(1)+' tracking FPS · latency '+Math.round(data.latency_s*1000)+' ms'+refresh+(data.last_detection_score!==undefined?' · latest score '+data.last_detection_score.toFixed(3):'')+(data.redetect_error?' · '+data.redetect_error:'');
    }else if(data.state==='checking'){
      source='/api/camera/stream';$('fast-status').textContent='Checking depth and pose stability ('+(data.consecutive_quality_failures||0)+') · '+(data.redetect_error||'');
    }else if(data.state==='redetecting'){
      source='/api/camera/stream';$('fast-status').textContent='Recovering pose with FoundationPose…'+refresh;
    }else if(data.state==='recovering'){
      source='/api/camera/stream';$('fast-status').textContent=(data.redetect_error||'Pose perdida.')+refresh;
    }else{$('fast-status').textContent='Preparing tracking…';return;}
    if(data.identification){
      $('fast-status').textContent+=' · Part: '+(data.display_name||'unidentified')+' · CAD changes: '+(data.component_changes||0);
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
  if(!Number.isFinite(value)||(value!==0&&(value<1||value>3600))){$('fast-status').textContent='The interval must be 0 or between 1 and 3600 seconds.';return;}
  if(fastActive){try{await trackingApi('config',{redetect_interval_s:value});await trackingStatus();}catch(error){$('fast-status').textContent=error.message;}}
};
for(const [image,status] of [['camera-preview','camera-video-status'],['fast-preview','fast-video-status']]){
  $(image).addEventListener('vision-frame',event=>{const d=event.detail;$(status).textContent='Received video: '+(d.fps===null?'…':d.fps.toFixed(1)+' FPS')+' · antigüedad estimada '+Math.round(d.ageMs)+' ms';});
  $(image).addEventListener('vision-video-error',event=>{$(status).textContent=event.detail.message;});
}
setInterval(trackingStatus,300);
window.VisionCamera={sourceChanged:value=>{if(!value&&(fastActive||fastPreparing))stopFast();const entered=value&&!live;live=value;controls();if(entered){devices();recordings();status();}},selectionChanged,busyChanged:controls};
setInterval(()=>{if(live||active)status();},2000);
controls();
})();
