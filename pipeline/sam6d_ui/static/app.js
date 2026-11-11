(()=>{'use strict';
const $=id=>document.getElementById(id),escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let running=false,liveDetection=false,liveGeneration=0,liveTimer=null,liveCompleted=0,lastJobId=null;
const config=()=>({pose_backend:$('pose-backend').value,mode:$('mode').value,cad_units:$('cad_units').value,device:$('device-select').value,backbone:$('backbone').value,segmentor:$('segmentor').value,pose:$('pose').checked,fastsam_conf:Number($('fastsam_conf').value),fastsam_iou:Number($('fastsam_iou').value),
 min_score:Number($('min_score').value),confidence:Number($('confidence').value),stability:Number($('stability').value),pred_iou:Number($('pred_iou').value),points_per_batch:Number($('points_per_batch').value),nms:Number($('nms').value),aggregation:$('aggregation').value});
async function file(id){const f=$(id).files[0];if(!f)throw new Error('Missing file: '+id);return{name:f.name,data:await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=reject;r.readAsDataURL(f);})};}
function message(text,error=false){$('progress').textContent=text;$('progress').classList.toggle('error',error);}
function busy(value){running=value;$('run').disabled=$('run-example').disabled=value||liveDetection;window.VisionCamera?.busyChanged();window.VisionManual?.sync();}
async function submit(request){
  if(running)return false;
  busy(true);if(!liveDetection)$('output').hidden=true;
  if(request.auto_assembly)window.AssemblyIdentification?.pending();
  message('Validating files and submitting the job…');
  try{
    const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    const data=await response.json();if(!response.ok)throw new Error(data.error);
    await poll(data.id);lastJobId=data.id;return true;
  }catch(error){message(error.message,true);if(request.auto_assembly)window.AssemblyIdentification?.error(error.message);return false;}
  finally{busy(false);}
}
async function poll(id){
  for(;;){
    const response=await fetch('/api/jobs/'+encodeURIComponent(id));const job=await response.json();
    if(!response.ok)throw new Error(job.error);
    message(job.message);
    if(job.status==='complete'){
      window.AssemblyIdentification?.result(job.result);
      if(job.result.identification?.part_key&&$('assembly-select').value===job.result.identification.assembly){
        $('assembly-part').value=job.result.identification.part_key;
        $('assembly-part').onchange?.();
      }
      show(id,job.result);return;
    }
    if(job.status==='error')throw new Error(job.message+' · Details available at /files/'+id+'/error.log');
    await new Promise(resolve=>setTimeout(resolve,200));
  }
}
function liveNote(text){$('live-detection-status').textContent=text;}
function stopLiveDetection(reason='Live detection stopped.'){
  liveDetection=false;liveGeneration++;clearTimeout(liveTimer);liveTimer=null;busy(running);
  liveNote(reason+(running?' The submitted analysis will finish and retain its result.':''));
}
async function liveCycle(generation){
  if(!liveDetection||generation!==liveGeneration)return;
  const cycleStarted=performance.now();
  try{
    if($('file-source').value!=='realsense')throw new Error('Select live Intel RealSense.');
    liveNote('Analysing a current frame. Camera streaming continues.');
    const request=await buildRequest();request.continuous=true;
    if(!liveDetection||generation!==liveGeneration)return;
    const ok=await submit(request);
    if(!liveDetection||generation!==liveGeneration)return;
    if(!ok){stopLiveDetection('Detection stopped because of an error. See the result message.');return;}
    liveCompleted++;
    liveNote(liveCompleted+' analyses completed · latest result at '+new Date().toLocaleTimeString()+'.');
    const interval=$('camera-cad-source').value==='assembly'?Number($('sam-refresh').value)*1000:250;
    if($('camera-cad-source').value==='assembly'&&interval===0){stopLiveDetection('Identification completed. Interval 0: repetition disabled.');return;}
    if($('camera-cad-source').value==='assembly'&&(!Number.isFinite(interval)||interval<0||(interval>0&&interval<1000)||interval>3600000))throw new Error('The SAM interval must be 0 or between 1 and 3600 seconds.');
    const delay=$('camera-cad-source').value==='assembly'?interval-((performance.now()-cycleStarted)%interval):interval;
    liveTimer=setTimeout(()=>liveCycle(generation),delay);
  }catch(error){message(error.message,true);stopLiveDetection('Detection stopped: '+error.message);}
}
function startLiveDetection(){
  if(window.VisionManual?.active()){message('The manual box uses a frozen frame. Start fast tracking for live updates.',true);return;}
  if($('camera-cad-source').value==='assembly'){
    const interval=Number($('sam-refresh').value);
    if(!Number.isFinite(interval)||(interval!==0&&(interval<1||interval>3600))){message('The interval must be 0 or between 1 and 3600 seconds.',true);return;}
  }
  if(running||liveDetection)return;
  liveDetection=true;liveCompleted=0;const generation=++liveGeneration;busy(false);liveCycle(generation);
}
function show(id,result){const base='/files/'+id+'/';const manual=result.initialization==='manual_bbox';$('score-label').textContent=manual?'Automatic detection: not used':'SAM6D detection score';$('output').hidden=false;$('score').textContent=result.best_score===null?'—':result.best_score.toFixed(3);$('score').style.color=result.recognized?'#177867':'#9b6a21';$('time').textContent=(result.timing.inference_s+(result.timing.pose_pipeline_s||0)).toFixed(2)+' s';$('actual-device').textContent=result.device.toUpperCase();$('original').src=base+'input_rgb.png';$('overlay').src=base+'overlay.png';message(manual?'Pose estimated from the manual region; SAM6D detection was not run.':result.recognized?'Object detected: '+result.accepted_detections+' region(s) above threshold.':'No detections exceed the acceptance threshold. Lower-scoring candidates are retained.');$('result-note').textContent=(manual?'Standalone FoundationPose, initialized with a manual rectangular mask.':result.mode==='rgb'?'RGB mode without depth: similarity and appearance, without geometry.':'RGB-D mode: ISM score with appearance and geometry.')+' CAD: '+result.display_name+(result.pose_6d_estimated?'. 6D pose estimated with '+(result.backend==='foundationpose'?'FoundationPose':'PEM')+'.':'. No 6D pose estimated.');$('downloads').innerHTML=`<a href="${base}result.json" download>Download result JSON</a><a href="${base}input_rgb.png" download>Export original image</a><a href="${base}overlay.png" download>Export image with detections</a><a href="${base}detections.json" download>RLE masks</a>`;$('detections').innerHTML=result.detections.map((d,i)=>`<tr><td>${i+1}</td><td>${d.score.toFixed(3)}</td><td class="${d.score>=result.options.confidence?'accepted':'low'}">${d.score>=result.options.confidence?'Accepted':'Below threshold'}</td><td><a href="${base}mask_${String(i).padStart(3,'0')}.png" target="_blank">View mask</a></td></tr>`).join('')||'<tr><td colspan="4">No retained detections.</td></tr>';showPose(base,result);$('technical').textContent=JSON.stringify({timing_s:result.timing,inference_stages:result.inference_stages,model_reused:result.model_reused,reference_cache_hit:result.reference_cache_hit,renderer:result.renderer,formula:result.formula,options:result.options},null,2);}
function showPose(base,result){const pose=result.pose;$('tracking-video').hidden=true;$('pose-output').hidden=!result.pose_6d_estimated;if(result.pose_6d_estimated){const best=pose.poses[pose.best_index||0];$('pose-overlay').src=base+'pose_overlay.png';$('pose-values').textContent=JSON.stringify({method:result.backend||'pem',axes_frame:pose.visualization?.axes_frame||'estimator model',CAD_rotation:best.T_camera_from_input_cad_mm?.slice(0,3).map(row=>row.slice(0,3))||best.R,CAD_translation_mm:best.T_camera_from_input_cad_mm?.slice(0,3).map(row=>row[3])||best.t_mm,T_camera_from_CAD:best.T_camera_from_input_cad_mm,T_camera_from_model:best.T_camera_from_model_mm,timing_s:pose.timing},null,2);$('downloads').innerHTML+=`<a href="${base}pose.json" download>Pose 6D JSON</a><a href="${base}pose_overlay.png" download>Pose projection</a>`;if(pose.tracking){$('tracking-video').hidden=false;$('tracking-video').src=base+pose.tracking.video;$('downloads').innerHTML+=`<a href="${base}pose_tracking.jsonl" download>Video poses JSONL</a><a href="${base}pose_tracking.mp4" download>Video with pose</a>`;}}else if(result.options.pose){message('Pose not calculated: '+(pose?.status||'no result'));}}
$('pose').onchange=()=>{if($('pose').checked){$('mode').value='rgbd';$('device-select').value='cuda';$('mode').onchange();}};
$('segmentor').onchange=()=>{$('backbone').disabled=$('segmentor').value==='fastsam';};
$('mode').onchange=()=>{$('depth-files').hidden=$('mode').value!=='rgbd';if($('mode').value==='rgb')$('pose').checked=false;$('mode-note').textContent=$('mode').value==='rgbd'?'Aligned RGB-D and calibrated camera. The SAM-6D/BOP JSON scale is millimetres per raw unit.':'Without depth: similarity and appearance score, not the full SAM-6D geometric score.';};
let serverRoot='' ,browseKind='rgb',browseTarget=null,browseParent=null,browseSeq=0;
async function browse(path){const seq=++browseSeq;$('browser-status').textContent='Reading server directory…';$('browser-entries').replaceChildren();try{const response=await fetch('/api/browse?'+new URLSearchParams({path,kind:browseKind}));const data=await response.json();if(seq!==browseSeq)return;if(!response.ok)throw new Error(data.error);$('browser-path').value=data.path;browseParent=data.parent;$('browser-up').disabled=!data.parent;for(const entry of data.entries){const button=document.createElement('button');button.type='button';button.className=entry.directory?'folder':'server-file';button.textContent=(entry.directory?'Directory · ':'File · ')+entry.name;button.dataset.path=entry.path;button.onclick=()=>{if(entry.directory)browse(entry.path);else{$(browseTarget).value=entry.path;$('server-browser').close();}};$('browser-entries').append(button);}$('browser-status').textContent=data.truncated?'Showing the first 1000 entries. Enter a subdirectory to continue.':data.entries.length+' entries · Valid formats only for '+browseKind;}catch(error){if(seq===browseSeq)$('browser-status').textContent=error.message;}}
document.querySelectorAll('.browse-server').forEach(button=>button.onclick=()=>{browseKind=button.dataset.kind;browseTarget=button.dataset.target;const previous=$(browseTarget).value.trim();const folder=previous?previous.slice(0,previous.lastIndexOf('/')):serverRoot;$('server-browser').showModal();browse(folder);});
$('browser-close').onclick=()=>$('server-browser').close();$('browser-go').onclick=()=>browse($('browser-path').value);$('browser-up').onclick=()=>{if(browseParent)browse(browseParent);};$('browser-path').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();browse($('browser-path').value);}};
document.querySelectorAll('[data-location]').forEach(button=>button.onclick=()=>browse(serverRoot+(button.dataset.location?'/'+button.dataset.location:'')));
$('run-example').onclick=()=>submit({example:$('example-frame').value,example_cad:$('example-cad').value,options:config()});
fetch('/api/status').then(r=>r.json()).then(data=>{const env=data.environment;if(data.startup?.segmentor){$('segmentor').value=data.startup.segmentor;$('backbone').value=data.startup.backbone||'vit_h';$('segmentor').onchange();}window.visionEnvironment=data;
const assemblies=data.assemblies||[];
$('assembly-select').replaceChildren(...assemblies.map(a=>new Option(a.label,a.key)));
function chooseAssemblyPart(){const a=assemblies.find(a=>a.key===$('assembly-select').value);const p=a?.parts.find(p=>p.key===$('assembly-part').value);if(p)$('server-cad').value=p.path;}
function chooseAssembly(){const a=assemblies.find(a=>a.key===$('assembly-select').value);$('assembly-part').replaceChildren(...(a?.parts||[]).map(p=>new Option(p.name,p.key)));chooseAssemblyPart();window.AssemblyIdentification?.sync();}
$('assembly-select').onchange=chooseAssembly;$('assembly-part').onchange=chooseAssemblyPart;
chooseAssembly();
const ids={pose_backend:'pose-backend',device:'device-select'};
for(const [key,value] of Object.entries(data.defaults||{})){const el=$(ids[key]||key);if(el){if(el.type==='checkbox')el.checked=value;else el.value=value;}}
if($('pose-backend').value==='foundationpose'&&!data.foundationpose?.available)$('pose-backend').value='pem';
const cam=data.camera_defaults||{};if(cam.width)$('camera-resolution').value=cam.width+'x'+cam.height;if(cam.fps)$('camera-fps').value=cam.fps;
if(data.tracking_defaults?.identify_assembly)$('camera-cad-source').value='assembly';
if(data.tracking_defaults)$('sam-refresh').value=data.tracking_defaults.redetect_interval_s;$('foundationpose-state').textContent=data.foundationpose?.available?'FoundationPose ready · CUDA':(data.foundationpose?.reason||'FoundationPose unavailable');$('foundationpose-state').classList.toggle('error',!data.foundationpose?.available);$('pose-backend').querySelector('option[value=foundationpose]').disabled=!data.foundationpose?.available;serverRoot=data.file_root||serverRoot;document.querySelectorAll('[data-location]').forEach(button=>{if(button.dataset.location)button.hidden=true;else button.textContent='Root directory';});$('segmentor').querySelector('option[value=fastsam]').disabled=!env.fastsam_weights;$('pose').disabled=!env.pose_available||!env.cuda_available;$('device').textContent=env.cuda_available?env.gpu+' · CUDA available':'CPU · CUDA unavailable';if(!env.cuda_available)$('device-select').value='cpu';for(const option of $('backbone').options){option.disabled=!env.sam_weights[option.value];option.textContent=option.value.toUpperCase()+(env.sam_weights[option.value]?' · available':' · weights missing');}$('example-frame').innerHTML=data.examples.map(f=>`<option value="${escape(f.name)}">${escape(f.label)}</option>`).join('');$('example-frame').value=data.example_defaults?.frame||'';$('example-cad').innerHTML=data.catalog.map(p=>`<option value="${escape(p.key)}">${escape(p.key)} · ${escape(p.name)}</option>`).join('');$('example-cad').value=data.example_defaults?.cad||'';updateBackend();if(!env.dino_weights||!env.sam_weights.vit_h)message('Weights are missing. Read the installation guide before running.',true);}).catch(error=>message('Failed to query the server: '+error.message,true));

let compactWindow=null,compactTimer=null;
$('open-viewer').onclick=()=>{
  compactWindow=window.open('/viewer','vision-viewer','width=800,height=650');
  window.visionCompact=!!compactWindow&&!compactWindow.closed;window.VisionCamera?.busyChanged();
  clearInterval(compactTimer);if(window.visionCompact)compactTimer=setInterval(()=>{if(compactWindow.closed){window.visionCompact=false;clearInterval(compactTimer);window.VisionCamera?.busyChanged();}},500);
};
async function startupStatus(){
  try{const r=await fetch('/api/viewer/status');const data=await r.json();if(!r.ok)return;
    $('startup-state').textContent=data.startup.message;$('startup-state').classList.toggle('error',data.startup.state==='error');
    if(data.startup.state==='loading')setTimeout(startupStatus,500);
  }catch(error){$('startup-state').textContent='Failed to query preload status.';}
}
startupStatus();
function updateSource(){
  const source=$('file-source').value,live=source==='realsense';
  if(!live&&liveDetection)stopLiveDetection('Detection stopped after changing the image source.');
  const cadSource=live?$('camera-cad-source').value:source;
  const identify=live&&cadSource==='assembly';
  $('assembly-identification-note').hidden=!identify;
  window.AssemblyIdentification?.sync();
  document.querySelectorAll('.source-server').forEach(el=>el.hidden=!(source==='server'||live&&['server','assembly'].includes(cadSource)&&el.querySelector('#server-cad')));
  $('assembly-part').disabled=identify;
  $('server-cad').disabled=identify;
  document.querySelector('.browse-server[data-kind="cad"]').disabled=identify;
  if(identify){$('mode').value='rgbd';$('cad_units').value='mm';$('pose-initialization').value='automatic';}
  $('cad_units').disabled=identify;
  $('pose-initialization').disabled=identify;
  if(identify)$('mode').disabled=true;
  else $('mode').disabled=$('pose-backend').value==='foundationpose';
  document.querySelectorAll('.source-upload').forEach(el=>el.hidden=!(source==='upload'||live&&cadSource==='upload'&&el.querySelector('#cad')));
  $('depth-files').hidden=live||$('mode').value!=='rgbd';
  $('live-camera').hidden=!live;
  $('source-note').textContent=live?'Select the CAD, open the camera and click Start live detection, or analyse one current frame.':source==='server'?'Browse server files.':'Upload files from the computer running this browser.';
  $('run').textContent=live?'Analizar fotograma actual':'Detect object';
  window.VisionCamera?.sourceChanged(live);
}
const legacyModeChanged=$('mode').onchange;
$('mode').onchange=()=>{
  if($('pose-backend').value==='foundationpose')$('mode').value='rgbd';
  legacyModeChanged();updateSource();
};
function updateBackend(){
  const fp=$('pose-backend').value==='foundationpose',env=window.visionEnvironment;
  if(fp){$('mode').value='rgbd';$('device-select').value='cuda';$('pose').checked=true;}
  $('mode').disabled=fp;
  $('device-select').disabled=fp;
  $('pose').disabled=fp||!!env&&(!env.environment.pose_available||!env.environment.cuda_available);
  $('backend-note').textContent=fp?'FoundationPose automatically receives the SAM/FastSAM mask; no manual box is needed. Use RGB-D and CUDA. The result 3D box helps inspect the pose.':'SAM-6D segments the CAD; optional PEM pose requires RGB-D and CUDA.';
  $('mode').onchange();
  window.VisionCamera?.selectionChanged();
  window.VisionManual?.sync();
}
$('pose-backend').onchange=updateBackend;
$('file-source').onchange=updateSource;
$('camera-cad-source').onchange=()=>{updateSource();window.VisionManual?.sync();};
async function buildRequest(recordingId=null){
  const request={options:config()};
  if($('file-source').value==='realsense'){
    request.live_camera=true;
    if(recordingId)request.recording_id=recordingId;
    if($('camera-cad-source').value==='assembly'){
      if(recordingId)throw new Error('Select a fixed CAD to analyse a recording.');
      if(!$('assembly-select').value)throw new Error('Select an assembly.');
      request.auto_assembly=$('assembly-select').value;
    }else if($('camera-cad-source').value==='server'){
      const path=$('server-cad').value.trim();if(!path)throw new Error('No server CAD selected.');
      request.server_files={cad:path};
    }else request.cad=await file('cad');
  }else{
    const kinds=request.options.mode==='rgbd'?['rgb','cad','depth','camera']:['rgb','cad'];
    if($('file-source').value==='server'){
      request.server_files={};for(const kind of kinds){const path=$('server-'+kind).value.trim();if(!path)throw new Error('No server file selected: '+kind);request.server_files[kind]=path;}
    }else for(const kind of kinds)request[kind]=await file(kind);
  }
  window.VisionManual?.apply(request);
  return request;
}
$('form').onsubmit=async event=>{event.preventDefault();if(running||liveDetection)return;try{await submit(await buildRequest());}catch(error){message(error.message,true);}};
window.VisionApp={
  buildAlignmentRequest:async()=>{
    if(running||liveDetection)throw new Error('Stop continuous detection and wait for analyses to finish.');
    if($('file-source').value!=='realsense')throw new Error('Select live Intel RealSense.');
    if(window.VisionManual?.active())throw new Error('Select automatic SAM initialization.');
    const request=await buildRequest();
    request.options={...request.options,mode:'rgbd',device:'cuda',pose:true,pose_backend:'foundationpose',segmentor:'sam'};
    return request;
  },
  analyzeRecording:async id=>{if(running||liveDetection)return;try{if($('pose-backend').value!=='foundationpose')throw new Error('Select FoundationPose to track video.');await submit(await buildRequest(id));}catch(error){message(error.message,true);}},
  initializeFastTracking:async()=>{if(running||liveDetection)return null;if($('file-source').value!=='realsense'||$('pose-backend').value!=='foundationpose')throw new Error('Select RealSense and FoundationPose.');return await submit(await buildRequest())?lastJobId:null;},
  estimateManual:async()=>{if(running||liveDetection)return;try{await submit(await buildRequest());}catch(error){message(error.message,true);}},
  message,updateSource,startLiveDetection,stopLiveDetection,isBusy:()=>running,isLiveDetection:()=>liveDetection
};
updateSource();
})();
