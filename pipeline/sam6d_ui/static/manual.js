/* Freeze synchronized RGB-D before marking a box. Never apply coordinates to a later frame. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
let captureId=null,box=null,image=null,start=null,loading=false;
const active=()=>$('pose-backend').value==='foundationpose'&&$('pose-initialization').value==='manual';
function sync(){
  if(active()&&window.VisionApp.isLiveDetection())window.VisionApp.stopLiveDetection('Manual mode uses a frozen frame; use fast tracking for live updates.');
  window.VisionCamera?.busyChanged();
  if($('pose-backend').value==='foundationpose')$('backend-note').textContent=active()?'FoundationPose only: uses your box as an approximate rectangular mask. SAM, DINOv2 and CAD comparison are bypassed.':'Automatic initialization: SAM6D obtains the initial FoundationPose mask.';
  $('fast-start').textContent=active()?'Start tracking from my box':'Locate and start fast tracking';
  $('pose-init-choice').hidden=$('pose-backend').value!=='foundationpose';
  $('manual-panel').hidden=!active();
  $('manual-estimate').disabled=loading||!captureId||!box||window.VisionApp.isBusy()||window.VisionApp.isLiveDetection();
}
function draw(){
 const c=$('manual-canvas'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
 if(image)ctx.drawImage(image,0,0);
 if(box){ctx.strokeStyle='#23eab0';ctx.lineWidth=Math.max(2,c.width/300);ctx.strokeRect(...box);}
}
function position(event){const c=$('manual-canvas'),r=c.getBoundingClientRect();return [Math.round(Math.max(0,Math.min(c.width,(event.clientX-r.left)*c.width/r.width))),Math.round(Math.max(0,Math.min(c.height,(event.clientY-r.top)*c.height/r.height)))];}
$('manual-freeze').onclick=async()=>{
 if(loading)return;
 loading=true;captureId=null;box=null;sync();$('manual-status').textContent='Capturing RGB, depth and calibration…';
 try{
  if($('file-source').value!=='realsense')throw new Error('Select live Intel RealSense and open the camera.');
  const response=await fetch('/api/camera/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  const result=await response.json();if(!response.ok)throw new Error(result.error);
  const next=new Image();next.src='/camera-files/captures/'+result.id+'/rgb.png';await next.decode();
  image=next;captureId=result.id;const c=$('manual-canvas');c.width=next.naturalWidth;c.height=next.naturalHeight;c.hidden=false;draw();
  $('manual-status').textContent='Frozen frame. Drag a tight box around the object.';
 }catch(error){$('manual-status').textContent=error.message;}finally{loading=false;sync();}
};
$('manual-canvas').onpointerdown=event=>{if(!image)return;start=position(event);box=null;event.currentTarget.setPointerCapture(event.pointerId);draw();};
$('manual-canvas').onpointermove=event=>{if(!start)return;const p=position(event);box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(start[0]-p[0]),Math.abs(start[1]-p[1])];draw();};
$('manual-canvas').onpointerup=event=>{
 if(!start)return;const p=position(event);box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(start[0]-p[0]),Math.abs(start[1]-p[1])];start=null;
 if(box[2]<4||box[3]<4)box=null;
 draw();$('manual-status').textContent=box?'Box [x, y, width, height]: '+box.join(', ')+'. This frozen RGB-D frame will be used; avoid background and other parts.':'The box must be at least 4×4 pixels.';sync();
};
$('manual-canvas').onpointercancel=()=>{start=null;box=null;draw();sync();};
$('pose-initialization').onchange=sync;
$('manual-estimate').onclick=()=>window.VisionApp.estimateManual();
window.VisionManual={active,sync,apply:request=>{
 if(!active())return;
 if(!request.live_camera)throw new Error('The manual box uses a RealSense capture.');
 if(request.recording_id)throw new Error('Select automatic initialization to track a recording.');
 if(!captureId||!box)throw new Error('Capture a frame and draw the object box first.');
 request.capture_id=captureId;request.manual_bbox=box.slice();
}};
if(new URLSearchParams(location.search).get('manual')==='1'){$('pose-backend').value='foundationpose';$('pose-initialization').value='manual';$('pose-backend').dispatchEvent(new Event('change'));}
sync();
})();
