/* Robot controls are an independent extension of the existing vision interface. */
(()=>{'use strict';
const section=document.createElement('section');section.className='panel';section.id='cad-robot-alignment';
section.innerHTML=`<h2>Align robot to CAD</h2>
<p>RGB-D capture → SAM6D + FoundationPose → TCP target at <strong id="cad-align-offset">20 cm</strong>, perpendicular to the CAD plane. The tool Z axis points toward the plane.</p>
<p id="cad-align-config" class="note"></p>
<div class="camera-actions">
<button type="button" id="cad-align-connect" class="secondary">Connect robot</button>
<button type="button" id="cad-align-disconnect" class="secondary">Disconnect robot</button>
<button type="button" id="cad-align-plan" class="primary">Capture and calculate target</button>
<button type="button" id="cad-align-execute" class="primary" disabled>Execute displayed target</button>
<button type="button" id="cad-align-stop" class="secondary" style="color:#a00;border-color:#a00">Stop robot</button>
</div>
<p id="cad-align-message" role="status" aria-live="polite">Querying robot…</p>
<p id="cad-align-blocked" class="note"></p>
<div id="cad-align-preview" hidden><img id="cad-align-image" alt="Detection and axes used to calculate the target" style="max-width:100%"><p id="cad-align-target"></p><details><summary>Target matrices and capture data</summary><pre id="cad-align-matrices" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></div>
<p class="note">Review the frontal plane and camera–TCP calibration. Motion is linear and does not plan obstacle avoidance. This interface stop is software-based; keep the physical robot stop accessible.</p>`;
(document.querySelector('main')||document.body).append(section);
const $=id=>document.getElementById('cad-align-'+id);
let state=null,pending=false,polling=false,shown=null,lastError='';
async function api(action,data){
 const r=await fetch('/api/alignment/'+action,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 const value=await r.json();if(!r.ok)throw new Error(value.error||'Alignment error');return value;
}
function render(){
 if(!state)return;const cfg=state.config||{},busy=state.busy||pending,plan=state.plan;
 $('offset').textContent=(100*(cfg.offset_m||.2)).toFixed(0)+' cm';
 $('config').textContent='Robot: '+(cfg.robot_ip||'not configured')+' · '+state.settings_file;
 $('message').textContent=lastError||state.message;$('message').classList.toggle('error',!!lastError||state.state==='error');
 const blocked=[];
 if(!cfg.calibration_confirmed)blocked.push('Confirm calibration in settings.json first.');
 if(!cfg.motion_enabled)blocked.push('Motion is disabled in settings.json; you can calculate and review targets.');
 if(!cfg.camera_serial)blocked.push('Configure the calibrated camera serial number first.');
 if(state.settings_error)blocked.push(state.settings_error);
 const expired=plan&&(Date.now()/1000-plan.planned_at>plan.expires_in_s);
 if(expired)blocked.push('The target expired. Capture and calculate again.');
 $('blocked').textContent=blocked.join(' ');
 $('connect').disabled=busy||state.connected;
 $('disconnect').disabled=busy||!state.connected;
 $('plan').disabled=busy||!state.connected;
 $('execute').disabled=busy||!plan||expired||blocked.length>0;
 $('stop').disabled=!state.connected&&!state.busy;
 $('preview').hidden=!plan;
 if(plan&&shown!==plan.id){
  shown=plan.id;$('image').src=plan.image_url;
  $('target').textContent='TCP in base coordinates [m, rad]: '+plan.tcp_target_m_rad.map(v=>v.toFixed(5)).join(', ')+' · distance '+(plan.translation_m*100).toFixed(1)+' cm · rotation '+plan.rotation_deg.toFixed(1)+'°';
  $('matrices').textContent=JSON.stringify({T_base_from_tcp_target:plan.T_base_from_tcp_target,T_base_from_cad:plan.T_base_from_cad,T_cad_from_plane:plan.T_cad_from_plane,T_plane_from_tcp_target:plan.T_plane_from_tcp_target,capture_id:plan.capture_id,cad_key:plan.cad_key,planned_at:plan.planned_at},null,2);
 }
}
async function refresh(){if(polling)return;polling=true;try{state=await api('status');render();}catch(e){$('message').textContent=e.message;$('execute').disabled=true;$('plan').disabled=true;}finally{polling=false;}}
async function act(name,data={}){if(pending&&name!=='stop')return;pending=true;lastError='';render();try{state=await api(name,data);render();}catch(e){lastError=e.message;}finally{pending=false;render();}}
$('connect').onclick=()=>act('connect');$('disconnect').onclick=()=>act('disconnect');
$('stop').onclick=()=>act('stop');
$('plan').onclick=async()=>{try{const request=await window.VisionApp.buildAlignmentRequest();await act('plan',request);}catch(e){$('message').textContent=e.message;$('message').classList.add('error');}};
$('execute').onclick=()=>{if(state?.plan)act('execute',{plan_id:state.plan.id});};
refresh();setInterval(refresh,1000);
})();
