/* Robot controls are an independent extension of the vision interface. */
(()=>{'use strict';
const section=document.createElement('section');section.className='panel';section.id='cad-robot-alignment';
section.innerHTML=`<h2>Robot alignment</h2>
<p>Detect the selected CAD or identify an assembly part, review the target, then execute alignment.</p>
<div class="camera-actions"><label>Alignment target <select id="cad-align-target-mode" style="max-width:100%"><option value="face">CAD face standoff</option></select></label></div>
<p id="cad-align-target-help" class="note">The TCP target is <strong id="cad-align-offset">20 cm</strong> from the CAD +Y face center (bounding box unless a CAD-specific point is configured).</p>
<p id="cad-align-orientation" class="note"></p>
<p id="cad-align-rotation-policy" class="note"></p>
<p id="cad-align-fixed" class="note"></p>
<p id="cad-align-config" class="note"></p>
<p id="cad-align-calibration" class="note"></p>
<p id="cad-align-camera" class="note"></p>
<div class="camera-actions"><label>Alignment speed (mm/s) <input id="cad-align-speed" type="number" min="0.001" max="250" step="any" style="width:130px"></label>
<button type="button" id="cad-align-speed-apply" class="secondary">Apply speed</button></div>
<p class="note">Applying speed invalidates the current target. Acceleration and controller safety limits remain unchanged.</p>
<div class="camera-actions">
<button type="button" id="cad-align-connect" class="secondary">Connect robot</button>
<button type="button" id="cad-align-disconnect" class="secondary">Disconnect robot</button>
<button type="button" id="cad-align-plan" class="primary">Detect pose</button>
<button type="button" id="cad-align-execute" class="primary" disabled>Execute alignment</button>
<button type="button" id="cad-align-skill" class="primary" disabled hidden>Execute skill</button>
<button type="button" id="cad-align-stop" class="secondary" style="color:#a00;border-color:#a00">Stop robot</button>
</div>
<p id="cad-align-message" role="status" aria-live="polite">Querying robot…</p>
<p id="cad-align-blocked" class="note"></p>
<p id="cad-align-skill-info" class="note" hidden></p>
<div id="cad-align-preview" hidden><img id="cad-align-image" alt="Detection and axes used to calculate the target" style="max-width:100%"><p id="cad-align-target"></p><div id="cad-align-projections" style="display:flex;flex-wrap:wrap;gap:12px"></div><details><summary>Target matrices and calibration</summary><pre id="cad-align-matrices" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></div>
<p class="note">TCP, IK and controller limits are checked automatically. Collision avoidance is not computed. Keep the physical robot stop accessible.</p>`;
(document.querySelector('main')||document.body).append(section);
const $=id=>document.getElementById('cad-align-'+id);
let state=null,pending=false,polling=false,shown=null,lastError='',speedDirty=false,lastSelection=null,demoList=null;
const FACE_HELP=document.getElementById('cad-align-target-help').innerHTML;
function targetRequest(){const v=$('target-mode').value;if(v==='face')return{mode:'cad_face'};return{mode:'skill_demo',demo_id:v.slice(5)||null};}
const sameTarget=(a,b)=>JSON.stringify({mode:a?.mode,demo_id:a?.demo_id??null})===JSON.stringify({mode:b?.mode,demo_id:b?.demo_id??null});
function renderDemos(cfg){const demos=cfg.skill_demos||[],key=JSON.stringify(demos.map(d=>[d.demo_id,d.available,d.cad_key,d.problem]))+(cfg.skill_demos_error||'');if(key===demoList)return;demoList=key;const select=$('target-mode'),value=select.value;const options=[new Option('CAD face standoff','face')];if(demos.length){options.push(new Option('Skill demo of the detected CAD','demo:'));for(const d of demos){const skill=(d.skills||[]).map(s=>s.part_id+' → '+s.skill_id).join(', ');const o=new Option('Skill demo: '+d.demo_id+' · '+(d.cad_key||'no pose')+(skill?' · '+skill:'')+(d.available?'':' (unavailable: '+(d.problem||'no pose')+')'),'demo:'+d.demo_id);o.disabled=!d.available;options.push(o);}}select.replaceChildren(...options);select.value=[...select.options].some(o=>o.value===value&&!o.disabled)?value:'face';}
const currentSelection=()=>window.VisionApp?.alignmentSelectionSignature?.()||null;
function selectionChanged(){const signature=currentSelection();if(lastSelection!==null&&signature!==lastSelection){lastSelection=signature;if(state&&(state.plan||state.skill_ready||['estimating','checking'].includes(state.state)))act('invalidate');}else lastSelection=signature;if(state)render();}
document.addEventListener('change',event=>{if(['file-source','camera-cad-source','server-cad','cad','cad_units','assembly-select','assembly-part'].includes(event.target.id))selectionChanged();});
document.addEventListener('input',event=>{if(event.target.id==='server-cad')selectionChanged();});
async function api(action,data){
 const r=await fetch('/api/alignment/'+action,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 const value=await r.json();if(!r.ok)throw new Error(value.error||'Alignment error');return value;
}
function render(){
 if(!state)return;
 const cfg=state.config||{},busy=state.busy||pending,plan=state.plan,hand=cfg.hand_eye||{},camera=state.active_camera,trial=cfg.execution_mode==='supervised_trial';
 renderDemos(cfg);$('target-mode').disabled=busy;
 const demoMode=$('target-mode').value!=='face';
 if(demoMode)$('target-help').textContent='The TCP target reproduces the TCP pose relative to the CAD recorded in the skill demo: T_base_tcp = T_base_cad(now) · (T_tcp_cam · T_cam_cad(demo))⁻¹. Replay the skill afterwards from the aligned pose (DINOBOT ask_start_pose=False).'+(cfg.skill_demos_error?' Skill demos: '+cfg.skill_demos_error:'');
 else{if($('target-help').innerHTML!==FACE_HELP)$('target-help').innerHTML=FACE_HELP;$('offset').textContent=(100*(cfg.offset_m||.2)).toFixed(0)+' cm';}
 $('orientation').hidden=demoMode;$('orientation').textContent=trial&&cfg.orientation_mode==='plane_frame'?'Orientation: +TCP Y = +CAD Z; +TCP Z = −CAD Y (toward the CAD front face); +TCP X = +CAD X.':cfg.orientation_mode==='cad_axes_inward'?'Orientation: −TCP X = +CAD Z; +TCP Y = +CAD X; +TCP Z = −CAD Y (toward the CAD front face).':cfg.orientation_mode==='cad_axes'?'Orientation: −TCP X = +CAD Z; −TCP Y = +CAD X; +TCP Z = +CAD Y (outward).':cfg.orientation_mode==='minimum_z'?'Orientation: tool Z points toward the CAD front face using the minimum rotation.':'Orientation: configured plane frame.';
 if(!speedDirty&&document.activeElement!==$('speed')&&Number.isFinite(cfg.speed_m_s))$('speed').value=Number((cfg.speed_m_s*1000).toFixed(6));
 $('speed').max=(cfg.max_speed_m_s||.25)*1000;
 $('speed').disabled=busy;
 $('speed-apply').disabled=busy||!speedDirty||!$('speed').value||!$('speed').checkValidity()||!cfg.settings_sha256||state.needs_reconnect;
 const selection=JSON.parse(window.VisionApp?.alignmentSelectionSignature?.()||'{}');
 $('fixed').textContent=selection.source==='assembly'?'Alignment source: SAM6D automatic identification · '+(selection.assembly||'select an assembly'):'Alignment source: selected CAD · '+(selection.cad||selection.file?.name||'select a CAD in the camera controls');
 const aligned=state.state==='completed'&&state.connected&&!!state.result;
 $('plan').textContent=aligned?'Align again (detect pose)':selection.source==='assembly'?'Identify CAD and detect pose':'Detect pose';
 $('config').textContent='Robot: '+(cfg.robot_ip||'not configured');
 $('calibration').textContent='Hand-eye calibration: '+(hand.status||'unknown')+(hand.robot_model?' · '+hand.robot_model:'')+(hand.source_xml?' · '+hand.source_xml.split('/').pop():'')+(cfg.preview_only?' · Preview only':'');
 $('camera').textContent=camera?'Active RealSense: '+camera.serial:'Open the RealSense camera using the camera controls.';
 $('message').textContent=lastError||state.message;$('message').classList.toggle('error',!!lastError||state.state==='error');
 const blocked=[];
 if(!state.connected)blocked.push('Connect the robot.');
 if(!trial&&!cfg.calibration_confirmed)blocked.push('Calibration is not approved for motion.');
 if(cfg.preview_only)blocked.push('Preview-only mode: motion is blocked.');else if(!cfg.motion_enabled)blocked.push('Motion is disabled.');
 if(!cfg.camera_serial||hand.preview_ready===false)blocked.push('matrix_calibration.yaml must declare the calibrated camera serial and calibration TCP.');
 if(!trial&&hand.status==='provisional')blocked.push('Provisional calibration: preview only.');
 if(trial&&plan&&!plan.controller_path_checked)blocked.push('Controller checks have not passed for this target.');
 if(state.needs_reconnect)blocked.push('Settings changed: disconnect and reconnect the robot.');
 if(state.settings_error)blocked.push(state.settings_error);
 if(speedDirty)blocked.push('Apply the selected speed before detecting or executing.');
 if(plan&&plan.selection_signature!==undefined&&plan.selection_signature!==null&&plan.selection_signature!==currentSelection())blocked.push('CAD selection changed. Detect a new target.');
 if(plan&&!sameTarget(plan.alignment_target,targetRequest()))blocked.push('Alignment target changed. Detect a new target.');
 const expired=plan&&(Date.now()/1000-plan.planned_at>plan.expires_in_s);
 const rotationBlocked=!!plan?.rotation_limit_exceeded;
 $('rotation-policy').textContent='Rotation allowance: '+(cfg.max_rotation_deg??180)+'° by the shortest angular path, with no downward-Z condition. Controller IK and joint limits remain active.';
 if(rotationBlocked)blocked.push('Required rotation: '+plan.rotation_deg.toFixed(1)+'°; allowed: '+plan.rotation_limit_deg+'°. Change motion.max_rotation_deg and detect a new target.');
 if(plan?.controller_check_error)blocked.push('Controller check: '+plan.controller_check_error);
 if(expired)blocked.push('The target expired. Detect the pose again.');
 $('blocked').textContent=blocked.join(' ');
 $('connect').disabled=busy||state.connected;
 $('disconnect').disabled=busy||!state.connected;
 $('plan').disabled=busy||speedDirty||!state.connected||state.needs_reconnect||hand.preview_ready===false;
 $('execute').disabled=busy||!plan||expired||blocked.length>0;
 $('stop').disabled=!state.connected&&!state.busy;
 const skill=state.skill_ready,running=state.state==='skill';
 $('skill').hidden=!(demoMode||skill||running);
 $('skill').disabled=busy||!skill||!state.connected||!cfg.skill_execution_available;
 $('skill-info').hidden=!(skill||running||demoMode);
 $('skill-info').textContent=running?'Skill running in DINOBOT. Stop robot sends stopJ.':skill?'Aligned to '+skill.demo_id+' ('+(skill.skills||[]).map(s=>s.skill_id).join(', ')+'). Choose Align again (detect pose) to refine the alignment, or Execute skill to replay from this pose at '+(cfg.skill_speed_factor||1)+'×.':!cfg.skill_execution_available?'Skill execution needs alignment.skill_python (DINOBOT interpreter) in settings.json.':'Execute skill becomes available after a completed skill-demo alignment.';
 $('preview').hidden=!plan;
 if(plan&&shown!==(plan.id+'|'+plan.controller_path_checked)){
  shown=plan.id+'|'+plan.controller_path_checked;$('image').src=plan.image_url;
  const demo=plan.skill_demo;
  $('target').textContent='CAD: '+(plan.cad_name||plan.cad_key)+' ['+plan.cad_key+'] · '+(plan.selection_mode||'selected')+' · '+(demo?'Skill demo '+demo.demo_id+' ('+(demo.skills||[]).map(s=>s.skill_id).join(', ')+') · object moved '+(plan.object_displacement_from_demo_m*100).toFixed(1)+' cm / '+plan.object_rotation_from_demo_deg.toFixed(1)+'° since the demo':(plan.plane_origin_source||'CAD plane'))+' · Target TCP in base coordinates [m, rad]: '+plan.tcp_target_m_rad.map(v=>v.toFixed(5)).join(', ')+' · travel '+(plan.translation_m*100).toFixed(1)+' cm · rotation '+plan.rotation_deg.toFixed(1)+'° · speed '+((plan.speed_m_s||cfg.speed_m_s)*1000).toFixed(1)+' mm/s';
  drawProjections(plan);$('matrices').textContent=JSON.stringify(plan,null,2);
 }
}
function drawProjections(plan){const container=$('projections');container.replaceChildren();const frames=[['Current TCP',plan.T_base_from_tcp_capture,'#586b7c'],plan.T_base_from_plane?['CAD plane',plan.T_base_from_plane,'#197bbb']:['CAD',plan.T_base_from_cad,'#197bbb'],['Target TCP',plan.T_base_from_tcp_target,'#008264']];for(const [a,b] of [[0,1],[0,2]]){const points=frames.map(([,T])=>T.slice(0,3).map(row=>row[3]));const lo=[a,b].map(i=>Math.min(...points.map(p=>p[i]))-.05),hi=[a,b].map(i=>Math.max(...points.map(p=>p[i]))+.05),scale=220/Math.max(hi[0]-lo[0],hi[1]-lo[1]);const xy=p=>[25+(p[a]-lo[0])*scale,260-(p[b]-lo[1])*scale];const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 330 290');svg.style.cssText='width:320px;max-width:100%;border:1px solid #dce4e8';const label=document.createElementNS(ns,'text');label.setAttribute('x',12);label.setAttribute('y',18);label.setAttribute('font-size',12);label.textContent='Base '+('XYZ'[a])+('XYZ'[b])+' · positions in meters';svg.append(label);frames.forEach(([name,T,color],index)=>{const [x,y]=xy(points[index]);const circle=document.createElementNS(ns,'circle');circle.setAttribute('cx',x);circle.setAttribute('cy',y);circle.setAttribute('r',5);circle.setAttribute('fill',color);const text=document.createElementNS(ns,'text');text.setAttribute('x',x+7);text.setAttribute('y',y-6);text.setAttribute('font-size',10);text.textContent=name;svg.append(circle,text);});const start=xy(points[1]),end=xy(points[2]);const line=document.createElementNS(ns,'line');line.setAttribute('x1',start[0]);line.setAttribute('y1',start[1]);line.setAttribute('x2',end[0]);line.setAttribute('y2',end[1]);line.setAttribute('stroke','#008264');line.setAttribute('stroke-dasharray','4 3');svg.append(line);container.append(svg);}}
async function refresh(){if(polling)return;polling=true;try{state=await api('status');selectionChanged();render();}catch(e){$('message').textContent=e.message;$('execute').disabled=true;$('plan').disabled=true;}finally{polling=false;}}
async function act(name,data={}){if(pending&&name!=='stop')return;pending=true;lastError='';render();try{state=await api(name,data);render();}catch(e){lastError=e.message;}finally{pending=false;render();}}
$('connect').onclick=()=>act('connect');$('disconnect').onclick=()=>act('disconnect');
$('speed').oninput=()=>{speedDirty=true;render();};
$('speed-apply').onclick=async()=>{const speed=Number($('speed').value)/1000;await act('speed',{speed_m_s:speed,settings_sha256:state?.config?.settings_sha256});if(!lastError&&state?.config?.speed_m_s===speed)speedDirty=false;render();};
$('stop').onclick=()=>act('stop');
$('target-mode').onchange=()=>{if(state&&(state.plan||state.skill_ready||['estimating','checking'].includes(state.state)))act('invalidate');else render();};
$('plan').onclick=async()=>{try{const request=await window.VisionApp.buildAlignmentRequest();request.alignment_target=targetRequest();lastSelection=currentSelection();await act('plan',request);}catch(e){$('message').textContent=e.message;$('message').classList.add('error');}};
$('execute').onclick=()=>{if(!state?.plan)return;if(!window.confirm('Move the physical robot to the displayed alignment target for '+(state.plan.cad_name||state.plan.cad_key)+(state.plan.skill_demo?' (skill demo '+state.plan.skill_demo.demo_id+')':'')+' now?'+(state.config.execution_mode==='supervised_trial'?' Hand-eye calibration remains provisional.':'')))return;act('execute',{plan_id:state.plan.id,confirmed:true,selection_signature:currentSelection(),alignment_target:targetRequest()});};
$('skill').onclick=()=>{const skill=state?.skill_ready;if(!skill)return;if(!window.confirm('Execute skill '+(skill.skills||[]).map(s=>s.skill_id).join(', ')+' (demo '+skill.demo_id+') on the physical robot now?\n\nCAD Tracking releases the RTDE connection to DINOBOT, which activates and OPENS the gripper before replaying the recorded trajectory from the aligned pose. Keep the emergency stop accessible.'))return;act('execute-skill',{plan_id:skill.plan_id,confirmed:true});};
refresh();setInterval(refresh,1000);
})();
