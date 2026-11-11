// Explicit QA of the simulated HTTP fixture; never uses real hardware.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {writeFileSync} from 'node:fs';
const url=process.argv[2],out=process.argv[3],debug=process.argv[4]==='automatic'?'9256':process.argv[4]==='trial'?'9255':'9254';
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-port='+debug,'--user-data-dir='+out+'/chrome','about:blank'],{stdio:'ignore'});
process.on('exit',()=>chrome.kill());
const endpoint='http://127.0.0.1:'+debug;
for(let i=0;i<100;i++){try{if((await fetch(endpoint+'/json/version')).ok)break;}catch{}await new Promise(r=>setTimeout(r,100));}
const target=await(await fetch(endpoint+'/json/new?about:blank',{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j;});
let seq=0;const pending=new Map(),errors=[];
socket.onmessage=({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);};
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const wait=async expression=>{for(let i=0;i<120;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw new Error('Timeout '+expression);};
await send('Runtime.enable');await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1100,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url});
await wait(`document.querySelector('#cad-align-camera')?.textContent.includes('123')`);
assert(await evaluate(`document.querySelector('#cad-align-calibration').textContent.includes('provisional')`));
if(process.argv[4]!=='preview'){
 await wait(`window.visionEnvironment && document.querySelector('#assembly-select').options.length>0`);
 await evaluate(`document.querySelector('#file-source').value='realsense';document.querySelector('#file-source').onchange();document.querySelector('#camera-cad-source').value='server';document.querySelector('#camera-cad-source').onchange();document.querySelector('#server-cad').value='part.stl';document.querySelector('#server-cad').dispatchEvent(new Event('change',{bubbles:true}));`);
 if(process.argv[4]==='automatic')await evaluate(`document.querySelector('#camera-cad-source').value='assembly';document.querySelector('#camera-cad-source').onchange();document.querySelector('#camera-cad-source').dispatchEvent(new Event('change',{bubbles:true}));`);
 const request=await evaluate(`window.VisionApp.buildAlignmentRequest()`);
 if(process.argv[4]==='automatic')assert.equal(request.auto_assembly,'printed_assembly');
 else assert.equal(request.server_files.cad,'part.stl');
 assert.equal(request.target_cad_key,undefined);assert.equal(request.cad,undefined);
 assert.equal(typeof request.selection_signature,'string');
 if(process.argv[4]==='rotation')assert(await evaluate(`document.querySelector('#cad-align-orientation').textContent.includes('−TCP X = +CAD Z')`));
 else assert(await evaluate(`document.querySelector('#cad-align-orientation').textContent.includes('+TCP Y = +CAD Z')`));
 if(process.argv[4]!=='rotation')assert(await evaluate(`document.querySelector('#cad-align-orientation').textContent.includes('+TCP Z = −CAD Y')`));
 await evaluate(`document.querySelector('#cad-align-speed').value='15';document.querySelector('#cad-align-speed').dispatchEvent(new Event('input'));`);
 assert(await evaluate(`document.querySelector('#cad-align-plan').disabled`));
 await evaluate(`document.querySelector('#cad-align-speed-apply').click()`);
 await wait(`document.querySelector('#cad-align-message').textContent.includes('15 mm/s')&&!document.querySelector('#cad-align-plan').disabled`);
 await evaluate(`document.querySelector('#cad-align-plan').click()`);
 await wait(`!document.querySelector('#cad-align-preview').hidden`);
 if(process.argv[4]==='rotation'){
  assert(await evaluate(`document.querySelector('#cad-align-execute').disabled`));
  assert.equal(await evaluate(`document.querySelector('#cad-align-allow-rotation')`),null);
 }else await wait(`!document.querySelector('#cad-align-execute').disabled`);
 assert.equal(await evaluate(`document.querySelector('#cad-align-review')`),null);
 assert.equal(await evaluate(`document.querySelector('#cad-align-check')`),null);
 // Reject the final confirmation and verify that no execution request is sent.
 await evaluate(`window.confirm=()=>false;document.querySelector('#cad-align-execute').click()`);
 const state=await evaluate(`fetch('/api/alignment/status').then(r=>r.json())`);
 assert.equal(state.state,'planned');assert.equal(state.plan.hand_eye.status,'provisional');
 assert.equal(state.plan.collision_checked,false);assert.equal(state.plan.accuracy_validated,false);
 assert(state.plan.controller_path_checked);
 assert.equal(state.plan.speed_m_s,.015);
 assert.equal(state.plan.cad_key,process.argv[4]==='automatic'?'03_option_cassette':'02_control_panel');
 assert.equal(state.plan.selection_mode,process.argv[4]==='automatic'?'automatic':'selected');
 if(process.argv[4]==='trial'){
  await evaluate(`document.querySelector('#server-cad').value='cassette.stl';document.querySelector('#server-cad').dispatchEvent(new Event('change',{bubbles:true}));`);
  assert(await evaluate(`document.querySelector('#cad-align-execute').disabled`));
  await wait(`document.querySelector('#cad-align-preview').hidden&&!document.querySelector('#cad-align-plan').disabled`);
  await evaluate(`document.querySelector('#cad-align-plan').click()`);
  await wait(`document.querySelector('#cad-align-target').textContent.includes('03_option_cassette')&&!document.querySelector('#cad-align-execute').disabled`);
 }

 await evaluate(`document.querySelector('#cad-align-speed').value='20';document.querySelector('#cad-align-speed').dispatchEvent(new Event('input'));`);
 assert(await evaluate(`document.querySelector('#cad-align-execute').disabled`));
 await evaluate(`document.querySelector('#cad-align-speed-apply').click()`);
 await wait(`document.querySelector('#cad-align-preview').hidden&&!document.querySelector('#cad-align-plan').disabled`);
 assert(await evaluate(`document.querySelector('#cad-align-execute').disabled`));
}else{
await evaluate(`document.querySelector('#cad-align-connect').click()`);
await wait(`!document.querySelector('#cad-align-plan').disabled`);
await evaluate(`window.VisionApp.buildAlignmentRequest=async()=>({live_camera:true,server_files:{cad:'part.stl'}});document.querySelector('#cad-align-plan').click()`);
await wait(`!document.querySelector('#cad-align-preview').hidden`);
const state=await evaluate(`fetch('/api/alignment/status').then(r=>r.json())`);
assert(state.plan.preview_only);assert.equal(state.plan.controller_path_checked,false);
assert.equal(state.plan.hand_eye.status,'provisional');
assert.equal(await evaluate(`document.querySelectorAll('#cad-align-projections svg').length`),2);
assert(await evaluate(`document.querySelector('#cad-align-execute').disabled`));
const response=await evaluate(`fetch('/api/alignment/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({plan_id:${JSON.stringify(state.plan.id)}})}).then(async r=>({status:r.status,body:await r.json()}))`);
assert.equal(response.status,400);assert(response.body.error.includes('Preview-only'));
}
await evaluate(`document.querySelector('#cad-robot-alignment').scrollIntoView()`);
const desktop=await send('Page.captureScreenshot',{format:'png'});writeFileSync(out+'/synthetic_hand_eye_desktop.png',Buffer.from(desktop.data,'base64'));
await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
assert(await evaluate('document.body.scrollWidth<=window.innerWidth+1'));
assert.deepEqual(errors,[]);
writeFileSync(out+'/browser_proof.json',JSON.stringify({status:'passed',synthetic_hardware:true,mode:process.argv[4],operator_confirmation:true,shared_yaml:true,preview_20cm:true,execution_not_requested:true,mobile_overflow:false,javascript_errors:errors},null,2));
console.log('Browser hand-eye checks passed; no execution requested.');
await send('Page.close');socket.close();chrome.kill();
