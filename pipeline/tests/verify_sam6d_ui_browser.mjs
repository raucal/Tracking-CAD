// Explicit browser integration test against the running local GPU service.
import {readFileSync,writeFileSync} from 'node:fs';
import {spawn} from 'node:child_process';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
const root='/home/raca/projects', endpoint='http://127.0.0.1:9240';
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-dev-shm-usage','--remote-debugging-port=9240',`--user-data-dir=/tmp/sam6d-ui-qa-${process.pid}`,'about:blank'],{stdio:'ignore'});
process.on('exit',()=>chrome.kill());
for(let i=0;i<100;i++){try{if((await fetch(endpoint+'/json/version')).ok)break;}catch{}await new Promise(r=>setTimeout(r,100));}
const target=await(await fetch(endpoint+'/json/new?about:blank',{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j;});
let seq=0;const pending=new Map(),errors=[];
socket.onmessage=({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);};
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const wait=async expression=>{for(let i=0;i<180;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,500));}throw new Error('Timeout: '+expression);};
await send('Runtime.enable');await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1100,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url:'http://127.0.0.1:8770/'});
await wait('document.querySelectorAll("#example-cad option").length===10');
assert(await evaluate('document.querySelector("#depth-files").hidden'));
await evaluate('document.querySelector("#run").click()');
assert(await evaluate('document.querySelector("#progress").textContent.includes("Falta seleccionar")'));
await evaluate('document.querySelector("#mode").value="rgbd";document.querySelector("#mode").dispatchEvent(new Event("change"))');
assert.equal(await evaluate('document.querySelector("#depth-files").hidden'),false);
const manifest=JSON.parse(readFileSync(root+'/SAM6D/pipeline/prepared/manifest.json','utf8'));
const frame=manifest.frames.find(f=>f.name==='control_board');
const paths={rgb:resolve(root+'/SAM6D/pipeline/prepared',frame.source_rgb),cad:root+'/PDDL/Assembly CAD/Assembly parts/05_control_board.stl',depth:resolve(root+'/SAM6D/pipeline/prepared',frame.source_depth),camera:resolve(root+'/SAM6D/pipeline/prepared',frame.source_metadata)};
assert.equal(await evaluate('document.querySelector("#file-source").value'),'server');
for(const [id,path] of Object.entries(paths)){
 await evaluate(`document.querySelector('.browse-server[data-kind="${id}"]').click()`);
 await wait(`Array.from(document.querySelectorAll('#browser-entries button')).some(b=>b.dataset.path===${JSON.stringify(path)})`);
 await evaluate(`Array.from(document.querySelectorAll('#browser-entries button')).find(b=>b.dataset.path===${JSON.stringify(path)}).click()`);
 assert.equal(await evaluate(`document.querySelector('#server-${id}').value`),path);
}
await evaluate('document.querySelector("#run").click()');
await wait('!document.querySelector("#output").hidden');
await wait('document.querySelector("#overlay").naturalWidth>0');
assert.equal(await evaluate('document.querySelector("#actual-device").textContent'),'CUDA');
assert(Number(await evaluate('document.querySelector("#score").textContent'))>0.5);
assert.equal(await evaluate('document.querySelector("#original").naturalWidth'),640);
const saved=await evaluate('fetch(document.querySelector("#downloads a").href.replace("result.json","inputs.json")).then(r=>r.json())');
assert.deepEqual(saved.server_paths,paths);
// The optional upload mode remains available, but is never needed for server inputs.
await evaluate('document.querySelector("#file-source").value="upload";document.querySelector("#file-source").dispatchEvent(new Event("change"))');
assert(await evaluate('!document.querySelector("#rgb").closest("label").hidden'));
await evaluate('document.querySelector("#file-source").value="server";document.querySelector("#file-source").dispatchEvent(new Event("change"))');
// Exercise the user-visible FastSAM + PEM workflow, not only the backend API.
assert.equal(await evaluate('document.querySelector("#pose").disabled'),false);
await evaluate('document.querySelector("#segmentor").value="fastsam";document.querySelector("#segmentor").dispatchEvent(new Event("change"));document.querySelector("#pose").click();document.querySelector("#run-example").click()');
await wait('!document.querySelector("#output").hidden && !document.querySelector("#pose-output").hidden');
await wait('document.querySelector("#pose-overlay").naturalWidth>0');
assert.equal(await evaluate('document.querySelector("#pose-overlay").naturalWidth'),1280);
assert(await evaluate('document.querySelector("#pose-values").textContent.includes("traslacion_mm")'));
const pose=await evaluate(`fetch(document.querySelector('#downloads a[href$="pose.json"]').href).then(r=>r.json())`);
assert.equal(pose.status,'estimated');assert(pose.poses.length>0);
assert(pose.poses[0].t_mm[2]>0);
assert.equal(pose.poses[0].T_camera_from_input_cad_mm.length,4);
const screenshot=await send('Page.captureScreenshot',{format:'png'});
writeFileSync(root+'/SAM6D/results/interface_preview.png',Buffer.from(screenshot.data,'base64'));
await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
assert(await evaluate('document.body.scrollWidth<=window.innerWidth+1'));
assert.deepEqual(errors,[]);
const proof={status:'passed',server_file_picker:true,server_paths_verified:true,upload_mode_available:true,rgbd_detection:true,cuda:true,fastsam_pose:true,pose_json_download:true,mobile_overflow:false,javascript_errors:errors};
writeFileSync(root+'/SAM6D/pipeline/prepared/sam6d_ui/browser_verification.json',JSON.stringify(proof,null,2));
console.log(JSON.stringify(proof));
await send('Page.close');socket.close();chrome.kill();
