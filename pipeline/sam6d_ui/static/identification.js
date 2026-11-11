/* Keep the current component visible independently of the initial job result. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
let assembly=null,key=null,name=null,score=null,confirmed=false;
let detail='Open the camera and start identification.';
function render(){
  $('identified-component-title').textContent=key?(confirmed?'Identified part':'Last identified part'):'Assembly identification';
  $('identified-component-name').textContent=name||'Awaiting identification';
  $('identified-component-meta').textContent=key?'CAD: '+key+(Number.isFinite(score)?' · Score SAM6D: '+score.toFixed(3):''):'';
  $('identified-component-detail').textContent=detail;
}
function sync(){
  const active=$('file-source').value==='realsense'&&$('camera-cad-source').value==='assembly';
  $('identified-component').hidden=!active;
  const selected=active?$('assembly-select').value:null;
  if(selected!==assembly){
    assembly=selected;key=name=score=null;confirmed=false;
    detail='Open the camera and start identification.';
  }
  render();return active;
}
function pending(){
  if(!sync())return;
  confirmed=false;detail='Comparing the current frame against assembly parts…';render();
}
function result(value){
  if(!sync()||value.identification?.assembly!==assembly)return;
  key=value.identification.part_key;name=key?value.display_name:'No reliable identification';
  score=key?value.best_score:null;confirmed=!!key;
  detail=key?'Identification updated at '+new Date().toLocaleTimeString()+'.':'No part exceeds the acceptance threshold.';
  render();
}
function tracking(value){
  if(!sync()||!value.identification)return;
  confirmed=value.running&&value.state==='tracking';
  if(confirmed){
    if(key!==value.cad_key)score=null;
    key=value.cad_key;name=value.display_name||value.cad_key;
    if(Number.isFinite(value.last_detection_score))score=value.last_detection_score;
    detail=value.sam_running?'Tracking this part · SAM is checking the assembly…':
      'Seguimiento activo · '+(value.component_changes||0)+' part changes'+
      (value.last_redetect_at?' · Last check: '+new Date(value.last_redetect_at*1000).toLocaleTimeString():'');
  }else detail=value.error||value.redetect_error||(value.running?'Finding and confirming the current part…':'Tracking stopped.');
  render();
}
function error(message){if(sync()){confirmed=false;detail=message;render();}}
window.AssemblyIdentification={sync,pending,result,tracking,error};
})();
