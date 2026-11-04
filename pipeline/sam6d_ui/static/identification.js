/* Keep the current component visible independently of the initial job result. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
let assembly=null,key=null,name=null,score=null,confirmed=false;
let detail='Abre la cámara e inicia la identificación.';
function render(){
  $('identified-component-title').textContent=key?(confirmed?'Pieza identificada':'Última pieza identificada'):'Identificación del ensamblaje';
  $('identified-component-name').textContent=name||'Pendiente de identificar';
  $('identified-component-meta').textContent=key?'CAD: '+key+(Number.isFinite(score)?' · Score SAM6D: '+score.toFixed(3):''):'';
  $('identified-component-detail').textContent=detail;
}
function sync(){
  const active=$('file-source').value==='realsense'&&$('camera-cad-source').value==='assembly';
  $('identified-component').hidden=!active;
  const selected=active?$('assembly-select').value:null;
  if(selected!==assembly){
    assembly=selected;key=name=score=null;confirmed=false;
    detail='Abre la cámara e inicia la identificación.';
  }
  render();return active;
}
function pending(){
  if(!sync())return;
  confirmed=false;detail='Comparando el fotograma actual con las piezas del ensamblaje…';render();
}
function result(value){
  if(!sync()||value.identification?.assembly!==assembly)return;
  key=value.identification.part_key;name=key?value.display_name:'Sin identificación fiable';
  score=key?value.best_score:null;confirmed=!!key;
  detail=key?'Identificación actualizada a las '+new Date().toLocaleTimeString()+'.':'Ninguna pieza supera el umbral de aceptación.';
  render();
}
function tracking(value){
  if(!sync()||!value.identification)return;
  confirmed=value.running&&value.state==='tracking';
  if(confirmed){
    if(key!==value.cad_key)score=null;
    key=value.cad_key;name=value.display_name||value.cad_key;
    if(Number.isFinite(value.last_detection_score))score=value.last_detection_score;
    detail=value.sam_running?'Siguiendo esta pieza · SAM está comprobando el ensamblaje…':
      'Seguimiento activo · '+(value.component_changes||0)+' cambios de pieza'+
      (value.last_redetect_at?' · Última comprobación: '+new Date(value.last_redetect_at*1000).toLocaleTimeString():'');
  }else detail=value.error||value.redetect_error||(value.running?'Buscando y confirmando la pieza actual…':'Seguimiento detenido.');
  render();
}
function error(message){if(sync()){confirmed=false;detail=message;render();}}
window.AssemblyIdentification={sync,pending,result,tracking,error};
})();
