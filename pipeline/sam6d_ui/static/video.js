/* Latest-only preview: one finite JPEG request, one decode, then the next request. */
(()=>{'use strict';
const players=new Map();
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function stop(img){const old=players.get(img);if(!old)return;old.stopped=true;old.controller?.abort();if(old.url)URL.revokeObjectURL(old.url);players.delete(img);img.removeAttribute('src');delete img.dataset.liveSource;}
function play(img,source){
  if(players.get(img)?.source===source)return;
  stop(img);
  const player={source,stopped:false,controller:null,url:null,lastSequence:-1,lastPaint:null,meanSeconds:null};
  players.set(img,player);img.dataset.liveSource=source;img.hidden=false;
  (async()=>{
    while(!player.stopped){
      if(document.hidden){await pause(200);continue;}
      const started=performance.now();let timeout=null;
      try{
        player.controller=new AbortController();timeout=setTimeout(()=>player.controller.abort(),2000);
        const response=await fetch('/api/'+source+'/frame.jpg?after='+player.lastSequence,{cache:'no-store',signal:player.controller.signal});
        if(response.status===204){await pause(1000/30);continue;}
        if(!response.ok)throw new Error('Could not receive video: HTTP '+response.status);
        const blob=await response.blob();if(player.stopped)break;
        const url=URL.createObjectURL(blob),old=player.url;player.url=url;
        img.src=url;
        try{await img.decode();}finally{if(old)URL.revokeObjectURL(old);}
        if(player.stopped)break;
        img.hidden=false;
        const painted=performance.now();
        const period=player.lastPaint===null?null:(painted-player.lastPaint)/1000;player.lastPaint=painted;
        if(period!==null)player.meanSeconds=player.meanSeconds===null?period:.8*player.meanSeconds+.2*period;
        player.lastSequence=Number(response.headers.get('X-Frame-Sequence'));
        const details={source,sequence:player.lastSequence,fps:player.meanSeconds?1/player.meanSeconds:null,ageMs:Number(response.headers.get('X-Frame-Age-Ms'))+painted-started,transferMs:painted-started,bytes:blob.size};
        img.dataset.frameAgeMs=String(Math.round(details.ageMs));img.dataset.frameSequence=String(player.lastSequence);
        img.dispatchEvent(new CustomEvent('vision-frame',{detail:details}));
      }catch(error){if(!player.stopped){img.hidden=true;img.dispatchEvent(new CustomEvent('vision-video-error',{detail:{message:error.name==='AbortError'?'Video connection timed out; retrying with a fresh image.':error.message}}));await pause(100);}}
      finally{clearTimeout(timeout);player.controller=null;}
      await pause(Math.max(0,1000/30-(performance.now()-started)));
    }
  })();
}
window.VisionVideo={play,stop,active:img=>players.has(img)};
window.addEventListener('pagehide',()=>{for(const img of [...players.keys()])stop(img);});
})();
