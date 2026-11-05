import {Game,W,H,R,PADDLE_W,PADDLE_Y,clamp,inputs} from './game.js';
import {Network} from './network.js';
const $=id=>document.getElementById(id), game=new Game(),ctx=$('game').getContext('2d'),chart=$('chart').getContext('2d');
const STORAGE='breakout-lab-model-v1';
let mode='human',model=null,network=null,worker=null,history=[],keys=new Set(),target=null,last=0,accumulator=0,noticeTimer;
function notice(message){$('notice').textContent=message;$('notice').hidden=false;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>$('notice').hidden=true,6000);}
function install(data,persist=true){network=Network.load(data);model=data;if(persist){try{localStorage.setItem(STORAGE,JSON.stringify(data));}catch{notice('Model loaded, but browser storage is unavailable. Export weights to keep them.');}}
  $('model-title').textContent='Your paddle has a brain.';
  $('model-info').textContent=`769 learned parameters${Number.isFinite(data.error)?` · ${data.error.toFixed(1)} px validation error`:''}. Weights loaded and ready to play.`;
  $('model-dot').style.background='var(--mint)';$('deploy').disabled=false;$('export').disabled=false;
}
try{const saved=localStorage.getItem(STORAGE);if(saved)install(JSON.parse(saved),false);}catch{notice('Saved weights could not be loaded. Train or import a model.');}
function setMode(next){if(next==='ai'&&!network){notice('Train a network or import saved weights first.');return;}mode=next;keys.clear();target=null;
  for(const id of ['human','ai']){$(id).classList.toggle('selected',mode===id);$(id).setAttribute('aria-pressed',String(mode===id));}
  $('live-label').innerHTML=`<i></i> ${mode==='ai'?'NEURAL NETWORK':'HUMAN'} IN CONTROL`;sync();
}
$('human').onclick=()=>setMode('human');$('ai').onclick=()=>setMode('ai');
$('deploy').onclick=()=>{setMode('ai');game.reset();game.launch();sync();$('game').focus();};
function toggle(){if(game.state==='won'||game.state==='lost')game.reset();game.launch();sync();}
$('play').onclick=()=>{toggle();$('game').focus();};$('pause').onclick=toggle;
$('reset').onclick=()=>{game.reset();sync();};
window.addEventListener('keydown',e=>{
  if(e.target.closest('select,input,button,a'))return;
  if(['ArrowLeft','ArrowRight','a','d','A','D',' '].includes(e.key)){e.preventDefault();keys.add(e.key.toLowerCase());if(e.key===' '&&!e.repeat)toggle();}
});window.addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));
window.addEventListener('blur',()=>{keys.clear();if(game.state==='playing'){game.state='paused';sync();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&game.state==='playing'){game.state='paused';sync();}});
$('game').addEventListener('pointermove',e=>{if(mode==='human'&&e.pointerType!=='mouse'){const rect=$('game').getBoundingClientRect();target=(e.clientX-rect.left)/rect.width*W;}});
$('game').addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse'){e.preventDefault();$('game').setPointerCapture(e.pointerId);if(game.state!=='playing')toggle();const rect=$('game').getBoundingClientRect();target=(e.clientX-rect.left)/rect.width*W;}});
$('game').addEventListener('pointerup',()=>target=null);
function sync(){
  $('score').textContent=String(game.score).padStart(4,'0');$('bricks').textContent=game.bricks.filter(b=>b.alive).length;$('lives').textContent='● '.repeat(game.lives).trim()||'—';
  $('pause').textContent=game.state==='playing'?'Ⅱ Pause':game.state==='paused'?'▷ Resume':'▷ Launch';
  $('overlay').hidden=game.state==='playing';
  const copy={ready:[mode==='human'?"YOU’RE IN CONTROL":'NETWORK IN CONTROL',game.lives===3?'Meet your paddle.':'Another chance.',mode==='human'?'Move with ← → or A / D. Keep the ball in play.':'The saved network will move the paddle. Watch it learn.','Start playing'],paused:['TAKE A BREATHER','Game paused.','Your paddle will be right here.','Resume game'],won:['ALL CLEAR','Nicely played.','Every brick down. Ready for another round?','Play again'],lost:['EXPERIMENT COMPLETE','One more round?','Three lives, a little wiser. Give it another go.','Try again']}[game.state];
  if(copy){$('overlay-tag').textContent=copy[0];$('overlay-title').textContent=copy[1];$('overlay-copy').textContent=copy[2];$('play').textContent=copy[3]+' ↗';}
}
function rounded(x,y,w,h,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();}
function draw(){ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#a5b89a10';for(let x=20;x<W;x+=24)for(let y=18;y<H;y+=24)ctx.fillRect(x,y,1,1);
  const colors=['#bbef8b','#9bcf73','#78aa69','#5e8968','#426a5b'];
  for(const b of game.bricks)if(b.alive){rounded(b.x,b.y,b.w,b.h,3,colors[b.row]);ctx.fillStyle='#ffffff22';ctx.fillRect(b.x+3,b.y+2,b.w-6,1);}
  for(let i=0;i<game.trail.length;i++){const p=game.trail[i];ctx.beginPath();ctx.arc(p.x,p.y,R*i/game.trail.length,0,Math.PI*2);ctx.fillStyle=`rgba(197,244,157,${i/game.trail.length*.2})`;ctx.fill();}
  if(mode==='ai'&&network&&game.state==='playing'){const x=clamp((network.predict(inputs(game.ball,game.paddle))+1)/2*W,PADDLE_W/2,W-PADDLE_W/2);ctx.setLineDash([4,7]);ctx.strokeStyle='#b9f57735';ctx.beginPath();ctx.moveTo(x,PADDLE_Y-35);ctx.lineTo(x,H-10);ctx.stroke();ctx.setLineDash([]);}
  ctx.shadowColor='#b9f577';ctx.shadowBlur=16;rounded(game.paddle-PADDLE_W/2,PADDLE_Y,PADDLE_W,11,5,'#c6f397');ctx.beginPath();ctx.arc(game.ball.x,game.ball.y,R,0,Math.PI*2);ctx.fillStyle='#f1f7df';ctx.fill();ctx.shadowBlur=0;
  ctx.strokeStyle='#354231';ctx.beginPath();ctx.moveTo(24,H-16);ctx.lineTo(W-24,H-16);ctx.stroke();
}
function frame(t){const elapsed=Math.min((t-last)/1000, .08);last=t;accumulator+=elapsed;
  while(accumulator>=1/120){const prev=game.state;const movement=(keys.has('arrowright')||keys.has('d')?1:0)-(keys.has('arrowleft')||keys.has('a')?1:0);const aiTarget=mode==='ai'&&network?clamp((network.predict(inputs(game.ball,game.paddle))+1)*W/2,PADDLE_W/2,W-PADDLE_W/2):target;
    game.step(.5,movement,aiTarget);accumulator-=1/120;if(prev!==game.state)sync();}
  $('score').textContent=String(game.score).padStart(4,'0');$('bricks').textContent=game.bricks.filter(b=>b.alive).length;draw();requestAnimationFrame(frame);
}
function drawChart(){const w=620,h=130;chart.clearRect(0,0,w,h);chart.strokeStyle='#354132';chart.lineWidth=1;for(let y=15;y<h;y+=36){chart.beginPath();chart.moveTo(0,y);chart.lineTo(w,y);chart.stroke();}if(history.length<2)return;const max=Math.max(...history)*1.1;chart.beginPath();history.forEach((v,i)=>{const x=i/(history.length-1)*w,y=h-8-v/max*(h-16);i?chart.lineTo(x,y):chart.moveTo(x,y);});chart.strokeStyle='#b9f577';chart.lineWidth=2;chart.stroke();}
function stopTraining(){worker?.terminate();worker=null;$('train').innerHTML='✧ <span>Train neural network</span><span>↗</span>';$('epochs').disabled=false;}
$('epochs').onchange=()=>{if(!worker)$('epoch').textContent=`0 / ${$('epochs').value}`;};
$('train').onclick=()=>{if(worker){stopTraining();$('train-status').textContent='Training stopped';return;}
  history=[];drawChart();$('epochs').disabled=true;$('train').innerHTML='□ <span>Stop training</span>';$('train-status').textContent='Starting training…';
  try{worker=new Worker(new URL('./train-worker.js',import.meta.url),{type:'module'});worker.onmessage=({data})=>{
    if(data.type==='progress'){history.push(data.error);$('train-status').textContent='Learning in the background';$('epoch').textContent=`${data.epoch} / ${data.epochs}`;$('progress').style.width=`${data.epoch/data.epochs*100}%`;$('loss').innerHTML=`${data.error.toFixed(1)} <small>px</small>`;$('samples').textContent=`${data.examples.toLocaleString()} training examples`;drawChart();}
    if(data.type==='complete'){try{install(data.model);$('train-status').textContent='Training complete · weights saved';notice('Training complete. Select Watch AI play to try your network.');}catch(e){notice(e.message);}stopTraining();}
    if(data.type==='error'){notice(data.message);stopTraining();$('train-status').textContent='Training failed. Please retry.';}
  };worker.onerror=()=>{stopTraining();$('train-status').textContent='Training failed';notice('Could not start training. Run the app with npm start and open localhost:3000.');};worker.postMessage({epochs:Number($('epochs').value)});
  }catch(e){stopTraining();notice(e.message);}
};
$('export').onclick=()=>{if(!model)return;const url=URL.createObjectURL(new Blob([JSON.stringify(model,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='breakout-weights.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{const file=$('file').files[0];if(!file)return;try{if(file.size>150000)throw new Error('Weight file is too large.');const data=JSON.parse(await file.text());install(data);notice('Weights imported. Your network is ready to play.');}catch(e){notice('Could not import weights: '+e.message);}$('file').value='';};
sync();drawChart();requestAnimationFrame(frame);
