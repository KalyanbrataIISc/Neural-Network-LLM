import {Network,rng} from './network.js';
import {coach,inputs,W,PADDLE_Y} from './game.js';
export function sample(random){const angle=(random()*2-1)*.88;const speed=4.5+random();const ball={x:8+random()*(W-16),y:20+random()*(PADDLE_Y-30),vx:Math.sin(angle)*speed,vy:Math.cos(angle)*speed*(random()<.65?1:-1)};return {input:inputs(ball,58+random()*(W-116)),target:coach(ball)/W*2-1};}
export function evaluate(network,validation){let error=0;for(const s of validation)error+=Math.abs(network.predict(s.input)-s.target)*W/2;return error/validation.length;}
export function train({epochs=500,onProgress=()=>{}}={}){
  const random=rng(2026),rngValidation=rng(808),network=new Network(null,random),validation=Array.from({length:512},()=>sample(rngValidation));
  let best=Infinity,checkpoint;onProgress({epoch:0,epochs,error:evaluate(network,validation),examples:0});
  for(let epoch=1;epoch<=epochs;epoch++){
    for(let j=0;j<1024;j++){const s=sample(random);network.train(s.input,s.target,.012*(1-.7*epoch/epochs));}
    if(epoch%5===0||epoch===epochs){const error=evaluate(network,validation);if(error<best){best=error;checkpoint=structuredClone(network.serialize());}onProgress({epoch,epochs,error,examples:epoch*1024});}
  }
  return {...checkpoint,trainedAt:new Date().toISOString(),epochs,error:best,examples:epochs*1024,method:'supervised-interception'};
}
