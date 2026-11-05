import {train} from './trainer.js';
self.onmessage=({data})=>{try {const model=train({epochs:data.epochs,onProgress:progress=>self.postMessage({type:'progress',...progress})});self.postMessage({type:'complete',model});}catch(error){self.postMessage({type:'error',message:error.message});}};
