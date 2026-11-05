export const SHAPE = [5,24,24,1];
export function rng(seed=42) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export class Network {
  constructor(weights, random = Math.random) {
    this.weights = weights || SHAPE.slice(1).map((n,l)=>Array.from({length:n},()=>Array.from({length:SHAPE[l]+1},(_,j)=>j===SHAPE[l]?0:(random()*2-1)*Math.sqrt(6/(SHAPE[l]+n)))));
    this.activations = SHAPE.map(n=>new Float64Array(n));
    this.deltas = SHAPE.map(n=>new Float64Array(n));
  }
  predict(input) {
    this.activations[0].set(input);
    for(let l=0;l<this.weights.length;l++) {
      const a=this.activations[l], out=this.activations[l+1];
      for(let j=0;j<out.length;j++) { const w=this.weights[l][j]; let sum=w[a.length]; for(let i=0;i<a.length;i++) sum+=w[i]*a[i]; out[j]=Math.tanh(sum); }
    }
    return this.activations.at(-1)[0];
  }
  train(input,target,rate=.008) {
    const pred=this.predict(input), a=this.activations, d=this.deltas;
    d[3][0]=(pred-target)*(1-pred*pred);
    for(let l=2;l>0;l--) for(let i=0;i<SHAPE[l];i++) { let sum=0; for(let j=0;j<SHAPE[l+1];j++) sum+=this.weights[l][j][i]*d[l+1][j]; d[l][i]=sum*(1-a[l][i]*a[l][i]); }
    for(let l=0;l<3;l++) for(let j=0;j<SHAPE[l+1];j++) { const w=this.weights[l][j], delta=rate*d[l+1][j]; for(let i=0;i<SHAPE[l];i++) w[i]-=delta*a[l][i]; w[SHAPE[l]]-=delta; }
    return (pred-target)**2;
  }
  serialize() { return {version:1,shape:SHAPE,weights:this.weights}; }
  static load(data) {
    if(data?.version!==1 || JSON.stringify(data.shape)!==JSON.stringify(SHAPE) || !Array.isArray(data.weights) || data.weights.length!==3) throw new Error('Unsupported model format.');
    for(let l=0;l<3;l++) if(!Array.isArray(data.weights[l]) || data.weights[l].length!==SHAPE[l+1] || data.weights[l].some(row=>!Array.isArray(row)||row.length!==SHAPE[l]+1||row.some(v=>!Number.isFinite(v)||Math.abs(v)>100))) throw new Error('Invalid model weights.');
    return new Network(data.weights);
  }
}
