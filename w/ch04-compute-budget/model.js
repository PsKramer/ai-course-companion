// Model for ch04-compute-budget (4.2 scaling laws): L(N,D) = E + A/N^alpha + B/D^beta
const CB={FITS:{epoch:{E:1.8172,A:482.01,B:2085.43,a:0.3478,b:0.3658,name:'Epoch 2024 refit'},hoff:{E:1.69,A:406.4,B:410.7,a:0.34,b:0.28,name:'Hoffmann et al. 2022, method 3'}},
 loss(f,N,D){return f.E+f.A/Math.pow(N,f.a)+f.B/Math.pow(D,f.b)},
 lossC(f,N,C){return this.loss(f,N,C/(6*N))},
 optC(f,C){let best=null;for(let l=8;l<=13;l+=0.0005){const N=10**l,L=this.lossC(f,N,C);if(!best||L<best.L)best={N,D:C/(6*N),L}}return best},
 Dneed(f,N,Lt){const r=Lt-f.E-f.A/Math.pow(N,f.a);if(r<=0)return Infinity;return Math.pow(f.B/r,1/f.b)},
 lifetime(f,N,Lt,S){const D=this.Dneed(f,N,Lt);return{D,train:6*N*D,inf:2*N*S,tot:6*N*D+2*N*S}},
 optLife(f,Lt,S){let best=null;for(let l=8;l<=13;l+=0.0005){const N=10**l,r=this.lifetime(f,N,Lt,S);if(isFinite(r.tot)&&(!best||r.tot<best.tot))best={N,...r}}return best},
 h100h(C){return C/(0.4*989e12*3600)}};
if(typeof module!=='undefined')module.exports=CB;
