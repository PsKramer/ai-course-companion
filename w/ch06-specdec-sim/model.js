// Model for ch06-specdec-sim (6.6 speculative decoding: when does it pay?)
const SDS={N:8.03e9,SKV:131072,
 tau:(a,g)=>(1-Math.pow(a,g+1))/(1-a),
 calc(o){// o:{a,g,B,T,eff,draftGB,bw,peak}
  const bytes=2*this.N+o.B*o.T*this.SKV;const mem=bytes/o.bw;
  const normal=Math.max(mem,2*this.N*o.B/(o.eff*o.peak));
  const verify=Math.max(mem,2*this.N*o.B*(o.g+1)/(o.eff*o.peak));
  const draft=o.g*o.draftGB*1e9/o.bw;const tau=o.g>0?this.tau(o.a,o.g):1;
  const base=o.B/normal,spec=o.B*tau/(verify+draft);
  return{tau,normal,verify,draft,base,spec,speedup:spec/base,memBound:verify===mem}},
 crossover(o){for(let B=1;B<=4096;B++){if(this.calc({...o,B}).speedup<1)return B}return null}};
if(typeof module!=='undefined')module.exports=SDS;
