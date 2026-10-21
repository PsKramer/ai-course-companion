// Model for ch08-efficiency-stack (Figure 8.6.1 ladder)
const EFF={N:8.03e9,SKV:131072,BW:3.35e12,
 calc(o){// o:{wbits,kvBytes(2|1),draft,gamma,alpha,hybrid,T,draftCost}
  const bw=o.wbits/8,skv=this.SKV*(o.kvBytes/2)/(o.hybrid?8:1);
  const tw=this.N*bw/this.BW,tkv=o.T*skv/this.BW;
  const tau=o.draft?(1-Math.pow(o.alpha,o.gamma+1))/(1-o.alpha):1;
  const round=(o.draft?o.gamma*o.draftCost*tw:0)+tw+tkv;
  return{bw,skv,tw,tkv,tau,round,tps:tau/round,wGB:this.N*bw/1e9,kvGB:o.T*skv/1e9,bytes:this.N*bw+o.T*skv}}};
if(typeof module!=='undefined')module.exports=EFF;
