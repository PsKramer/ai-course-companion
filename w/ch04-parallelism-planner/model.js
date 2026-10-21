// Port of chapters/ch04/exercises/ex_4_5_1_solution.py (first-order parallelism planner)
const PLAN={NVLINK:450e9,IB:50e9,PEAK:989e12,MFU:0.40,
 ring:(b,n)=>2*(n-1)/n*b,
 act(m,rc){return rc==='full'?2*m.d:2*(6*m.d+2*m.h_kv*m.d_h+3*m.d_ff)},
 plan(m,n,tp,pp,mode,gs=512,mb=1,rc='none',shard=8,tpAcross=false){
  const dp=Math.floor(n/(tp*pp));if(dp<1||dp*tp*pp!==n)return{err:'GPUs must equal TP × PP × data-parallel size'};
  const P=m.N/(tp*pp),nm=Math.floor(gs/(dp*mb));if(nm<1)return{err:'Global batch too small: fewer than 1 micro-batch per data-parallel rank'};
  let state;if(mode==='ddp')state=16*P;else if(mode==='zero1')state=4*P+12*P/dp;else if(mode==='zero2')state=2*P+14*P/dp;else if(mode==='fsdp')state=16*P/dp;else state=16*P/Math.min(shard,dp);
  const inflight=pp>1?Math.min(pp,nm):1;const acts=this.act(m,rc)/tp*(m.L/pp)*m.T*mb*inflight;
  const g=2*P;let di=0,de=0;
  if(['ddp','zero1','zero2'].includes(mode))de=this.ring(g,dp);else if(mode==='fsdp')de=1.5*this.ring(g,dp);else{const s=Math.min(shard,dp);di=1.5*this.ring(g,s);de=dp/s>1?this.ring(g/s,dp/s):0}
  const tpb=tp>1?(m.L/pp)*nm*4*this.ring(mb*m.T*m.d*2,tp):0;const ppb=pp>1?nm*2*(mb*m.T*m.d*2/tp):0;
  const bubble=(pp-1)/(nm+pp-1);const tc=6*m.N*gs*m.T/(n*this.PEAK*this.MFU);
  const nvl=di+(tpAcross?0:tpb),ib=de+ppb+(tpAcross?tpb:0);const tn=nvl/this.NVLINK,ti=ib/this.IB;
  const tcb=tc/(1-bubble);const step=Math.max(tcb,tn,ti);const lim=step===tcb?(bubble>0?'compute (incl. pipeline bubble)':'compute'):step===tn?'NVLink communication':'InfiniBand communication';
  return{dp,nm,state,acts,total:state+acts,nvl,ib,bubble,tc,tn,ti,step,lim}}};
if(typeof module!=='undefined')module.exports=PLAN;
