// Model for ch06-kv-calc (6.2 KV-cache math)
const KVC={
 skv(o){// bytes per token (average over T for sliding-window hybrid)
  if(o.type==='mla')return o.L*o.dc*o.bkv;
  const per=2*o.hkv*o.dh*o.bkv;
  if(o.type==='swa'){const Lg=Math.min(o.Lg,o.L),Ll=o.L-Lg;return per*(Lg*o.T+Ll*Math.min(o.T,o.W))/o.T}
  return per*o.L},
 calc(o){const s=this.skv(o),w=o.N*o.bw,head=0.1*o.hbm,budget=o.hbm-w-head;const perSeq=s*o.T;
  const tokens=Math.max(0,budget/s),users=Math.max(0,Math.floor(budget/perSeq));const need=o.users*perSeq;
  const sFull=o.type==='swa'?2*o.hkv*o.dh*o.bkv*o.L:s;
  return{s,w,head,budget,perSeq,tokens,users,need,fits:need<=budget&&budget>0,cross:w/s,kvStep:need,stepBytes:w+need,stepMs:(w+need)/o.bwth*1e3}}};
if(typeof module!=='undefined')module.exports=KVC;
