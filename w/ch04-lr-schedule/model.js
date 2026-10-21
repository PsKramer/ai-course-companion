// Model for ch04-lr-schedule (4.3): linear warmup, then cosine (to floor) or WSD (warmup-stable-decay, linear to 0)
const LRS={
 cosine(s,o){if(s<o.warm)return o.peak*s/o.warm;const p=Math.min(1,(s-o.warm)/(o.total-o.warm));return o.peak*(o.floor+(1-o.floor)*0.5*(1+Math.cos(Math.PI*p)))},
 wsd(s,o){if(s<o.warm)return o.peak*s/o.warm;const d0=o.total*(1-o.decay);if(s<d0)return o.peak;return o.peak*Math.max(0,(o.total-s)/(o.total-d0))},
 branch(s,o,b){// decay branched off the plateau at step b, lasting decay*b steps
  const len=o.decay*b;if(s<b)return this.wsd(s,o);if(s>b+len)return null;return o.peak*(1-(s-b)/len)},
 area(fn,o,n=20000){let a=0;for(let i=0;i<n;i++)a+=fn.call(this,(i+0.5)/n*o.total,o);return a/n}};
if(typeof module!=='undefined')module.exports=LRS;
