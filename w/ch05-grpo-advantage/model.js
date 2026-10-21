const GA={
 adv(G,c,std){const r=Array.from({length:G},(_,i)=>i<c?1:0);const m=c/G;const s=Math.sqrt(m*(1-m));return r.map(x=>s===0?0:(std?(x-m)/s:x-m))},
 zeroShare:(p,G)=>Math.pow(p,G)+Math.pow(1-p,G),
 half(G){let lo=0.5,hi=1;for(let i=0;i<60;i++){const m=(lo+hi)/2;if(this.zeroShare(m,G)<0.5)lo=m;else hi=m}return(lo+hi)/2},
 stdW:q=>q<=0||q>=1?Infinity:1/Math.sqrt(q*(1-q)),
 weights(G,c,lc,lw,agg){const tot=c*lc+(G-c)*lw;return agg==='seq'?{wc:1/(G*lc),ww:1/(G*lw)}:{wc:1/tot,ww:1/tot}}};
if(typeof module!=='undefined')module.exports=GA;
