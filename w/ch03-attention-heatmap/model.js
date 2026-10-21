const AH={
 row(scores,i,mask){const T=scores.length,r=scores[i],n=mask?i+1:T;let m=-Infinity;for(let j=0;j<n;j++)m=Math.max(m,r[j]);
  const w=new Array(T).fill(null);let z=0;for(let j=0;j<n;j++){w[j]=Math.exp(r[j]-m);z+=w[j]}for(let j=0;j<n;j++)w[j]/=z;return w},
 matrix(scores,mask){return scores.map((_,i)=>this.row(scores,i,mask))},
 entropy(w){let h=0;w.forEach(x=>{if(x>0)h-=x*Math.log2(x)});return h},
 top(w,k){return w.map((x,j)=>[x,j]).filter(p=>p[0]!==null).sort((a,b)=>b[0]-a[0]).slice(0,k)},
 kvBytes:(L,hkv,dh,b)=>2*L*hkv*dh*b};
if(typeof module!=='undefined')module.exports=AH;
