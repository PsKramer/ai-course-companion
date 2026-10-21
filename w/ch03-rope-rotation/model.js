const RP={DH:128,
 omega(th,i){return Math.pow(th,-2*i/this.DH)},
 lambda(th,i){return 2*Math.PI/this.omega(th,i)},
 untrained(th,T){let c=0,first=null;for(let i=0;i<this.DH/2;i++)if(this.lambda(th,i)>T){c++;if(first===null)first=i}return{count:c,first}},
 vec:(function(){let s=7;const r=()=>{s=(s*16807)%2147483647;return s/2147483647};const v=[];for(let i=0;i<128;i++){const u=r(),w=r();v.push(Math.sqrt(-2*Math.log(u+1e-12))*Math.cos(2*Math.PI*w))}const n=Math.hypot(...v);return v.map(x=>x/n)})(),
 // score between the same unit vector placed at query and key, distance d (positions), as in RoPE: sum over pairs of |pair|^2 cos(d*omega_i)
 score(th,d){let s=0;for(let i=0;i<64;i++){const a=this.vec[2*i],b=this.vec[2*i+1];s+=(a*a+b*b)*Math.cos(d*this.omega(th,i))}return s}};
if(typeof module!=='undefined')module.exports=RP;
