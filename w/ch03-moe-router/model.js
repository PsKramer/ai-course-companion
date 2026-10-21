const MOE=(function(){
const D=64;
function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function gauss(r){return Math.sqrt(-2*Math.log(r()+1e-12))*Math.cos(2*Math.PI*r())}
const sig=z=>1/(1+Math.exp(-z));
// Course-MoE parameter math (3.5 Step 3)
const ATT=37748736,ROUTER=4096*128,NORMS=2*4096,EXP=3*4096*1536,V=128256*4096,L=48;
function params(k,shared,E=128){const per=ATT+4096*E+NORMS;const act=L*(per+(k+(shared?1:0))*EXP)+V+4096;const tot=L*(per+(E+(shared?1:0))*EXP)+2*V+4096;return{act,tot,flops:2*act,perExpertAll:L*EXP}}
function make(E,seed){const r=rng(seed);const W=[];for(let i=0;i<=D;i++){const row=new Float64Array(E);for(let e=0;e<E;e++)row[e]=gauss(r)*0.12;W.push(row)}
 for(let e=0;e<E;e++)W[D][e]=-1.0+(e%13===0?1.6:0)+(e%7===0?0.7:0);   // some experts popular a priori
 const C=[];for(let c=0;c<6;c++){const v=new Float64Array(D);for(let i=0;i<D;i++)v[i]=gauss(r)*1.2;C.push(v)}
 return{E,W,C,bias:new Float64Array(E),r:rng(seed+99),hist:[]}}
function batch(st,n){const X=[];for(let t=0;t<n;t++){const c=st.C[Math.floor(st.r()*st.C.length)];const x=new Float64Array(D+1);for(let i=0;i<D;i++)x[i]=c[i]+gauss(st.r);x[D]=1;X.push(x)}return X}
function aff(st,x){const E=st.E,s=new Float64Array(E);for(let e=0;e<E;e++){let z=0;for(let i=0;i<=D;i++)z+=x[i]*st.W[i][e];s[e]=sig(z)}return s}
function route(st,X,k,useBias){const E=st.E,cnt=new Float64Array(E),out=[];for(const x of X){const s=aff(st,x);const idx=[...s.keys()].sort((a,b)=>(s[b]+(useBias?st.bias[b]:0))-(s[a]+(useBias?st.bias[a]:0))).slice(0,k);
  const g=idx.map(i=>s[i]);const z=g.reduce((a,b)=>a+b,0);idx.forEach(i=>cnt[i]++);out.push({s,idx,g:g.map(v=>v/z)})}return{cnt,out}}
function stats(cnt){const E=cnt.length,mean=cnt.reduce((a,b)=>a+b,0)/E;let mx=0,zero=0;cnt.forEach(c=>{mx=Math.max(mx,c);if(c===0)zero++});return{mean,max:mx,ratio:mx/mean,zero}}
function step(st,n,k,mode,gamma,alpha){const X=batch(st,n);const R=route(st,X,k,mode==='bias');const E=st.E,mean=n*k/E;
 if(mode==='bias')for(let e=0;e<E;e++)st.bias[e]-=gamma*Math.sign(R.cnt[e]-mean);
 if(mode==='aux'){const f=R.cnt.map(c=>c/(n*k)*E);const G=st.W.map(()=>new Float64Array(E));
  R.out.forEach((o,t)=>{const s=o.s;let S=0;s.forEach(v=>S+=v);let fp=0;for(let e=0;e<E;e++)fp+=f[e]*s[e]/S;
   for(let e=0;e<E;e++){const dz=(f[e]-fp)/(n*S)*s[e]*(1-s[e]);if(dz!==0)for(let i=0;i<=D;i++)G[i][e]+=X[t][i]*dz}});
  const lr=alpha*E*40;for(let i=0;i<=D;i++)for(let e=0;e<E;e++)st.W[i][e]-=lr*G[i][e]}
 st.hist.push(R.cnt);if(st.hist.length>10)st.hist.shift();return{X,R,st:stats(R.cnt)}}
function smooth(st){const E=st.E,c=new Float64Array(E);st.hist.forEach(h=>h.forEach((v,e)=>c[e]+=v));return stats(c)}
return{make,batch,route,step,stats,smooth,params,aff};})();
if(typeof module!=='undefined')module.exports=MOE;
