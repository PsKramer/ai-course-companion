const SY={
 mats(P,M,seed){let s=seed>>>0;const r=()=>{s=(s*1103515245+12345)>>>0;return(s>>>16)/65536};
  const X=[...Array(M)].map(()=>[...Array(P)].map(()=>Math.floor(r()*4)));const W=[...Array(P)].map(()=>[...Array(P)].map(()=>1+Math.floor(r()*3)));return{X,W}},
 total(P,M,mode){return mode==='ws'?M+2*P-2:Math.ceil(M/P)*(3*P-2)},
 state(P,M,mode,t,X,W){const pe=[];let busy=0;
  if(mode==='ws'){for(let k=0;k<P;k++){pe.push([]);for(let n=0;n<P;n++){const m=t-k-n;const on=m>=0&&m<M;let ps=null;if(on){ps=0;for(let j=0;j<=k;j++)ps+=X[m][j]*W[j][n];busy++}pe[k].push({on,stat:W[k][n],act:on?X[m][k]:null,ps,m})}}
   const done=[];for(let m=0;m<M;m++)for(let n=0;n<P;n++)if(m+n+P-1<=t){let y=0;for(let j=0;j<P;j++)y+=X[m][j]*W[j][n];done.push({m,n,y})}return{pe,busy,done}}
  const L=3*P-2,tile=Math.floor(t/L),tau=t-tile*L;
  for(let r=0;r<P;r++){pe.push([]);for(let n=0;n<P;n++){const m=tile*P+r,k=tau-r-n;const on=m<M&&k>=0&&k<P;let acc=null;if(m<M){acc=0;for(let j=0;j<=Math.min(k,P-1);j++)if(j>=0)acc+=X[m][j]*W[j][n]}if(on)busy++;pe[r].push({on,stat:acc,act:on?X[m][k]:null,w:on?W[k][n]:null,m})}}
  const done=[];for(let m=0;m<M;m++){const tl=Math.floor(m/P);if((tl+1)*L-1<=t)for(let n=0;n<P;n++){let y=0;for(let j=0;j<P;j++)y+=X[m][j]*W[j][n];done.push({m,n,y})}}return{pe,busy,done,tile}},
 util(P,M,mode,t,X,W){let b=0;for(let c=0;c<=t;c++)b+=this.state(P,M,mode,c,X,W).busy;return b/(P*P*(t+1))}};
if(typeof module!=='undefined')module.exports=SY;
