// Model for ch08-agent-context-cost (mirrors chapters/ch08/exercises/ex_8_2_1_solution.py)
const ACC={N:8.03e9,KV:131072,BW:3.35e12,PEAK:989e12,MFU:0.5,
 episode(turns,S,u,r,cache,window){const N=this.N,W=2*N;let ctx=S,pre=0,preS=0,dec=0,cached=0,overflow=null;const bars=[];
  for(let t=0;t<turns;t++){const nw=cache?ctx-cached:ctx;pre+=nw;preS+=2*N*nw/(this.MFU*this.PEAK);
   dec+=(r*W+this.KV*(r*ctx+r*(r-1)/2))/this.BW;
   bars.push({turn:t+1,ctxBefore:ctx,newPrefill:nw});
   ctx+=r+u;cached=cache?ctx-u:0;if(overflow===null&&ctx>window)overflow=t+1;}
  return{ctx,pre,preS,dec,decTok:turns*r,kvGB:ctx*this.KV/1e9,overflow,bars}},
 firstOverflow(S,u,r,window){const d=r+u;let t=1;while(S+t*d<=window)t++;return t},
 fullTurns(S,u,r,window){return Math.max(0,Math.floor((window-S)/(r+u)))}};
if(typeof module!=='undefined')module.exports=ACC;
