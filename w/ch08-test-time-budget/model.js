// Model for ch08-test-time-budget (mirrors chapters/ch08/exercises/ex_8_1_1_solution.py)
const TTB={N:8.03e9,KV:131072,BW:3.35e12,KVBUDGET:56e9,
 coverage:(p,n)=>1-Math.pow(1-p,n),
 minN:(p,c)=>Math.ceil(Math.log(1-c)/Math.log(1-p)),
 decodeSeconds(B,prompt,out){const w=2*this.N;return (out*w+B*this.KV*(out*prompt+out*(out-1)/2))/this.BW},
 maxPerWave(prompt,out){return Math.floor(this.KVBUDGET/((prompt+out)*this.KV))},
 run(p,n,out,prompt,price){const per=this.maxPerWave(prompt,out);const waves=[];let left=n;
  if(per<1)return{p,n,cov:this.coverage(p,n),per,waves:[],secs:Infinity,tokens:n*out,usd:Infinity,usdPerSolve:Infinity,seq:n*this.decodeSeconds(1,prompt,out),kvAll:n*(prompt+out)*this.KV};
  while(left>0){const b=Math.min(per,left);waves.push({B:b,s:this.decodeSeconds(b,prompt,out)});left-=b}
  const secs=waves.reduce((a,w)=>a+w.s,0);const usd=secs*price/3600;const cov=this.coverage(p,n);
  return{p,n,cov,per,waves,secs,tokens:n*out,usd,usdPerSolve:usd/cov,seq:n*this.decodeSeconds(1,prompt,out),kvAll:n*(prompt+out)*this.KV}}};
if(typeof module!=='undefined')module.exports=TTB;
