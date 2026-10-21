const RF={
 CHIPS:{h100:{name:'H100 SXM',bf16:989e12,fp8:1979e12,fp4:null,bw:3.35e12,sms:132,clk:1.83e9},
  b200:{name:'B200',bf16:2250e12,fp8:4500e12,fp4:9000e12,bw:7.7e12,sms:148,clk:1.85e9},
  mi355x:{name:'MI355X',bf16:2500e12,fp8:5000e12,fp4:null,bw:8e12,sms:null,clk:null},
  tpu7x:{name:'TPU7x',bf16:2307e12,fp8:4614e12,fp4:null,bw:7.38e12,sms:null,clk:null}},
 B:{bf16:2,fp8:1,fp4:0.5},
 gemm(M,K,N,b){const F=2*M*K*N,Q=b*(M*K+K*N+M*N);return{F,Q,I:F/Q}},
 attain(I,peak,bw){return Math.min(peak,I*bw)},
 ridge:(peak,bw)=>peak/bw,
 smemBW:c=>c.sms?c.sms*128*c.clk:null,
 tileI:(m,n,b)=>m*n/(m+n)*(2/b),
 crossM(K,N,b,ridge){for(let M=1;M<=65536;M++)if(this.gemm(M,K,N,b).I>=ridge)return M;return null}};
if(typeof module!=='undefined')module.exports=RF;
