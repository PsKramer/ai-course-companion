const AR={
 ring(S,n,a,b){return{lat:2*(n-1)*a,bw:2*(n-1)/n*S/b}},
 tree(S,n,a,b){return{lat:2*Math.log2(n)*a,bw:2*S/b}},
 nvls(S,n,a,b){return{lat:2*a,bw:S/b}},
 hier(S,n,g,ai,bi,ao,bo){const inner={lat:2*(g-1)*ai,bw:2*(g-1)/g*S/bi};const outer=n/g>1?this.ring(S/g,n/g,ao,bo):{lat:0,bw:0};return{lat:inner.lat+outer.lat,bw:inner.bw+outer.bw,outer:outer.lat+outer.bw}},
 T:o=>o.lat+o.bw,
 cross(n,a,b){return n*b*a*(n-1-Math.log2(n))},
 busBW(S,n,t){return 2*(n-1)/n*S/t}};
if(typeof module!=='undefined')module.exports=AR;
