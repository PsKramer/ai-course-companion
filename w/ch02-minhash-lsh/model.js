// Model for ch02-minhash-lsh: word shingles, exact Jaccard, MinHash with universal hashes (a*x + c) mod p, LSH banding
const MH=(function(){
const P=2147483647; // Mersenne prime 2^31-1
function mulmod(a,x){const xh=Math.floor(x/65536),xl=x%65536;return ((a*xh%P)*65536+a*xl)%P}
function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function fnv(str){let h=0x811c9dc5;for(let i=0;i<str.length;i++){h^=str.charCodeAt(i);h=Math.imul(h,0x01000193)}return(h>>>0)%P}
let A=[],C=[];function ensure(k){if(A.length>=k)return;const r=rng(12345);A=[];C=[];for(let i=0;i<Math.max(k,9000);i++){A.push(1+Math.floor(r()*(P-1)));C.push(Math.floor(r()*P))}}
const cache=new Map();
function shingles(words,n){const s=new Set();for(let i=0;i+n<=words.length;i++)s.add(words.slice(i,i+n).join(' ').toLowerCase());return s}
function jaccard(a,b){let i=0;a.forEach(x=>{if(b.has(x))i++});return i/(a.size+b.size-i)}
function sig(set,k){ensure(k);const out=new Float64Array(k).fill(Infinity);set.forEach(sh=>{let hv=cache.get(sh);if(!hv||hv.length<k){const x=fnv(sh);hv=new Float64Array(k);for(let j=0;j<k;j++)hv[j]=(mulmod(A[j],x)+C[j])%P;cache.set(sh,hv)}for(let j=0;j<k;j++)if(hv[j]<out[j])out[j]=hv[j]});return out}
function bands(s1,s2,b,r){const m=[];for(let i=0;i<b;i++){let ok=true;for(let j=0;j<r;j++)if(s1[i*r+j]!==s2[i*r+j]){ok=false;break}m.push(ok)}return m}
const curve=(s,b,r)=>1-Math.pow(1-Math.pow(s,r),b);
const thresh=(b,r)=>Math.pow(1/b,1/r);
function edit(words,rate,seed,vocab){const r=rng(seed);const out=words.slice(),ed=new Set();const k=Math.round(rate*words.length);const idx=[...words.keys()];
 for(let i=idx.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}
 idx.slice(0,k).forEach(i=>{let w;do{w=vocab[Math.floor(r()*vocab.length)]}while(w.toLowerCase()===words[i].toLowerCase());out[i]=w;ed.add(i)});return{words:out,edited:ed}}
function compare(w1,w2,n,b,r){const a=shingles(w1,n),c=shingles(w2,n);const k=b*r;const s1=sig(a,k),s2=sig(c,k);const bm=bands(s1,s2,b,r);return{J:jaccard(a,c),s1,s2,bm,cand:bm.some(x=>x)}}
function trials(w1,rate,n,b,r,T,vocab,seed0){let caught=0,Js=0;for(let t=0;t<T;t++){const e=edit(w1,rate,seed0+1000+t,vocab);const c=compare(w1,e.words,n,b,r);if(c.cand)caught++;Js+=c.J}if(cache.size>20000)cache.clear();return{caught,meanJ:Js/T}}
return{shingles,jaccard,sig,bands,curve,thresh,edit,compare,trials,clear:()=>cache.clear()};})();
const MH_DOC="City officials announced on Tuesday that the riverside bridge will close for six weeks of repairs starting next month. Engineers found corrosion in two of the steel supports during a routine inspection in the spring, and the council voted to fund the work from the emergency maintenance budget. Buses will follow a detour through the market district, adding about ten minutes to the morning commute, and the ferry will run an extra crossing every hour during peak times. Local shop owners said they were worried about losing customers, but the mayor promised a weekend festival on the waterfront once the bridge reopens. Cyclists and pedestrians can still use the temporary walkway on the north side, which will stay open from six in the morning until midnight.";
const MH_VOCAB="apple window silver garden quickly yellow seven harbor mountain paper engine river orange purple gentle winter summer basket candle forest lantern marble ocean pepper rocket saddle thunder velvet wagon zebra anchor biscuit copper desert feather glacier hammer island jungle kettle ladder meadow needle orchard pillow quartz ribbon shadow tunnel umbrella violin whistle".split(' ');
if(typeof module!=='undefined')module.exports={MH,MH_DOC,MH_VOCAB};
