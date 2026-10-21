// Byte-level BPE, a line-for-line port of chapters/ch02/exercises/ex_2_3_1_solution.py
const BPE=(function(){
const PATS={gpt:/'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/gu,
 nodigit:/'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/gu};
const enc=new TextEncoder(),dec=new TextDecoder('utf-8',{fatal:false});
function pretokenize(text,mode){if(mode==='none')return text?[text]:[];return text.match(PATS[mode])||[]}
function bytes(s){return Array.from(enc.encode(s))}
function mergeWord(w,a,b,id){const out=[];let i=0;while(i<w.length){if(i+1<w.length&&w[i]===a&&w[i+1]===b){out.push(id);i+=2}else{out.push(w[i]);i+=1}}return out}
function train(text,nMerges,mode){
 let words=new Map();for(const ch of pretokenize(text,mode)){const k=bytes(ch).join(',');words.set(k,(words.get(k)||0)+1)}
 let W=[...words].map(([k,f])=>[k.split(',').map(Number),f]);
 const merges=[];const vocab=[];for(let i=0;i<256;i++)vocab.push([i]);
 for(let id=256;id<256+nMerges;id++){const counts=new Map();
  for(const[w,f]of W)for(let i=0;i+1<w.length;i++){const k=w[i]*65536+w[i+1];counts.set(k,(counts.get(k)||0)+f)}
  if(!counts.size)break;let best=-1,bc=-1;for(const[k,c]of counts){if(c>bc||(c===bc&&k<best)){bc=c;best=k}}
  const a=Math.floor(best/65536),b=best%65536;merges.push({a,b,id,count:bc});vocab.push(vocab[a].concat(vocab[b]));
  const nw=new Map();for(const[w,f]of W){const m=mergeWord(w,a,b,id);const k=m.join(',');const e=nw.get(k);if(e)e[1]+=f;else nw.set(k,[m,f])}W=[...nw.values()];}
 return{merges,vocab}}
function encode(text,model,m,mode){const rank=new Map();model.merges.slice(0,m).forEach((x,i)=>rank.set(x.a*65536+x.b,i));const ids=[];
 for(const ch of pretokenize(text,mode)){let w=bytes(ch);while(w.length>=2){let best=-1,br=Infinity;for(let i=0;i+1<w.length;i++){const r=rank.get(w[i]*65536+w[i+1]);if(r!==undefined&&r<br){br=r;best=i}}
  if(best<0)break;const mg=model.merges[br];w=mergeWord(w,mg.a,mg.b,mg.id)}ids.push(...w)}return ids}
function show(model,id){return dec.decode(new Uint8Array(model.vocab[id]))}
return{pretokenize,train,encode,show,bytes};})();
const BPE_CORPUS="the quick brown fox jumps over the lazy dog. ".repeat(20)+"the theory of the thermal throughput is thorough. ".repeat(15)+"tokenization turns text into tokens; tokens turn into embeddings. ".repeat(10)+"Größe and naïve café — 東京 — emoji 🙂 survive byte-level BPE. ".repeat(3);
if(typeof module!=='undefined')module.exports={BPE,BPE_CORPUS};
