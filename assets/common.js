// shared helpers
const $=s=>document.querySelector(s);
function fmt(x,d=2){if(!isFinite(x))return'∞';const a=Math.abs(x);
 const units=[[1e24,'Y'],[1e21,'Z'],[1e18,'E'],[1e15,'P'],[1e12,'T'],[1e9,'G'],[1e6,'M'],[1e3,'k']];
 for(const[u,s]of units)if(a>=u)return(x/u).toFixed(d)+s;return x.toFixed(d)}
function sci(x,d=2){if(x===0)return'0';const e=Math.floor(Math.log10(Math.abs(x)));return (x/10**e).toFixed(d)+'×10'+String(e).replace(/-?\d+/,m=>[...m].map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]||'⁻').join(''))}
function setupCanvas(cv,aspect){ // HiDPI responsive canvas; returns ctx + logical size
 const dpr=window.devicePixelRatio||1;const w=cv.clientWidth;const h=Math.round(w*aspect);
 cv.style.height=h+'px';cv.width=w*dpr;cv.height=h*dpr;const ctx=cv.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w,h}}
function onResize(fn){let t;window.addEventListener('resize',()=>{clearTimeout(t);t=setTimeout(fn,80)});}
function bindRanges(ids,fn){ids.forEach(id=>{const el=document.getElementById(id);el.addEventListener('input',fn)});fn()}
function val(id){const e=document.getElementById(id);return parseFloat(e.value)}
// "What to notice" is revealed after the first interaction (STYLE_GUIDE §7)
(function(){function rev(){document.querySelectorAll('.notice.hidden').forEach(n=>n.classList.remove('hidden'))}
 let armed=false;window.addEventListener('load',()=>setTimeout(()=>{armed=true},50));
 ['input','change','pointerdown'].forEach(ev=>document.addEventListener(ev,e=>{if(armed&&e.isTrusted)rev()},{capture:true}));})();
