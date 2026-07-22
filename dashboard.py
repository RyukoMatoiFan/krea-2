"""Minimal live training dashboard (a tiny TensorBoard) for any run here.

Every trainer writes ``<output_dir>/metrics.jsonl`` (one JSON per log step with
``step``/``loss``/``lr``/``s_per_step``/``peak_gb``, optional ``val_loss`` and ``total``)
and the t2i / edit / multi-ref trainers also write decoded ``<output_dir>/samples/*.png``.
A ``samples/prompts.json`` ({idx: text}) captions each preview.

Curated T2I previews are saved as a single 4-column CONTACT SHEET per step
(``stepNNNNNN_dashboard.png``). The dashboard slices that sheet back into one card per
prompt server-side (``/tile``) so each preview is its own hoverable / clickable image --
no trainer change or restart needed. Per-tile files (``..idxK..``) are used directly.

EDIT previews are instead one full-width ROW per example ([source | model edit | target]).
A ``samples/layout.json`` (``{"mode":"edit"}``) switches ``/tile`` to row-slicing so each
example becomes its own card, captioned by its instruction from ``samples/prompts.json``.

Optional ``--base-dir`` (a directory of per-prompt ``idxK.png`` base-model tiles, e.g.
``<output_dir>/base_previews``) enables a click-to-toggle BASE vs current preview per
prompt for easy comparison.

Features: live stat chips + progress/ETA, loss(+val)/VRAM/speed charts, a sample grid with
prev/next step pagination, click-to-toggle base vs preview, and a click-to-zoom lightbox.

  python dashboard.py --run runs/my-run --total 150000 --port 8090 \
    --base-dir runs/my-run/base_previews
"""
import argparse
import html
import json
import math
import os
import re
import threading
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse, parse_qs

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_SHEET_COLS = 4  # T2I preview contact sheets are built 4-wide (train_t2i_full_cached._sample)
_IMG_CACHE = {}  # {path: (mtime, PIL.Image)} -- decode a contact sheet once per version, not per poll
# Composed edit "cards" (one whole image per example: prompt header + 2x2 labelled tiles on black).
# Built once per (sheet, mtime) for all examples so a gallery load decodes the big sheet a single time.
_CARD_CACHE = {}  # {"_key": (path, mtime), k: png_bytes}
_CARD_LOCK = threading.Lock()
_FONT_CACHE = {}  # {size: PIL.ImageFont}

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · krea2</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 :root{
   --bg:#0a0c10; --panel:#11151c; --panel2:#161b24; --line:#222a36;
   --ink:#e7ecf3; --mut:#8a94a6; --accent:#6ea8fe; --good:#56d364; --warn:#e3b341; --bad:#f2756b;
 }
 *{box-sizing:border-box}
 body{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
   margin:0;background:radial-gradient(1200px 600px at 70% -10%,#121a28 0%,var(--bg) 60%);
   color:var(--ink);-webkit-font-smoothing:antialiased}
 header{position:sticky;top:0;z-index:10;backdrop-filter:blur(8px);
   background:rgba(10,12,16,.78);border-bottom:1px solid var(--line);padding:12px 22px}
 .htop{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
 .dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 10px var(--good);
   animation:pulse 2s infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
 h1{font-size:15px;font-weight:650;margin:0;letter-spacing:.2px}
 h1 .sub{color:var(--mut);font-weight:400;margin-left:6px}
 .chips{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
 .chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
   padding:5px 12px;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
 .chip b{color:var(--accent);font-weight:650} .chip.v b{color:var(--warn)}
 .chip .k{color:var(--mut);margin-right:5px}
 .barwrap{margin-top:10px}
 .bar{background:var(--panel2);border:1px solid var(--line);border-radius:8px;height:10px;overflow:hidden}
 .fill{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--good));
   transition:width .6s cubic-bezier(.4,0,.2,1)}
 .prog{color:var(--mut);font-size:12px;margin-top:5px;font-variant-numeric:tabular-nums}
 .wrap{max-width:1500px;margin:0 auto;padding:20px 22px 40px}
 .charts{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
 @media(max-width:900px){.charts{grid-template-columns:1fr}}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px 10px}
 .card h3{margin:0 0 8px;font-size:12px;font-weight:600;color:var(--mut);
   text-transform:uppercase;letter-spacing:.6px}
 canvas{max-height:230px}
 .ebars{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:5px 18px;margin:2px 2px 6px}
 .ebar{display:flex;align-items:center;gap:8px;font-size:11.5px}
 .ebar .nm{flex:0 0 46%;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .ebar .tr{flex:1;height:9px;background:#141922;border-radius:5px;overflow:hidden}
 .ebar .fl{height:100%;border-radius:5px}
 .ebar .pc{flex:0 0 34px;text-align:right;color:var(--ink);font-variant-numeric:tabular-nums}
 .sech{display:flex;align-items:center;gap:12px;margin:30px 2px 14px;flex-wrap:wrap}
 .sech h2{font-size:15px;margin:0;font-weight:650}
 .sech .muted{color:var(--mut);font-size:12px}
 .pager{display:flex;align-items:center;gap:8px;margin-left:auto}
 .pager button{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
   width:34px;height:30px;border-radius:9px;cursor:pointer;font-size:15px;line-height:1;transition:.12s}
 .pager button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
 .pager button:disabled{opacity:.35;cursor:default}
 .pager .lbl{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums;min-width:150px;text-align:center}
 .hint{font-size:11px;color:var(--mut);margin:0 2px 12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}
 figure.scard{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;
   overflow:hidden;transition:transform .15s ease,border-color .15s ease}
 figure.scard:hover{transform:translateY(-3px);border-color:#33405a}
 .imgwrap{position:relative;aspect-ratio:1/1;background:#05070b;overflow:hidden;cursor:pointer}
 .imgwrap img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s ease}
 figure.scard:hover .imgwrap img{transform:scale(1.05)}
 .badge{position:absolute;top:8px;left:8px;background:rgba(8,10,14,.82);border:1px solid var(--line);
   color:var(--accent);font-size:11px;font-weight:650;padding:2px 8px;border-radius:999px;letter-spacing:.3px}
 .badge.base{color:var(--warn);border-color:#5a4a1f}
 .zoom{position:absolute;top:7px;right:7px;width:26px;height:26px;border-radius:8px;
   background:rgba(8,10,14,.82);border:1px solid var(--line);color:var(--ink);cursor:zoom-in;
   display:flex;align-items:center;justify-content:center;font-size:13px;opacity:0;transition:.12s}
 figure.scard:hover .zoom{opacity:1} .zoom:hover{border-color:var(--accent);color:var(--accent)}
 figcaption{padding:9px 11px 11px;font-size:12px;line-height:1.45;color:var(--mut);
   display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;min-height:38px}
 .empty{color:var(--mut);font-size:13px;padding:30px 0}
 /* edit runs: 4 tiles as a 2x2 grid per card (source | ground truth / base | trained) */
 figure.scard.wide{grid-column:1/-1}
 .tiles4{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#05070b}
 .tile{position:relative;aspect-ratio:1/1;overflow:hidden;cursor:zoom-in}
 .tile img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s ease}
 .tile:hover img{transform:scale(1.06)}
 .tlabel{position:absolute;top:5px;left:5px;background:rgba(8,10,14,.82);border:1px solid var(--line);
   font-size:10px;font-weight:650;padding:1px 6px;border-radius:999px;letter-spacing:.2px}
 .tile.missing{display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px}
 /* composed edit card: one whole image (prompt + 2x2 tiles + labels, baked on black) */
 .cimg{cursor:zoom-in;background:#05070b;line-height:0}
 .cimg img{width:100%;display:block;transition:transform .2s ease}
 figure.scard:hover .cimg img{transform:scale(1.015)}
 /* lightbox: preview + base side by side */
 #lb{position:fixed;inset:0;z-index:50;background:rgba(5,7,11,.93);display:none;
   align-items:center;justify-content:center;flex-direction:column;gap:14px;padding:24px;cursor:zoom-out}
 #lb.on{display:flex}
 #lbrow{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start;max-width:88vw}
 .lbfig{margin:0;display:flex;flex-direction:column;gap:6px;align-items:center;min-width:0}
 .lbfig img{max-width:42vw;max-height:38vh;width:100%;border-radius:12px;border:1px solid var(--line);
   object-fit:contain;background:#05070b;cursor:default}
 .lbfig.solo{grid-column:1/-1}
 .lbfig.solo img{max-width:94vw;max-height:90vh;width:auto}
 .lbfig figcaption .tg{font-weight:700;letter-spacing:.5px;font-size:12px}
 #lbcap{color:var(--ink);font-size:13px;max-width:90vw;text-align:center;line-height:1.5}
 #lbhint{color:var(--mut);font-size:12px}
</style></head><body>
<header>
 <div class="htop">
  <span class="dot"></span>
  <h1>__TITLE__ <span class="sub" id="sub">live training</span></h1>
  <div class="chips" id="chips"></div>
 </div>
 <div class="barwrap"><div class="bar"><div class="fill" id="fill"></div></div>
  <div class="prog" id="prog">waiting for metrics…</div></div>
</header>
<div class="wrap">
 <div class="charts">
  <div class="card"><h3>loss &amp; val</h3><canvas id="loss"></canvas></div>
  <div class="card"><h3>grad norm (DiT / TE)</h3><canvas id="gnorm"></canvas></div>
  <div class="card"><h3>peak VRAM (GB)</h3><canvas id="vram"></canvas></div>
  <div class="card"><h3>sec / step</h3><canvas id="speed"></canvas></div>
 </div>
 <div class="sech" id="evalhdr" style="display:none"><h2>Held-out edit eval</h2><span class="muted" id="ecount"></span></div>
 <div class="charts" id="evalwrap" style="display:none">
  <div class="card" style="grid-column:1/-1"><h3>edit-success rate vs step — <b>overall</b> + family groups (judged on the held-out set)</h3><canvas id="evalc"></canvas></div>
 </div>
 <div class="ebars" id="ebars"></div>
 <div class="sech"><h2>Samples</h2><span class="muted" id="scount"></span>
  <div class="pager">
   <button id="first" title="first">⏮</button>
   <button id="prev" title="previous step">◀</button>
   <span class="lbl" id="plabel">—</span>
   <button id="next" title="next step">▶</button>
   <button id="last" title="latest">⏭</button>
  </div>
 </div>
 <div class="hint">Click a tile to <b>zoom</b> the full set · ◀ ▶ (or arrow keys) page through preview steps. <b>Edit runs:</b> each card is one example — <b>1 source</b> · <b>2 ground truth</b> · <b>3 base</b> (untrained model) · <b>4 trained</b> (current step).</div>
 <div class="grid" id="gallery"><div class="empty">no previews yet</div></div>
</div>
<div id="lb">
 <div id="lbrow"></div>
 <div id="lbcap"></div>
 <div id="lbhint">Esc or click the backdrop to close</div>
</div>
<script>
const TOTAL=__TOTAL__;
const esc=s=>(s+'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=s=>{s=Math.max(0,Math.round(s));const h=(s/3600|0),mn=((s%3600)/60|0),se=s%60;
 return (h?h+'h ':'')+(mn<10&&h?'0':'')+mn+'m '+(se<10?'0':'')+se+'s';};
const gridCfg=(extra={})=>({type:'line',data:{datasets:[]},options:{animation:false,responsive:true,
 interaction:{intersect:false,mode:'index'},
 scales:{x:{type:'linear',title:{display:true,text:'step',color:'#8a94a6'},grid:{color:'#1b2230'},ticks:{color:'#8a94a6'}},
  y:{grid:{color:'#1b2230'},ticks:{color:'#8a94a6'}}},
 plugins:{legend:{display:!!extra.legend,labels:{color:'#e7ecf3',boxWidth:10,usePointStyle:true}}}}});
const ds=(label,color,fill=false)=>({label,data:[],borderColor:color,
 backgroundColor:fill?color+'22':color,pointRadius:0,borderWidth:2,tension:.25,fill});
const loss=new Chart(document.getElementById('loss'),gridCfg({legend:true}));
loss.data.datasets=[
 {label:'loss (raw)',data:[],borderColor:'#6ea8fe44',backgroundColor:'#6ea8fe44',pointRadius:0,borderWidth:1,tension:.2},
 {label:'loss (ema)',data:[],borderColor:'#6ea8fe',backgroundColor:'#6ea8fe',pointRadius:0,borderWidth:2,tension:.25},
 {label:'val',data:[],borderColor:'#e3b341',backgroundColor:'#e3b341',pointRadius:0,borderWidth:2.5,tension:.25},
];
const gnorm=new Chart(document.getElementById('gnorm'),gridCfg({legend:true}));
gnorm.data.datasets=[ds('DiT','#6ea8fe'),ds('TE','#d2a8ff')];
const vram=new Chart(document.getElementById('vram'),gridCfg());vram.data.datasets=[ds('vram','#f2756b',true)];
const speed=new Chart(document.getElementById('speed'),gridCfg());speed.data.datasets=[ds('s/step','#56d364')];
const evalCfg=gridCfg({legend:true});evalCfg.options.scales.y.min=0;evalCfg.options.scales.y.max=1;
evalCfg.options.scales.y.ticks.callback=v=>Math.round(v*100)+'%';
const evalc=new Chart(document.getElementById('evalc'),evalCfg);
const EG=[['overall','#ffffff',3],['global/style','#56d364',2],['structural','#6ea8fe',2],['person','#d2a8ff',2],['text','#f2756b',2]];
evalc.data.datasets=EG.map(([l,c,w])=>({label:l,data:[],borderColor:c,backgroundColor:c,pointRadius:2,borderWidth:w,tension:.2,spanGaps:true}));
const chip=(k,v,cls='')=>`<span class="chip ${cls}"><span class="k">${k}</span><b>${v}</b></span>`;

let STEPS=[], curStep=null, pinned=false, data={items:[]}, lastKey='', CARDS=[];
const G=id=>document.getElementById(id);

function renderGallery(){
 const items=data.items||[];
 G('scount').textContent=items.length?`step ${(data.step||0).toLocaleString()} · ${items.length} previews`:'';
 const i=STEPS.indexOf(data.step);
 G('plabel').textContent=STEPS.length?`step ${(data.step||0).toLocaleString()}  (${i+1}/${STEPS.length})`:'—';
 G('first').disabled=G('prev').disabled=(i<=0);
 G('last').disabled=G('next').disabled=(i<0||i>=STEPS.length-1);
 // Re-render the gallery DOM ONLY when content changes -> images never re-fetch on
 // unchanged polls (no flicker).
 const key=(data.step)+'·'+items.map(it=>(it.card||it.src)+'|'+(it.base||'')).join('~');
 if(key===lastKey) return;
 lastKey=key;
 CARDS=items;
 G('gallery').innerHTML=items.length? items.map((it,ci)=>{
   if(it.card) return cardEdit(it,ci);
   const hasBase=!!it.base;
   return `<figure class="scard" data-prev="${esc(it.src)}" data-base="${esc(it.base||'')}" data-prompt="${esc(it.prompt||'')}" data-idx="${it.idx}">`
    +`<div class="imgwrap" onclick="imgClick(this)" ondblclick="imgDbl(this)">`
    +`<span class="badge">#${it.idx} · PREVIEW</span>`
    +`<img loading="lazy" src="${esc(it.src)}"></div>`
    +`<figcaption>${it.prompt?esc(it.prompt):'<span style=opacity:.5>no prompt</span>'}`
    +`${hasBase?'':' <span style="color:#f2756b">· no base yet</span>'}</figcaption></figure>`;
  }).join('') : '<div class="empty">no previews yet</div>';
}
const TLC=['#9aa0a8','#78c878','#e3b341','#6ea8fe'];  // source / GT / base / trained
function cardEdit(it,ci){
 // One whole composed image per example (prompt + 2x2 labelled tiles, baked server-side) so the
 // entire set can be grabbed / saved / shared as a single picture. Click zooms it full-size.
 return `<figure class="scard"><div class="cimg" onclick="zoomImg(this)" title="click to enlarge (right-click the big image to save)">`
   +`<img loading="lazy" src="${esc(it.card)}"></div></figure>`;
}
let clickT=null;  // single click = zoom; double click = toggle base (delay disambiguates)
function imgClick(w){if(clickT)return;clickT=setTimeout(()=>{clickT=null;zoom(w.closest('.scard'));},220);}
function imgDbl(w){if(clickT){clearTimeout(clickT);clickT=null;}tog(w);}
function tog(wrap){
 const f=wrap.closest('.scard'),img=wrap.querySelector('img'),b=wrap.querySelector('.badge');
 const base=f.dataset.base; if(!base) return;
 if(img.dataset.mode==='base'){img.src=f.dataset.prev;img.dataset.mode='prev';b.textContent=`#${f.dataset.idx} · PREVIEW`;b.classList.remove('base');}
 else{img.src=base;img.dataset.mode='base';b.textContent=`#${f.dataset.idx} · BASE`;b.classList.add('base');}
}
function zoom(f){
 G('lbrow').innerHTML=`<figure class="lbfig"><img class="lbimg" src="${esc(f.dataset.prev)}"><figcaption><span class="tg" style="color:#6ea8fe">PREVIEW</span></figcaption></figure>`
  +(f.dataset.base?`<figure class="lbfig"><img class="lbimg" src="${esc(f.dataset.base)}"><figcaption><span class="tg" style="color:#e3b341">BASE</span></figcaption></figure>`:'');
 G('lbcap').innerHTML=`#${f.dataset.idx} · step ${(data.step||0).toLocaleString()} — ${esc(f.dataset.prompt||'')}`
   +(f.dataset.base?'':' <span style="color:#f2756b">(no base sample)</span>');
 G('lb').classList.add('on');
}
function zoomImg(el){
 const src=el.querySelector('img').src;
 G('lbrow').innerHTML=`<figure class="lbfig solo"><img class="lbimg" src="${esc(src)}"></figure>`;
 G('lbcap').innerHTML='';
 G('lb').classList.add('on');
}
G('lb').onclick=e=>{if(!e.target.classList.contains('lbimg'))G('lb').classList.remove('on');};
document.addEventListener('keydown',e=>{
 if(e.key==='Escape')G('lb').classList.remove('on');
 else if(!G('lb').classList.contains('on')){if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);}});

async function loadSamples(step){
 const q=step!=null?`?step=${step}`:'';
 try{data=await (await fetch('api/samples'+q)).json();}catch(e){return;}
 STEPS=data.steps||[]; curStep=data.step;
 renderGallery();
}
function go(d){const i=STEPS.indexOf(curStep);let j=i+d;if(j<0)j=0;if(j>STEPS.length-1)j=STEPS.length-1;
 if(STEPS[j]!=null){pinned=(j!==STEPS.length-1);loadSamples(STEPS[j]);}}
G('prev').onclick=()=>go(-1); G('next').onclick=()=>go(1);
G('first').onclick=()=>{if(STEPS.length){pinned=true;loadSamples(STEPS[0]);}};
G('last').onclick=()=>{pinned=false;loadSamples();};

async function pollMetrics(){
 try{
  const m=await (await fetch('api/metrics')).json();
  if(!m.length)return;
  const lr_=m.filter(r=>r.loss!=null);
  loss.data.datasets[0].data=lr_.map(r=>({x:r.step,y:r.loss}));
  let e=null;const ema=[];for(const r of lr_){e=e==null?r.loss:e*0.9+r.loss*0.1;ema.push({x:r.step,y:e});}
  loss.data.datasets[1].data=ema;
  loss.data.datasets[2].data=m.filter(r=>r.val_loss!=null).map(r=>({x:r.step,y:r.val_loss}));
  gnorm.data.datasets[0].data=m.filter(r=>r.dit_grad_norm!=null).map(r=>({x:r.step,y:r.dit_grad_norm}));
  gnorm.data.datasets[1].data=m.filter(r=>r.te_grad_norm!=null).map(r=>({x:r.step,y:r.te_grad_norm}));
  vram.data.datasets[0].data=m.map(r=>({x:r.step,y:r.peak_gb}));
  speed.data.datasets[0].data=m.map(r=>({x:r.step,y:r.s_per_step}));
  loss.update();gnorm.update();vram.update();speed.update();
  const last=m[m.length-1],step=last.step||0,total=TOTAL||last.total||0;
  const vals=m.filter(r=>r.val_loss!=null),sps=last.s_per_step||0,eta=total&&sps?fmt((total-step)*sps):'—';
  const teg=m.filter(r=>r.te_grad_norm!=null);
  G('chips').innerHTML=chip('step',step.toLocaleString()+(total?' / '+total.toLocaleString():''))
   +chip('loss',(last.loss||0).toFixed(4))+(vals.length?chip('val',vals[vals.length-1].val_loss.toFixed(4),'v'):'')
   +(teg.length?chip('TE |∇|',teg[teg.length-1].te_grad_norm.toFixed(3),'v'):'')
   +chip('s/it',sps.toFixed(2))+chip('VRAM',(last.peak_gb||0).toFixed(1)+'G')
   +chip('lr',(last.lr||0).toExponential(1))+chip('ETA',eta);
  if(total){G('fill').style.width=Math.min(100,100*step/total)+'%';
   G('prog').textContent=`${(100*step/total).toFixed(1)}% · step ${step.toLocaleString()} of ${total.toLocaleString()} · ETA ${eta}`;}
  else G('prog').textContent=`step ${step.toLocaleString()} · pass --total for a progress bar`;
 }catch(e){G('sub').textContent='metrics unavailable';}
}
const evcol=v=>{const h=Math.round(120*Math.max(0,Math.min(1,v)));return `hsl(${h},58%,46%)`;};
async function pollEval(){
 let ev;try{ev=await (await fetch('api/eval')).json();}catch(e){return;}
 if(!ev||!ev.length){return;}
 G('evalhdr').style.display='';G('evalwrap').style.display='';
 EG.forEach((g,i)=>{const key=g[0];
  evalc.data.datasets[i].data=ev.map(r=>({x:r.step,y:key==='overall'?r.overall:(r.groups?r.groups[key]:null)}))
   .filter(p=>p.y!=null);});
 evalc.update();
 const last=ev[ev.length-1],pf=last.per_family||{};
 const fams=Object.keys(pf).sort((a,b)=>pf[a]-pf[b]);  // worst first = what needs attention
 G('ecount').textContent=`step ${(last.step||0).toLocaleString()} · overall ${Math.round(last.overall*100)}%`
  +(last.overall_partial!=null?` (incl. partial ${Math.round(last.overall_partial*100)}%)`:'')+` · K=${last.n_per_family}/family`;
 G('ebars').innerHTML=fams.map(f=>{const v=pf[f];return `<div class="ebar" title="${esc(f)} — ${Math.round(v*100)}%">`
  +`<span class="nm">${esc(f)}</span><span class="tr"><span class="fl" style="width:${Math.round(v*100)}%;background:${evcol(v)}"></span></span>`
  +`<span class="pc">${Math.round(v*100)}%</span></div>`;}).join('');
}
async function tick(){await pollMetrics(); await pollEval(); if(!pinned) await loadSamples();}
tick();setInterval(tick,2500);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
  run = "."
  samples_dir = None
  base_dir = None
  total = 0

  def log_message(self, *a):
    pass

  def _send(self, code, body, ctype="application/json"):
    if isinstance(body, str):
      body = body.encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Cache-Control", "no-store")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    u = urlparse(self.path)
    path, qs = u.path, parse_qs(u.query)
    if path == "/":
      title = html.escape(os.path.basename(os.path.abspath(self.run)))
      page = _PAGE.replace("__TITLE__", title).replace("__TOTAL__", str(int(self.total)))
      self._send(200, page, "text/html; charset=utf-8")
    elif path == "/api/metrics":
      self._send(200, json.dumps(self._metrics()))
    elif path == "/api/samples":
      step = int(qs["step"][0]) if "step" in qs else None
      self._send(200, json.dumps(self._samples(step)))
    elif path == "/api/eval":
      self._send(200, json.dumps(self._eval()))
    elif path.startswith("/samples/"):
      self._serve_image(path[len("/samples/"):])
    elif path.startswith("/tile/"):
      self._serve_tile(path[len("/tile/"):])
    elif path.startswith("/cell/"):
      self._serve_cell(path[len("/cell/"):])
    elif path.startswith("/basecell/"):
      self._serve_basecell(path[len("/basecell/"):])
    elif path.startswith("/card/"):
      self._serve_card(path[len("/card/"):])
    elif path.startswith("/baseimg/"):
      self._serve_base(path[len("/baseimg/"):])
    else:
      self._send(404, "{}")

  def _metrics(self):
    p = os.path.join(self.run, "metrics.jsonl")
    rows = []
    if os.path.exists(p):
      for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
          try:
            rows.append(json.loads(line))
          except json.JSONDecodeError:
            pass
    return rows

  @staticmethod
  def _fam_group(name):
    """Bucket an edit-family name into one of 4 super-groups for the eval chart lines."""
    s = (name or "").lower()
    if "text" in s or "font" in s or "translate" in s:
      return "text"
    person_kw = ("person", "expression", "accessor", "pose", "age /", "clothing", "caricature",
                 "funko", "lego", "simpson", "pixar", "anime", "sticker", "line-art", "western comic")
    if any(k in s for k in person_kw):
      return "person"
    struct_kw = ("remove", "replace one object", "object category", "add a new object",
                 "add new scene", "size/shape", "outpaint", "relocate", "zoom", "attribute")
    if any(k in s for k in struct_kw):
      return "structural"
    return "global/style"

  def _eval(self):
    """Parse eval_scores.jsonl and attach per-record super-group means for the dashboard chart."""
    p = os.path.join(self.run, "eval_scores.jsonl")
    out = []
    if os.path.exists(p):
      for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
          continue
        try:
          r = json.loads(line)
        except json.JSONDecodeError:
          continue
        groups = {"global/style": [], "structural": [], "person": [], "text": []}
        for fam, v in (r.get("per_family") or {}).items():
          groups[self._fam_group(fam)].append(v)
        r["groups"] = {g: (round(sum(vs) / len(vs), 4) if vs else None) for g, vs in groups.items()}
        out.append(r)
    out.sort(key=lambda r: r.get("step", 0))
    return out

  def _samples_root(self):
    return self.samples_dir or os.path.join(self.run, "samples")

  def _base_root(self):
    return self.base_dir or os.path.join(self.run, "base_previews")

  def _prompts(self):
    p = os.path.join(self._samples_root(), "prompts.json")
    if os.path.isfile(p):
      try:
        return {str(k): v for k, v in json.load(open(p, encoding="utf-8")).items()}
      except Exception:
        return {}
    return {}

  def _layout(self):
    """Optional ``samples/layout.json``; ``{"mode":"edit"}`` makes /tile slice by row."""
    p = os.path.join(self._samples_root(), "layout.json")
    if os.path.isfile(p):
      try:
        return json.load(open(p, encoding="utf-8"))
      except Exception:
        return {}
    return {}

  @staticmethod
  def _stepof(f):
    m = re.search(r"step(\d+)", f)
    return int(m.group(1)) if m else -1

  def _base_src(self, k):
    p = os.path.join(self._base_root(), f"idx{k}.png")
    return f"baseimg/{k}" if os.path.isfile(p) else None

  def _samples(self, step=None):
    """One card per prompt for a chosen preview step (default latest). ``steps`` lists every
    available preview step for pagination; each item carries its BASE counterpart if present."""
    d = self._samples_root()
    if not os.path.isdir(d):
      return {"step": 0, "steps": [], "items": []}
    prompts = self._prompts()
    imgs = [f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS)]
    steps = sorted({self._stepof(f) for f in imgs if self._stepof(f) >= 0})
    if not steps:
      return {"step": 0, "steps": [], "items": []}
    target = step if (step in steps) else steps[-1]
    cur = [f for f in imgs if self._stepof(f) == target]
    tiled = sorted(f for f in cur if re.search(r"idx(\d+)", f))
    items = []
    if tiled:
      for f in tiled:
        k = int(re.search(r"idx(\d+)", f).group(1))
        items.append({"src": f"samples/{quote(f)}", "base": self._base_src(k),
                      "prompt": prompts.get(str(k), ""), "idx": k})
    else:
      sheet = sorted(cur)[0]
      n = len(prompts) or 1
      if n <= 1:
        items.append({"src": f"samples/{quote(sheet)}", "base": self._base_src(0),
                      "prompt": prompts.get("0", ""), "idx": 0})
      else:
        edit = self._layout().get("mode") == "edit"
        sq = quote(sheet)
        for k in range(n):
          item = {"src": f"tile/{sq}/{k}", "base": self._base_src(k),
                  "prompt": prompts.get(str(k), ""), "idx": k}
          if edit:
            # One whole composed image per example (prompt on top + 2x2 labelled tiles on black:
            # source, ground truth, base-model output, trained output) -- grabbable as a single
            # picture. Server-composed at /card so the big step sheet is decoded once for all 26.
            item["card"] = f"card/{sq}/{k}"
          items.append(item)
    return {"step": target, "steps": steps, "items": items}

  def _safe(self, root, name):
    name = unquote(name)
    if "/" in name or "\\" in name or ".." in name:
      return None
    p = os.path.join(root, name)
    return p if os.path.isfile(p) else None

  def _serve_file(self, p):
    ext = os.path.splitext(p)[1].lower()
    with open(p, "rb") as f:
      self._send(200, f.read(), "image/png" if ext == ".png" else "image/jpeg")

  def _serve_image(self, name):
    p = self._safe(self._samples_root(), name)
    self._serve_file(p) if p else self._send(404, "{}")

  def _serve_base(self, k):
    try:
      k = int(k)
    except ValueError:
      return self._send(404, "{}")
    p = os.path.join(self._base_root(), f"idx{k}.png")
    self._serve_file(p) if os.path.isfile(p) else self._send(404, "{}")

  def _sheet(self, p):
    """Decode a contact sheet once per (path, mtime) and cache the PIL image, so a tall
    edit sheet (many rows, tens of MB) is not re-decoded on every poll / tile request."""
    from PIL import Image
    mt = os.path.getmtime(p)
    hit = _IMG_CACHE.get(p)
    if hit and hit[0] == mt:
      return hit[1]
    img = Image.open(p).convert("RGB")
    _IMG_CACHE.clear()  # only ever need the most-recent sheet resident
    _IMG_CACHE[p] = (mt, img)
    return img

  def _serve_tile(self, rest):
    """/tile/<file>/<k> -> crop preview k out of a contact sheet.

    T2I sheets are a 4-column grid; EDIT sheets (``layout.json`` mode ``edit``) are one
    full-width row per example ([source | model edit | target])."""
    try:
      name, k = rest.rsplit("/", 1)
      k = int(k)
    except ValueError:
      return self._send(404, "{}")
    p = self._safe(self._samples_root(), name)
    if not p:
      return self._send(404, "{}")
    try:
      img = self._sheet(p)
      n = max(1, len(self._prompts()))
      W, H = img.size
      if self._layout().get("mode") == "edit":
        th = H // n                       # one full-width row per example
        box = (0, k * th, W, k * th + th)
      else:
        cols = _SHEET_COLS
        rows = math.ceil(n / cols)
        tw, th = W // cols, H // rows
        col, row = k % cols, k // cols
        box = (col * tw, row * th, col * tw + tw, row * th + th)
      buf = BytesIO()
      img.crop(box).save(buf, "PNG")
      self._send(200, buf.getvalue(), "image/png")
    except Exception:
      self._send(404, "{}")

  def _crop_cell(self, img, k, n, c, cols=3):
    """Crop cell (row k, column c of ``cols``) from an edit contact sheet / base row."""
    W, H = img.size
    th = H // n
    cw = W // cols
    box = (c * cw, k * th, c * cw + cw, k * th + th)
    buf = BytesIO()
    img.crop(box).save(buf, "PNG")
    return buf.getvalue()

  def _serve_cell(self, rest):
    """/cell/<sheet>/<k>/<c> -> one cell (row k, column c of 3) of an EDIT sheet:
    column 0 = source, 1 = trained model output, 2 = target (ground truth)."""
    try:
      name, k, c = rest.rsplit("/", 2)
      k, c = int(k), int(c)
    except ValueError:
      return self._send(404, "{}")
    p = self._safe(self._samples_root(), name)
    if not p or c not in (0, 1, 2):
      return self._send(404, "{}")
    try:
      n = max(1, len(self._prompts()))
      self._send(200, self._crop_cell(self._sheet(p), k, n, c), "image/png")
    except Exception:
      self._send(404, "{}")

  def _serve_basecell(self, rest):
    """/basecell/<k>/<c> -> one cell of base_previews/idx{k}.png (a single
    [source | base-output | target] row): column 1 is the untrained-model output."""
    try:
      k, c = rest.rsplit("/", 1)
      k, c = int(k), int(c)
    except ValueError:
      return self._send(404, "{}")
    p = os.path.join(self._base_root(), f"idx{k}.png")
    if not os.path.isfile(p) or c not in (0, 1, 2):
      return self._send(404, "{}")
    try:
      self._send(200, self._crop_cell(self._sheet(p), 0, 1, c), "image/png")
    except Exception:
      self._send(404, "{}")

  @staticmethod
  def _card_font(size):
    f = _FONT_CACHE.get(size)
    if f is None:
      from PIL import ImageFont
      for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                   "DejaVuSans.ttf"):
        try:
          f = ImageFont.truetype(cand, size)
          break
        except Exception:
          continue
      if f is None:
        f = ImageFont.load_default()
      _FONT_CACHE[size] = f
    return f

  def _compose_card(self, sheet, base, k, n, prompt, step):
    """One example -> a single image: wrapped prompt on top, then a 2x2 of labelled tiles
    (source, ground truth, base-model output, trained output) on black. Returns a PIL image."""
    from PIL import Image, ImageDraw
    T, PAD, GAP, LBL_H = 512, 20, 8, 30
    W, H = sheet.size
    th, cw = H // n, W // 3
    def cell(img, c, cols):
      w = img.size[0] // cols
      row = k if cols == 3 else 0
      rh = img.size[1] // (n if cols == 3 else 1)
      return img.crop((c * w, row * rh, c * w + w, row * rh + rh)).resize((T, T))
    src = cell(sheet, 0, 3)
    trained = cell(sheet, 1, 3)
    tgt = cell(sheet, 2, 3)
    # base_previews/idx{k}.png is a SINGLE-row [source|base|target] image, not the
    # tall 26-row sheet -- crop its middle third at full height (not via cell(),
    # which would slice a 1/n-tall sliver and stretch it into noise).
    def bcell(img, c):
      w = img.size[0] // 3
      return img.crop((c * w, 0, c * w + w, img.size[1])).resize((T, T))
    base_out = bcell(base, 1) if base is not None else None
    tiles = [
      (src, "1 · SOURCE", (154, 160, 168)),
      (tgt, "2 · GROUND TRUTH", (120, 200, 120)),
      (base_out, "3 · BASE (untrained)", (227, 179, 65)),
      (trained, f"4 · TRAINED (step {step:,})".replace(",", " "), (110, 168, 254)),
    ]
    f_lbl, f_pr = self._card_font(19), self._card_font(19)
    colW = PAD * 2 + T * 2 + GAP
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    def wrap(txt, fnt, maxw):
      out, cur = [], ""
      for wd in txt.split():
        t = (cur + " " + wd).strip()
        if tmp.textlength(t, font=fnt) <= maxw:
          cur = t
        else:
          out.append(cur)
          cur = wd
      if cur:
        out.append(cur)
      return out
    plines = wrap(f"#{k} · {prompt}", f_pr, colW - PAD * 2)[:3]
    PR_H = 6 + len(plines) * 24 + 10
    rowH = LBL_H + T
    canvas = Image.new("RGB", (colW, PAD + PR_H + rowH * 2 + GAP + PAD), (16, 17, 20))
    d = ImageDraw.Draw(canvas)
    for i, ln in enumerate(plines):
      d.text((PAD, PAD + i * 24), ln, font=f_pr, fill=(210, 212, 218))
    y0 = PAD + PR_H
    for i, (tile, lab, col) in enumerate(tiles):
      x = PAD + (i % 2) * (T + GAP)
      y = y0 + (i // 2) * (rowH + GAP)
      d.text((x + 2, y), lab, font=f_lbl, fill=col)
      if tile is not None:
        canvas.paste(tile, (x, y + LBL_H))
      else:
        d.rectangle([x, y + LBL_H, x + T, y + LBL_H + T], fill=(20, 22, 26))
        d.text((x + T // 2 - 12, y + LBL_H + T // 2 - 8), "n/a", font=f_lbl, fill=(138, 148, 166))
      d.rectangle([x, y + LBL_H, x + T, y + LBL_H + T], outline=(60, 62, 68), width=1)
    return canvas

  def _ensure_cards(self, name, p):
    """Build + cache the composed card PNGs for every example of sheet ``p`` (one decode of the
    big step sheet for the whole gallery). No-op if already built for this (path, mtime)."""
    from PIL import Image
    key = (p, os.path.getmtime(p))
    with _CARD_LOCK:
      if _CARD_CACHE.get("_key") == key:
        return
      prompts = self._prompts()
      n = max(1, len(prompts))
      step = self._stepof(name)
      broot = self._base_root()
      sheet = Image.open(p).convert("RGB")
      cards = {"_key": key}
      for k in range(n):
        bp = os.path.join(broot, f"idx{k}.png")
        base = Image.open(bp).convert("RGB") if os.path.isfile(bp) else None
        buf = BytesIO()
        self._compose_card(sheet, base, k, n, prompts.get(str(k), ""), step).save(buf, "PNG")
        cards[k] = buf.getvalue()
        if base is not None:
          base.close()
      sheet.close()
      _CARD_CACHE.clear()
      _CARD_CACHE.update(cards)

  def _serve_card(self, rest):
    """/card/<sheet>/<k> -> one composed image (prompt + 2x2 labelled tiles) for example k."""
    try:
      name, k = rest.rsplit("/", 1)
      k = int(k)
    except ValueError:
      return self._send(404, "{}")
    p = self._safe(self._samples_root(), name)
    if not p:
      return self._send(404, "{}")
    try:
      self._ensure_cards(name, p)
      data = _CARD_CACHE.get(k)
      self._send(200, data, "image/png") if data else self._send(404, "{}")
    except Exception:
      self._send(404, "{}")


def main():
  ap = argparse.ArgumentParser(description="Minimal live training dashboard.")
  ap.add_argument("--run", required=True, help="run output_dir (holds metrics.jsonl + samples/)")
  ap.add_argument("--samples-dir", default=None, help="override the samples dir")
  ap.add_argument("--base-dir", default=None, help="per-prompt base tiles (idxK.png) for base/preview toggle")
  ap.add_argument("--total", type=int, default=0, help="total steps (progress/ETA; auto from metrics 'total')")
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--host", default="0.0.0.0")
  args = ap.parse_args()
  Handler.run = args.run
  Handler.samples_dir = args.samples_dir
  Handler.base_dir = args.base_dir
  Handler.total = args.total
  srv = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"[dashboard] {args.run} -> http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
