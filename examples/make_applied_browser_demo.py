"""Generate the self-contained in-browser "decision cockpit" demo.

Trains a DQN on NonstationaryInventory, exports its weights to JSON, and writes a single
self-contained HTML file (docs/demo/inventory.html) that simulates the environment in plain
JavaScript and races three policies on the *same* demand stream:

    * fixed base-stock  (grey)  - the textbook rule; one order-up-to level, blind to demand
    * adaptive tracking (green) - orders up to a smoothed estimate of recent demand
    * learned policy    (blue)  - a DQN trained only from reward

You watch the cumulative-profit lines diverge and see the honest ordering emerge. No server,
no CDN, no dependencies; it runs entirely in the browser and works on GitHub Pages.

Run: python examples/make_applied_browser_demo.py [--device auto]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from decisionrl import baselines as B
from decisionrl.algorithms import DQN
from decisionrl.envs import NonstationaryInventory
from decisionrl.serving import export_json
from decisionrl.utils import Logger, set_seed

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>decisionrl - decision cockpit</title>
<style>
 body{font-family:system-ui,Segoe UI,sans-serif;background:#0b1020;color:#e2e8f0;text-align:center;margin:0;padding:24px}
 h1{font-size:20px;font-weight:800;margin:0 0 4px}.sub{color:#94a3b8;font-size:13px;margin:0 auto 14px;max-width:660px;line-height:1.5}
 canvas{background:#0f172a;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.4);max-width:100%}
 .row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-top:14px;font-variant-numeric:tabular-nums}
 .card{background:#111827;border-radius:10px;padding:10px 16px;min-width:150px}
 .lbl{font-size:12px;color:#94a3b8}.big{font-size:22px;font-weight:800}
 .bs{color:#94a3b8}.tr{color:#22c55e}.rl{color:#3b82f6}
 .ctrl{margin-top:14px}
 button{margin:4px;padding:8px 14px;border:0;border-radius:8px;background:#1e293b;color:#e2e8f0;font-weight:600;cursor:pointer}
 button.on{background:#2563eb;color:#fff}
 .verdict{margin-top:12px;font-size:14px;color:#cbd5e1;min-height:20px}
</style></head><body>
<h1>decisionrl &mdash; decision cockpit</h1>
<div class="sub">Three inventory policies face the <b>same</b> drifting demand: the textbook
<span class="bs">fixed base-stock</span>, an <span class="tr">adaptive rule</span> that tracks
recent demand, and a <span class="rl">learned DQN</span>. Cumulative profit, live, in your browser.</div>
<canvas id="c" width="720" height="320"></canvas>
<div class="row">
 <div class="card"><div class="lbl bs">fixed base-stock</div><div class="big bs" id="bsp">0</div></div>
 <div class="card"><div class="lbl rl">learned (DQN)</div><div class="big rl" id="rlp">0</div></div>
 <div class="card"><div class="lbl tr">adaptive tracking</div><div class="big tr" id="trp">0</div></div>
 <div class="card"><div class="lbl">demand regime</div><div class="big" id="reg">-</div></div>
</div>
<div class="ctrl">
 <button id="s0" class="on" onclick="setScenario(0)">regime shifts</button>
 <button id="s1" onclick="setScenario(1)">fast switching</button>
 <button id="s2" onclick="setScenario(2)">demand spike</button>
 <button onclick="reset()">restart</button>
</div>
<div class="verdict" id="verdict"></div>
<script>
const POLICY = __POLICY__;
const P = __PARAMS__;
const SCEN = [
 {name:"regime shifts", low:P.low, high:P.high, switchP:P.switchP},
 {name:"fast switching", low:P.low, high:P.high, switchP:0.20},
 {name:"demand spike", low:P.low, high:P.high*1.5, switchP:P.switchP},
];
let sc = 0;
const ACT = POLICY.activation === 'relu' ? (v)=>(v>0?v:0) : Math.tanh;
function act(obs){
  let x = obs.slice(); const L = POLICY.layers;
  for(let i=0;i<L.length;i++){
    const w=L[i].w,b=L[i].b,y=new Array(w.length);
    for(let o=0;o<w.length;o++){let s=b[o];const r=w[o];for(let j=0;j<r.length;j++)s+=r[j]*x[j];y[o]=s;}
    if(i<L.length-1)for(let o=0;o<y.length;o++)y[o]=ACT(y[o]);
    x=y;
  }
  let bi=0;for(let i=1;i<x.length;i++)if(x[i]>x[bi])bi=i;return bi;
}
function clip(v){return Math.max(0,Math.min(P.maxOrder,v));}
function pois(l){let Lp=Math.exp(-l),k=0,p=1;do{k++;p*=Math.random();}while(p>Lp);return k-1;}
let bs,rl,tr,high,ewma,bsP,rlP,trP,hist,t;
function reset(){bs=8;rl=8;tr=8;high=Math.random()<.5;ewma=high?SCEN[sc].high:SCEN[sc].low;
  bsP=0;rlP=0;trP=0;hist=[];t=0;document.getElementById('verdict').textContent="";}
function setScenario(i){sc=i;for(let k=0;k<3;k++)document.getElementById('s'+k).className=(k===i?"on":"");reset();}
function stepReward(inv0,order,demand){
  const after=Math.min(inv0+order,P.maxInv), sales=Math.min(after,demand), lost=demand-sales, nxt=after-sales;
  return [nxt, P.price*sales-P.unitCost*order-P.holding*nxt-P.stockout*lost];
}
const cv=document.getElementById('c'),cx=cv.getContext('2d');
function line(series,color){
  const n=hist.length; if(n<2)return;
  let lo=Infinity,hi=-Infinity;
  for(const h of hist){lo=Math.min(lo,h.bsP,h.rlP,h.trP);hi=Math.max(hi,h.bsP,h.rlP,h.trP);}
  const pad=20,H=cv.height,W=cv.width,rng=(hi-lo)||1;
  cx.strokeStyle=color;cx.lineWidth=2;cx.beginPath();
  for(let i=0;i<n;i++){const x=i/(n-1)*W,y=H-pad-(hist[i][series]-lo)/rng*(H-2*pad);
    if(i===0)cx.moveTo(x,y);else cx.lineTo(x,y);}
  cx.stroke();
}
function draw(){
  cx.clearRect(0,0,cv.width,cv.height);
  for(let i=0;i<hist.length;i++){if(hist[i].high){cx.fillStyle="rgba(239,68,68,.05)";
    cx.fillRect(i/Math.max(1,hist.length-1)*cv.width,0,cv.width/240+1,cv.height);}}
  line('bsP','#94a3b8'); line('rlP','#3b82f6'); line('trP','#22c55e');
}
function tick(){
  const s=SCEN[sc], mu=high?s.high:s.low, demand=pois(mu);
  const bsOrder=clip(Math.round(P.bestS-bs));
  const rlOrder=act([rl/P.maxInv, Math.min(ewma/P.maxOrder,1)]);
  const trOrder=clip(Math.round(ewma+P.bestSafety-tr));
  let r; [bs,r]=stepReward(bs,bsOrder,demand); bsP+=r;
  [rl,r]=stepReward(rl,rlOrder,demand); rlP+=r;
  [tr,r]=stepReward(tr,trOrder,demand); trP+=r;
  ewma=0.5*ewma+0.5*demand;
  if(Math.random()<s.switchP) high=!high;
  hist.push({bsP,rlP,trP,high}); if(hist.length>240)hist.shift();
  document.getElementById('bsp').textContent=bsP.toFixed(0);
  document.getElementById('rlp').textContent=rlP.toFixed(0);
  document.getElementById('trp').textContent=trP.toFixed(0);
  const reg=document.getElementById('reg');reg.textContent=high?"HIGH":"low";reg.style.color=high?"#ef4444":"#3b82f6";
  if(t>60){const best=Math.max(bsP,rlP,trP);
    const who=best===trP?["adaptive tracking","tr"]:best===rlP?["learned DQN","rl"]:["fixed base-stock","bs"];
    document.getElementById('verdict').innerHTML='leading: <b class="'+who[1]+'">'+who[0]+'</b>';}
  draw(); t++; if(t%600===0)reset();
}
reset(); setInterval(tick,60);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(0)
    print("Training DQN on NonstationaryInventory ...")
    agent = DQN(NonstationaryInventory(), learning_rate=5e-4, buffer_size=50_000,
                learning_starts=1000, target_update_interval=500, seed=0,
                logger=Logger(verbose=0), device=args.device)
    agent.learn(100_000)

    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    export_json(agent, tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        policy = json.load(f)
    os.unlink(tmp_path)

    env = NonstationaryInventory()
    best_s, _ = B.best_base_stock(NonstationaryInventory, seed=1)
    best_safety, _ = B.best_tracking_base_stock(NonstationaryInventory, seed=1)
    params = {"maxInv": env.max_inventory, "maxOrder": env.max_order,
              "low": env.demand_low, "high": env.demand_high, "switchP": env.switch_prob,
              "price": env.price, "unitCost": env.unit_cost, "holding": env.holding_cost,
              "stockout": env.stockout_penalty, "bestS": best_s, "bestSafety": best_safety}

    html = HTML.replace("__POLICY__", json.dumps(policy)).replace("__PARAMS__", json.dumps(params))
    out = os.path.join(DEMO_DIR, "inventory.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} (best base-stock S={best_s:.0f}, tracking safety={best_safety:.0f})")


if __name__ == "__main__":
    main()
