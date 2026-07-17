"""Generate a self-contained in-browser applied-RL demo.

Trains a DQN policy on NonstationaryInventory, exports its weights to JSON, and
writes a single self-contained HTML file (docs/demo/inventory.html) that simulates
the environment in plain JavaScript and races the **learned policy** against the
**best fixed base-stock** on the *same* demand stream — so you watch RL adapt to the
regime shifts the fixed rule can't. No server, no CDN; works on GitHub Pages.

Run: python examples/make_applied_browser_demo.py
"""

from __future__ import annotations

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
<title>decisionrl - RL vs base-stock on drifting demand</title>
<style>
 body{font-family:system-ui,Segoe UI,sans-serif;background:#0b1020;color:#e2e8f0;text-align:center;margin:0;padding:24px}
 h1{font-size:20px;font-weight:800;margin:0 0 4px}.sub{color:#94a3b8;font-size:14px;margin-bottom:16px}
 canvas{background:#0f172a;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.4)}
 .row{display:flex;gap:24px;justify-content:center;margin-top:14px;font-variant-numeric:tabular-nums}
 .card{background:#111827;border-radius:10px;padding:10px 18px;min-width:150px}
 .rl{color:#3b82f6}.bs{color:#94a3b8}.big{font-size:22px;font-weight:800}
 button{margin-top:14px;padding:8px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer}
</style></head><body>
<h1>decisionrl &mdash; RL vs the textbook formula on drifting demand</h1>
<div class="sub">A DQN policy (blue) and the best fixed base-stock rule (grey) face the <b>same</b> demand,
which switches between a low and a high regime. Watch RL track the regime the fixed rule can't. Runs entirely in your browser.</div>
<canvas id="c" width="640" height="300"></canvas>
<div class="row">
 <div class="card"><div class="rl">RL (DQN) profit</div><div class="big rl" id="rlp">0</div></div>
 <div class="card"><div>regime</div><div class="big" id="reg">-</div></div>
 <div class="card"><div class="bs">base-stock profit</div><div class="big bs" id="bsp">0</div></div>
</div>
<button onclick="reset()">restart</button>
<script>
const POLICY = __POLICY__;
const P = __PARAMS__;
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
function pois(l){let Lp=Math.exp(-l),k=0,p=1;do{k++;p*=Math.random();}while(p>Lp);return k-1;}
let rl,bs,high,ewma,rlP,bsP,hist,t;
function reset(){rl=8;bs=8;high=Math.random()<.5;ewma=high?P.high:P.low;rlP=0;bsP=0;hist=[];t=0;}
function stepReward(inv0,order,demand){
  const after=Math.min(inv0+order,P.maxInv), sales=Math.min(after,demand), lost=demand-sales, nxt=after-sales;
  const r=P.price*sales-P.unitCost*order-P.holding*nxt-P.stockout*lost;
  return [nxt,r];
}
const cv=document.getElementById('c'),cx=cv.getContext('2d');
function draw(){
  cx.clearRect(0,0,cv.width,cv.height);
  const base=cv.height-40, w=Math.max(1,cv.width/240);
  cx.strokeStyle="#1e293b";cx.beginPath();cx.moveTo(0,base);cx.lineTo(cv.width,base);cx.stroke();
  for(let i=0;i<hist.length;i++){
    const h=hist[i],x=i*w;
    cx.fillStyle=h.high?"rgba(239,68,68,.10)":"rgba(59,130,246,.06)";cx.fillRect(x,0,w+1,cv.height);
    cx.fillStyle="#3b82f6";cx.fillRect(x,base-h.rl*7,w,h.rl*7);
    cx.fillStyle="rgba(148,163,184,.7)";cx.fillRect(x+w*0.35,base-h.bs*7,w*0.5,h.bs*7);
  }
}
function tick(){
  const demand=pois(high?P.high:P.low);
  const rlOrder=act([rl/P.maxInv, Math.min(ewma/P.maxOrder,1)]);
  const bsOrder=Math.max(0,Math.min(P.maxOrder,Math.round(P.bestS-bs)));
  let r; [rl,r]=stepReward(rl,rlOrder,demand); rlP+=r;
  [bs,r]=stepReward(bs,bsOrder,demand); bsP+=r;
  ewma=0.5*ewma+0.5*demand;
  if(Math.random()<P.switchP) high=!high;
  hist.push({rl,bs,high}); if(hist.length>240)hist.shift();
  document.getElementById('rlp').textContent=rlP.toFixed(0);
  document.getElementById('bsp').textContent=bsP.toFixed(0);
  document.getElementById('reg').textContent=high?"HIGH":"low";
  document.getElementById('reg').style.color=high?"#ef4444":"#3b82f6";
  draw(); t++; if(t%400===0)reset();
}
reset(); setInterval(tick,60);
</script></body></html>"""


def main() -> None:
    set_seed(0)
    print("Training DQN on NonstationaryInventory ...")
    agent = DQN(NonstationaryInventory(), learning_rate=5e-4, buffer_size=50_000,
                learning_starts=1000, target_update_interval=500, seed=0, logger=Logger(verbose=0))
    agent.learn(100_000)

    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    export_json(agent, tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        policy = json.load(f)
    os.unlink(tmp_path)

    env = NonstationaryInventory()
    best_s, _ = B.best_base_stock(NonstationaryInventory, seed=1)
    params = {"maxInv": env.max_inventory, "maxOrder": env.max_order,
              "low": env.demand_low, "high": env.demand_high, "switchP": env.switch_prob,
              "price": env.price, "unitCost": env.unit_cost, "holding": env.holding_cost,
              "stockout": env.stockout_penalty, "bestS": best_s}

    html = HTML.replace("__POLICY__", json.dumps(policy)).replace("__PARAMS__", json.dumps(params))
    out = os.path.join(DEMO_DIR, "inventory.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} (best base-stock S={best_s:.0f})")


if __name__ == "__main__":
    main()
