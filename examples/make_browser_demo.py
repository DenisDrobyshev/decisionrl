"""Generate a self-contained in-browser demo: watch a trained agent play.

Trains PPO on CartPole, exports the policy weights to JSON, and writes a single
self-contained HTML file (docs/demo/cartpole.html) that runs the policy with a few
matmuls in plain JavaScript and animates CartPole on a canvas — no server, no
onnxruntime, no CDN. Works directly on GitHub Pages.

Run: python examples/make_browser_demo.py
"""

from __future__ import annotations

import json
import os

from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.serving import export_json
from reinforce.utils import Logger, set_seed
from reinforce.zoo import save_to_zoo

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>reinforce — watch a trained agent play</title>
<style>
 body{font-family:system-ui,Segoe UI,sans-serif;background:#0b1020;color:#e2e8f0;text-align:center;margin:0;padding:24px}
 h1{font-size:20px;font-weight:800}.sub{color:#94a3b8;font-size:14px;margin-bottom:16px}
 canvas{background:#f8fafc;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.4)}
 .stat{margin-top:12px;font-variant-numeric:tabular-nums}button{margin-top:14px;padding:8px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer}
</style></head><body>
<h1>reinforce &mdash; PPO plays CartPole</h1>
<div class="sub">A policy trained with <code>reinforce</code>, running entirely in your browser (no server).</div>
<canvas id="c" width="520" height="320"></canvas>
<div class="stat">episode steps: <b id="steps">0</b> &nbsp;|&nbsp; best: <b id="best">0</b></div>
<button onclick="reset()">restart</button>
<script>
const POLICY = __POLICY__;
// --- tiny MLP forward (deterministic argmax policy) ---
function act(obs){
  let x = obs.slice();
  const L = POLICY.layers;
  for(let i=0;i<L.length;i++){
    const {w,b} = L[i]; const y = new Array(w.length);
    for(let o=0;o<w.length;o++){let s=b[o];const row=w[o];for(let j=0;j<row.length;j++)s+=row[j]*x[j];y[o]=s;}
    if(i<L.length-1){for(let o=0;o<y.length;o++)y[o]=Math.tanh(y[o]);} // hidden activation
    x=y;
  }
  let best=0;for(let i=1;i<x.length;i++)if(x[i]>x[best])best=i;return best;
}
// --- CartPole dynamics (matches reinforce.envs.CartPole) ---
const g=9.8,mc=1.0,mp=0.1,l=0.5,fm=10.0,tau=0.02,tot=mc+mp,pml=mp*l;
const xthr=2.4,ththr=12*2*Math.PI/360;
let s,steps,best=0;
function reset(){s=[rand(),rand(),rand(),rand()];steps=0;document.getElementById('steps').textContent=0;}
function rand(){return (Math.random()-0.5)*0.1;}
function step(a){
  let [x,xd,th,thd]=s;const f=a===1?fm:-fm;const ct=Math.cos(th),st=Math.sin(th);
  const tmp=(f+pml*thd*thd*st)/tot;
  const tha=(g*st-ct*tmp)/(l*(4/3-mp*ct*ct/tot));const xa=tmp-pml*tha*ct/tot;
  x+=tau*xd;xd+=tau*xa;th+=tau*thd;thd+=tau*tha;s=[x,xd,th,thd];steps++;
  return (x<-xthr||x>xthr||th<-ththr||th>ththr||steps>=500);
}
const cx=document.getElementById('c').getContext('2d');
function draw(){
  cx.clearRect(0,0,520,320);const [x,,th]=s;const px=260+x*90,py=240;
  cx.strokeStyle='#94a3b8';cx.beginPath();cx.moveTo(20,py);cx.lineTo(500,py);cx.stroke();
  cx.fillStyle='#334155';cx.fillRect(px-30,py,60,18);
  const tx=px+Math.sin(th)*120,ty=py-Math.cos(th)*120;
  cx.strokeStyle='#2563eb';cx.lineWidth=8;cx.lineCap='round';cx.beginPath();cx.moveTo(px,py);cx.lineTo(tx,ty);cx.stroke();cx.lineWidth=1;
}
function loop(){
  const done=step(act(s));draw();
  document.getElementById('steps').textContent=steps;
  if(done){best=Math.max(best,steps);document.getElementById('best').textContent=best;reset();}
}
reset();setInterval(loop,20);
</script></body></html>"""


def main() -> None:
    set_seed(0)
    print("training PPO on CartPole ...")
    agent = PPO(CartPole(), seed=0, logger=Logger(verbose=0))
    agent.learn(120_000)
    weights = export_json(agent, os.path.join(DEMO_DIR, "cartpole_policy.json"))
    with open(weights, encoding="utf-8") as f:
        policy = f.read()
    out = os.path.join(DEMO_DIR, "cartpole.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__POLICY__", policy))
    print("wrote", out)
    save_to_zoo(agent, "cartpole-ppo")  # also publish the ONNX policy to the model zoo
    print("saved cartpole-ppo to the model zoo")


if __name__ == "__main__":
    main()
