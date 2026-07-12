"""A lightweight live training dashboard (Flask + Plotly, no heavy deps).

Reads a metrics CSV written by :class:`~reinforce.utils.Logger` and serves a
self-contained web page that auto-refreshes one live chart per metric
(reward, loss, ...). Launch with ``reinforce dashboard run.csv`` and open the
printed URL while training writes to the same CSV.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

__all__ = ["read_metrics", "create_app", "run_dashboard"]


def read_metrics(csv_path: str) -> Dict[str, List[float]]:
    """Parse a Logger CSV into ``{column: [values]}`` (non-numeric cells -> NaN)."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    columns = list(rows[0].keys())
    data: Dict[str, List[float]] = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            v = row.get(c, "")
            try:
                data[c].append(float(v))
            except (TypeError, ValueError):
                data[c].append(float("nan"))
    return data


def _payload(csv_path: str) -> dict:
    metrics = read_metrics(csv_path)
    x = metrics.pop("step", None)
    return {"x": x, "series": metrics}


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>reinforce dashboard</title><script>__PLOTLYJS__</script>
<style>body{font-family:system-ui,sans-serif;margin:16px;background:#0b1020;color:#e2e8f0}
h1{font-size:18px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
.card{background:#111827;border-radius:10px;padding:8px}</style></head>
<body><h1>reinforce &mdash; live training dashboard</h1><div id="grid"></div>
<script>
const INTERVAL=__INTERVAL__;
async function refresh(){
  const r=await fetch('data'); const d=await r.json(); const grid=document.getElementById('grid');
  const x=d.x||[]; const names=Object.keys(d.series||{});
  for(const name of names){
    let el=document.getElementById('p_'+name);
    if(!el){el=document.createElement('div');el.className='card';el.id='p_'+name;grid.appendChild(el);}
    const y=d.series[name];
    Plotly.react(el,[{x:x.length?x:y.map((_,i)=>i),y:y,mode:'lines',line:{color:'#60a5fa'}}],
      {title:name,paper_bgcolor:'#111827',plot_bgcolor:'#111827',font:{color:'#e2e8f0'},
       margin:{l:48,r:16,t:36,b:32},height:280},{displayModeBar:false});
  }
}
refresh(); setInterval(refresh,INTERVAL);
</script></body></html>"""


def create_app(csv_path: str, interval_ms: int = 2000):
    """Build the Flask app (importable/testable without starting a server)."""
    from flask import Flask, jsonify

    try:
        from plotly.offline import get_plotlyjs
        plotlyjs = get_plotlyjs()
    except Exception:  # pragma: no cover - plotly always present in dev
        plotlyjs = ""

    app = Flask("reinforce-dashboard")
    page = _PAGE.replace("__PLOTLYJS__", plotlyjs).replace("__INTERVAL__", str(int(interval_ms)))

    @app.route("/")
    def index():
        return page

    @app.route("/data")
    def data():
        return jsonify(_payload(csv_path))

    return app


def run_dashboard(csv_path: str, host: str = "127.0.0.1", port: int = 8050,
                  interval_ms: int = 2000) -> None:  # pragma: no cover - runs a server
    """Serve the live dashboard for ``csv_path`` (blocks)."""
    app = create_app(csv_path, interval_ms)
    print(f"reinforce dashboard: http://{host}:{port}  (reading {csv_path})")
    app.run(host=host, port=port)
