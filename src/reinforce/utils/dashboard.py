"""Interactive training dashboards with Plotly (optional dependency).

Turn logged metrics into a self-contained interactive HTML dashboard (one panel
per metric). Accepts a :class:`~reinforce.utils.logger.HistoryLogger`, a metrics
dict ``{name: [(step, value), ...]}``, or a path to a CSV written by
:class:`~reinforce.utils.logger.Logger`.
"""

from __future__ import annotations

import csv
import math
from typing import Dict, List, Tuple

__all__ = ["plot_dashboard"]


def _to_series(history) -> Dict[str, Tuple[List[float], List[float]]]:
    # HistoryLogger (or anything exposing a .history mapping of name -> [(step, val)])
    if hasattr(history, "history"):
        history = history.history
    if isinstance(history, dict):
        out = {}
        for key, pts in history.items():
            if not pts:
                continue
            xs, ys = zip(*pts)
            out[key] = (list(xs), list(ys))
        return out
    if isinstance(history, str):  # CSV path (Logger output)
        with open(history, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        keys = [k for k in rows[0] if k != "step"]
        steps = [float(r["step"]) for r in rows]
        out = {}
        for k in keys:
            xs, ys = [], []
            for s, r in zip(steps, rows):
                if r.get(k) not in (None, ""):
                    xs.append(s)
                    ys.append(float(r[k]))
            if ys:
                out[k] = (xs, ys)
        return out
    raise TypeError("history must be a HistoryLogger, a metrics dict, or a CSV path")


def plot_dashboard(history, path: str = "dashboard.html",
                   title: str = "reinforce training dashboard") -> str:
    """Render an interactive HTML dashboard of all logged metrics. Returns ``path``.

    Requires ``pip install plotly``.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotly is required for plot_dashboard: pip install plotly") from exc

    series = _to_series(history)
    if not series:
        raise ValueError("no metrics to plot")
    keys = sorted(series)
    cols = 2 if len(keys) > 1 else 1
    rows = math.ceil(len(keys) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=keys)
    for i, key in enumerate(keys):
        xs, ys = series[key]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=key),
                      row=i // cols + 1, col=i % cols + 1)
    fig.update_layout(title=title, showlegend=False, height=300 * rows, template="plotly_white")
    fig.update_xaxes(title_text="step")
    fig.write_html(path)
    return path
