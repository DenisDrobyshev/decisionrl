import os

import pytest


def test_plot_dashboard_from_history_logger(tmp_path):
    pytest.importorskip("plotly")
    from reinforce.utils import HistoryLogger, plot_dashboard

    log = HistoryLogger()
    for step, val in [(10, 1.0), (20, 2.0), (30, 1.5)]:
        log.record("rollout/ep_return_mean", val)
        log.record("train/loss", 1.0 / step)
        log.dump(step)

    path = str(tmp_path / "dash.html")
    out = plot_dashboard(log, path=path, title="test")
    assert out == path and os.path.exists(path)
    assert os.path.getsize(path) > 1000  # a real interactive HTML document


def test_plot_dashboard_from_csv(tmp_path):
    pytest.importorskip("plotly")
    from reinforce.utils import Logger, plot_dashboard

    csv_path = str(tmp_path / "log.csv")
    log = Logger(csv_path=csv_path, verbose=0)
    for step in (10, 20, 30):
        log.record("metric", float(step))
        log.dump(step)

    out = plot_dashboard(csv_path, path=str(tmp_path / "d.html"))
    assert os.path.exists(out)


def test_plot_dashboard_empty_raises(tmp_path):
    pytest.importorskip("plotly")
    from reinforce.utils import plot_dashboard

    with pytest.raises(ValueError):
        plot_dashboard({}, path=str(tmp_path / "e.html"))
