"""Tests for the live training dashboard (Flask app + CSV parsing)."""

import pytest

from decisionrl.dashboard import create_app, read_metrics
from decisionrl.utils import Logger


def test_read_metrics_parses_logger_csv(tmp_path):
    csv_path = str(tmp_path / "run.csv")
    logger = Logger(verbose=0, csv_path=csv_path)
    for step in range(1, 4):
        logger.record("rollout/ep_return", float(step * 10))
        logger.record("train/loss", 1.0 / step)
        logger.dump(step)

    metrics = read_metrics(csv_path)
    assert metrics["rollout/ep_return"] == [10.0, 20.0, 30.0]
    assert metrics["step"] == [1.0, 2.0, 3.0]


def test_read_metrics_missing_file_is_empty(tmp_path):
    assert read_metrics(str(tmp_path / "nope.csv")) == {}


def test_dashboard_app_serves_page_and_data(tmp_path):
    pytest.importorskip("flask")
    csv_path = str(tmp_path / "run.csv")
    logger = Logger(verbose=0, csv_path=csv_path)
    logger.record("reward", 5.0)
    logger.dump(1)

    client = create_app(csv_path, interval_ms=1000).test_client()

    index = client.get("/")
    assert index.status_code == 200 and b"decisionrl" in index.data

    data = client.get("/data").get_json()
    assert data["series"]["reward"] == [5.0]
    assert data["x"] == [1.0]
