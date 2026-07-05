import numpy as np
import torch
import torch.nn as nn

from reinforce.utils import (
    HistoryLogger,
    RunningMeanStd,
    explained_variance,
    hard_update,
    set_seed,
    soft_update,
)


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    ta = torch.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    tb = torch.rand(5)
    np.testing.assert_array_equal(a, b)
    assert torch.equal(ta, tb)


def test_running_mean_std_matches_numpy():
    rms = RunningMeanStd(shape=(3,))
    data = np.random.randn(1000, 3) * 2.0 + 1.0
    for i in range(0, 1000, 100):
        rms.update(data[i : i + 100])
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(rms.var, data.var(axis=0), atol=1e-4)


def test_running_mean_std_normalize():
    rms = RunningMeanStd(shape=())
    data = np.random.randn(5000) * 3 + 5
    rms.update(data)
    normed = rms.normalize(data)
    assert abs(normed.mean()) < 0.05
    assert abs(normed.std() - 1.0) < 0.05


def test_explained_variance():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y_true.copy(), y_true) == 1.0
    # predicting the mean -> ev == 0
    assert explained_variance(np.full(4, y_true.mean()), y_true) == 0.0


def test_history_logger_records_curve():
    log = HistoryLogger()
    for step, val in [(10, 1.0), (20, 2.0), (30, 3.0)]:
        log.record("rollout/ep_return_mean", val)
        log.dump(step)
    xs, ys = log.curve()
    assert xs == [10, 20, 30]
    assert ys == [1.0, 2.0, 3.0]
    # unknown key -> empty
    assert log.curve("does/not/exist") == ([], [])


def test_soft_and_hard_update():
    src = nn.Linear(3, 3)
    tgt = nn.Linear(3, 3)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
        for p in tgt.parameters():
            p.fill_(0.0)
    soft_update(src, tgt, tau=0.5)
    for p in tgt.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5))
    hard_update(src, tgt)
    for ps, pt in zip(src.parameters(), tgt.parameters()):
        assert torch.allclose(ps, pt)
