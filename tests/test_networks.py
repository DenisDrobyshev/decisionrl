import numpy as np
import torch

from decisionrl.networks import (
    CategoricalActor,
    ContinuousQ,
    DeterministicActor,
    DuelingQNetwork,
    GaussianActor,
    NoisyLinear,
    QNetwork,
    RainbowNetwork,
    SquashedGaussianActor,
    VNetwork,
    build_mlp,
)


def test_build_mlp_shape():
    net = build_mlp(4, 2, hidden_sizes=(8, 8))
    out = net(torch.zeros(5, 4))
    assert out.shape == (5, 2)


def test_qnetwork_and_dueling_shapes():
    for cls in (QNetwork, DuelingQNetwork):
        net = cls(4, 3, hidden_sizes=(16, 16))
        out = net(torch.zeros(7, 4))
        assert out.shape == (7, 3)


def test_categorical_actor():
    actor = CategoricalActor(4, 3)
    dist = actor(torch.zeros(6, 4))
    a = dist.sample()
    assert a.shape == (6,)
    assert dist.log_prob(a).shape == (6,)
    probs = dist.probs
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-5)


def test_gaussian_actor():
    actor = GaussianActor(4, 2)
    dist = actor(torch.zeros(6, 4))
    a = dist.sample()
    assert a.shape == (6, 2)
    assert dist.log_prob(a).sum(-1).shape == (6,)


def test_squashed_gaussian_bounds_and_logprob():
    low = np.array([-2.0, -2.0], dtype=np.float32)
    high = np.array([2.0, 2.0], dtype=np.float32)
    actor = SquashedGaussianActor(4, 2, low, high, hidden_sizes=(32, 32))
    action, log_prob, det = actor.sample(torch.zeros(10, 4))
    assert action.shape == (10, 2)
    assert log_prob.shape == (10, 1)
    assert torch.isfinite(log_prob).all()
    assert (action >= -2.0 - 1e-4).all() and (action <= 2.0 + 1e-4).all()
    assert (det >= -2.0 - 1e-4).all() and (det <= 2.0 + 1e-4).all()


def test_deterministic_actor_bounds():
    low = np.array([-1.0], dtype=np.float32)
    high = np.array([3.0], dtype=np.float32)
    actor = DeterministicActor(4, 1, low, high, hidden_sizes=(16, 16))
    a = actor(torch.randn(20, 4) * 5)
    assert a.shape == (20, 1)
    assert (a >= -1.0 - 1e-4).all() and (a <= 3.0 + 1e-4).all()


def test_noisy_linear_train_vs_eval():
    layer = NoisyLinear(8, 4, sigma0=0.5)
    x = torch.zeros(1, 8)
    # in eval mode: deterministic (mean weights); repeated calls identical
    layer.eval()
    assert torch.allclose(layer(x), layer(x))
    # in train mode with fresh noise, output changes
    layer.train()
    layer.reset_noise()
    out1 = layer(x)
    layer.reset_noise()
    out2 = layer(x)
    assert not torch.allclose(out1, out2)


def test_rainbow_network_shape_and_reset_noise():
    net = RainbowNetwork(6, n_actions=3, n_atoms=11, hidden_sizes=(16,))
    logits = net(torch.zeros(5, 6))
    assert logits.shape == (5, 3, 11)
    net.reset_noise()  # should not raise


def test_value_and_continuous_q_shapes():
    v = VNetwork(4)
    assert v(torch.zeros(8, 4)).shape == (8,)
    q = ContinuousQ(4, 2)
    assert q(torch.zeros(8, 4), torch.zeros(8, 2)).shape == (8,)
