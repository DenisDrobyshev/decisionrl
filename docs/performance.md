# Performance and throughput

This page is about computational throughput (how fast the code runs), which is separate
from the learning-quality numbers in [Benchmarks](benchmarks.md). Reproduce it with:

```bash
python examples/benchmark_throughput.py --device cpu --steps 5000
```

The script reports, per algorithm:

- **env steps/sec** - pure environment stepping under a random policy, the Python
  simulation ceiling with no learning involved
- **train steps/sec** - the full training loop: act, step the environment, store the
  transition, run the gradient update
- **predict calls/sec** - deterministic inference throughput of the learned policy

## Measured throughput

Single CPU thread (`torch.set_num_threads(1)`), PyTorch 2.9, small default networks:

| Algorithm | Environment | env steps/s | train steps/s | predict/s |
|---|---|---:|---:|---:|
| PPO | CartPole | 94,000 | 1,500 | 9,600 |
| DQN | CartPole | 95,000 | 940 | 23,800 |
| SAC | Pendulum | 26,000 | 100 | 4,700 |

The gap between env steps/sec and train steps/sec is the cost of the gradient update.
It is largest for SAC, which updates two critics, an actor, and the temperature on every
step, so its per-step work is several backward passes rather than one.

## Choosing a device

For the small networks these environments use, a single CPU thread is usually the
fastest option for training. The per-step work is tiny, so throughput is dominated by
Python-side environment stepping and per-call overhead rather than matrix math, and a
GPU spends most of its time launching kernels instead of computing. A GPU becomes worth
it when the networks or batch sizes are large enough that the matrix math dominates the
launch overhead, for example wide multi-layer policies, image observations, or large
off-policy batch sizes.

Select the device explicitly:

```python
from decisionrl.algorithms import SAC

agent = SAC(env, device="cpu")   # or "cuda", or "auto" (default)
```

## Practical levers

- **Threads.** More CPU threads rarely helps small models and can hurt because of
  synchronisation overhead. Set `torch.set_num_threads(1)` for the tightest per-step
  latency, and raise it only for wide networks or large batches.
- **Batch size and update frequency.** For off-policy agents, `train_freq` and
  `gradient_steps` trade sample efficiency against wall-clock throughput: fewer, larger
  updates run faster per environment step.
- **`torch.compile`.** `decisionrl.utils.torch_utils.maybe_compile` wraps a module with
  `torch.compile` when a compiler backend is available and falls back to the eager
  module otherwise, so it is safe to call unconditionally. It helps most for larger
  networks; for the tiny defaults here the compile overhead is not repaid.

  ```python
  from decisionrl.utils.torch_utils import maybe_compile

  agent.policy = maybe_compile(agent.policy, enabled=True)
  ```

- **Inference.** Exported ONNX policies (see [Serving](serving.md)) run inference with
  onnxruntime alone and do not carry the PyTorch training stack, which lowers both
  latency and memory footprint for deployment.
