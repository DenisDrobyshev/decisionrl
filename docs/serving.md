# Serving trained policies

`reinforce.serving` turns a trained agent into a portable artifact and a tiny
inference service. The serving runtime needs only `onnxruntime` + FastAPI — no
PyTorch — so deployment images stay small.

Install the extra:

```bash
pip install "reinforce-rl[serve]"
```

## Export

```python
from reinforce.algorithms import PPO
from reinforce.envs import CartPole
from reinforce.serving import export_onnx, export_torchscript, OnnxPolicy

agent = PPO(CartPole(), seed=0).learn(50_000)
export_onnx(agent, "policy.onnx")          # writes policy.onnx + policy.onnx.json
export_torchscript(agent, "policy.pt")     # TorchScript alternative

policy = OnnxPolicy("policy.onnx")         # inference with onnxruntime only
action = policy.predict(obs)
```

The exporter freezes the **deterministic** policy (argmax for discrete agents; the
squashed/clamped mean for continuous ones). Supported agents: PPO, A2C, GRPO, SAC,
DDPG, TD3 and DQN.

## Serve over HTTP

```bash
REINFORCE_MODEL=policy.onnx uvicorn reinforce.serving.server:app --port 8000
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/info` | policy metadata (obs dim, action type, bounds) |
| POST | `/predict` | `{"observation": [...]}` → `{"action": ...}` |

```python
from reinforce.serving import create_app       # FastAPI app for the model
app = create_app("policy.onnx")
```

## Docker

`deploy/Dockerfile` builds a slim image (onnxruntime + FastAPI, no torch):

```bash
docker build -f deploy/Dockerfile -t reinforce-serve .
docker run -p 8000:8000 -e REINFORCE_MODEL=/models/policy.onnx reinforce-serve
```
