# Models

Place an exported policy here as `policy.onnx` (+ `policy.onnx.json`) before
building the Docker image, or mount one at runtime:

```bash
python -c "from decisionrl.algorithms import PPO; from decisionrl.envs import CartPole; \
from decisionrl.serving import export_onnx; \
export_onnx(PPO(CartPole(), seed=0).learn(50_000), 'deploy/models/policy.onnx')"

docker build -f deploy/Dockerfile -t decisionrl-serve .
docker run -p 8000:8000 decisionrl-serve
```
