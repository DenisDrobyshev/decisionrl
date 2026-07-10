# Serve an exported reinforce policy over HTTP.
#
# Build (from the repo root, with a policy.onnx + policy.onnx.json present here):
#   docker build -f deploy/Dockerfile -t reinforce-serve .
# Run:
#   docker run -p 8000:8000 -e REINFORCE_MODEL=/models/policy.onnx reinforce-serve
#
# The serving image needs neither PyTorch nor the training extras — only
# onnxruntime + FastAPI — so it stays small.
FROM python:3.12-slim

WORKDIR /app

# Install just the runtime deps (no torch) plus the library itself.
RUN pip install --no-cache-dir onnxruntime fastapi "uvicorn[standard]" numpy
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Bake in a default model location; override REINFORCE_MODEL at runtime.
COPY deploy/models /models
ENV REINFORCE_MODEL=/models/policy.onnx

EXPOSE 8000
CMD ["uvicorn", "reinforce.serving.server:app", "--host", "0.0.0.0", "--port", "8000"]
