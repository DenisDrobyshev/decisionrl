# Contributing to decisionrl

Thanks for your interest in improving `decisionrl`! Contributions of all kinds are
welcome — bug reports, new algorithms, environments, docs and tests.

## Development setup

```bash
git clone https://github.com/DenisDrobyshev/decisionrl.git
cd decisionrl
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a PR

```bash
ruff check .              # lint (must pass)
pytest -m "not slow"      # fast unit tests (seconds)
pytest                    # full suite incl. learning tests (a few minutes)
```

## Guidelines

- **Correctness first.** New algorithms must include at least one *learning* test
  (mark it `@pytest.mark.slow`) proving the agent beats a random policy on a small
  task, plus unit tests for any new components.
- **Keep the core dependency-light.** The core may only import `numpy` and
  `torch`. Anything else (Gymnasium, TensorBoard, ...) must be an optional extra
  guarded by a lazy import.
- **Respect the API.** Every agent implements `predict` / `learn` / `save` /
  `load` and accepts a `seed=`.
- **Handle `terminated` vs `truncated` correctly.** Bootstrap on truncation, stop
  on termination. This is the single most common source of silent RL bugs.
- **Match the surrounding style.** Type hints, short docstrings that cite the
  relevant paper, and `from __future__ import annotations` at the top.

## Adding an algorithm

1. Add `src/decisionrl/algorithms/<name>.py`.
2. Reuse existing components (`buffers`, `networks`, `exploration`, the
   `OnPolicyAgent` / `OffPolicyContinuousAgent` bases) where possible.
3. Export it from `src/decisionrl/algorithms/__init__.py` and the top-level
   `src/decisionrl/__init__.py`.
4. Add tests and an example.
5. Update the algorithm table in `README.md`.
