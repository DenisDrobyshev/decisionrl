"""A tiny, dependency-free experiment logger.

Records scalar metrics, prints them as a readable table, and can mirror them to
a CSV file. TensorBoard is used automatically *iff* it is installed and a
``tensorboard_dir`` is given, but is never a hard dependency.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

__all__ = ["Logger"]


class Logger:
    def __init__(
        self,
        csv_path: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        self.verbose = verbose
        self._values: Dict[str, float] = {}
        self._csv_path = csv_path
        self._csv_keys: List[str] = []
        self._start_time = time.time()

        if csv_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        self._tb_writer = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(tensorboard_dir)
            except ImportError:  # pragma: no cover - optional dependency
                if verbose:
                    print("[reinforce] tensorboard not installed; skipping TB logging.")

    def record(self, key: str, value: float) -> None:
        """Stage a scalar value under ``key`` for the next :meth:`dump`."""
        self._values[key] = value

    def record_dict(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.record(k, v)

    def dump(self, step: int) -> None:
        """Flush all staged values to stdout / CSV / TensorBoard."""
        if not self._values:
            return

        if self._tb_writer is not None:
            for k, v in self._values.items():
                self._tb_writer.add_scalar(k, v, step)

        if self._csv_path is not None:
            self._write_csv(step)

        if self.verbose:
            self._print_table(step)

        self._values.clear()

    def _print_table(self, step: int) -> None:
        elapsed = time.time() - self._start_time
        rows = [("step", str(step)), ("elapsed_s", f"{elapsed:.1f}")]
        for k, v in self._values.items():
            rows.append((k, f"{v:.4g}" if isinstance(v, float) else str(v)))
        width = max(len(k) for k, _ in rows) + 2
        line = "-" * (width + 18)
        print(line)
        for k, v in rows:
            print(f"| {k:<{width}}| {v:<14}|")
        print(line)

    def _write_csv(self, step: int) -> None:
        row = {"step": step, **self._values}
        new_keys = [k for k in row if k not in self._csv_keys]
        if new_keys:
            self._csv_keys.extend(new_keys)
            # Rewrite header when new columns appear (simple + safe for small logs).
            existing: List[dict] = []
            if os.path.exists(self._csv_path):
                with open(self._csv_path, newline="") as f:
                    existing = list(csv.DictReader(f))
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_keys)
                writer.writeheader()
                for r in existing:
                    writer.writerow(r)
                writer.writerow(row)
        else:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_keys)
                writer.writerow(row)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
