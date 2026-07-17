"""Reproducibility manifests: record exactly how a result was produced.

A headline number is only trustworthy if it can be reproduced. :func:`run_manifest`
captures the provenance of a run — git commit, seed, config, library versions,
platform, timestamp and final metrics — as a plain dict you can save to JSON next
to the model. It's dependency-free (stdlib only) and used by
:func:`decisionrl.config.run`.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["git_sha", "library_versions", "run_manifest", "save_manifest"]


def git_sha(short: bool = True) -> Optional[str]:
    """Current git commit hash, or ``None`` if not in a git repo / git missing."""
    try:
        args = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None


def library_versions() -> Dict[str, str]:
    """Versions of the packages that determine a run's numerical result."""
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("decisionrl", "numpy", "torch"):
        with contextlib.suppress(ImportError, AttributeError):
            versions[mod] = __import__(mod).__version__
    return versions


def run_manifest(config: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble a provenance record for a run (git, versions, config, metrics)."""
    manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "versions": library_versions(),
        "platform": platform.platform(),
        "seed": seed,
        "config": config,
        "metrics": metrics or {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: Dict[str, Any], path: str) -> str:
    """Write a manifest to ``path`` as pretty JSON; return the path."""
    Path(path).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
