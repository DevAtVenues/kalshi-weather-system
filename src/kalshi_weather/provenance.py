"""Run provenance: stamp every output with exactly what produced it.

Every signals row / candidate board records the git SHA (+dirty marker) and the
hashes of the calibration artifacts in force at write time. This is what makes a
pick reproducible (CLAUDE.md rule 8), lets the replay harness compare only rows
produced under the CURRENT calibration map (older rows differ legitimately), and
lets a post-mortem attribute any bad stretch to a model version instead of a vibe.

Cached per process — scheduled jobs are one process per run, so the stamp is
correct; a long-running dashboard keeps its startup stamp, which is also correct
(that IS the code it runs).
"""
from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_ARTIFACTS = {
    "calib_hash": _ROOT / "data" / "models" / "prob_calibration.json",
    "curve_hash": _ROOT / "data" / "calibration" / "diurnal_warming_curve.json",
}


def _md5_10(p: Path) -> str | None:
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()[:10]
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def provenance() -> dict:
    sha = _git("rev-parse", "--short=10", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"code_sha": sha + ("+dirty" if dirty else ""),
            **{k: _md5_10(p) for k, p in _ARTIFACTS.items()}}
