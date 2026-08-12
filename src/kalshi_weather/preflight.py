"""Entry-point preflight: a scheduled job must REFUSE to run hollow.

The failure mode this kills: a job starts in a broken environment (env vars
absent, calibration artifact missing, output dir gone after a rename) and still
exits 0 while producing nothing useful — the phantom-push / dead-book-check /
orderbook-hole class. Every scheduled entry point calls preflight(<job>) first;
on any problem it pushes a health alert (best-effort), prints loudly, and exits
nonzero — which launchd surfaces and scripts/health_check.py flags within one
window (stale scheduler marker + agent exit status).

Checks are deliberately DUMB — env var present, file exists, dir writable. No
network, no model logic, nothing that can itself misfire and block a good run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]

# job → hard requirements. Keep entries boring: (kind, target, why).
_REQUIREMENTS: dict[str, list[tuple[str, str, str]]] = {
    "run_live": [
        ("env",  "NTFY_TOPIC", "pushes silently no-op without it (.env not loaded?)"),
        ("file", "data/calibration/diurnal_warming_curve.json",
                 "same-day pricing falls back to the FALSIFIED linear taper"),
        ("file", "data/models/prob_calibration.json",
                 "probability recalibration map missing — raw probs would ship"),
        ("dir",  "data/signals", "signals output directory"),
        ("dir",  "logs/scheduler", "slot-guard state directory"),
    ],
    "candidate_pipeline": [
        ("file", "data/signals/signals_log.jsonl", "Stage 1 has no universe to read"),
        ("dir",  "data/analysis/candidates", "board output directory"),
    ],
    "daily_review": [
        ("env",  "NTFY_TOPIC", "watchdog/health alerts silently no-op without it"),
        ("dir",  "data/raw/labels", "settlement labels root"),
    ],
}


def preflight(job: str) -> None:
    problems: list[str] = []
    for kind, target, why in _REQUIREMENTS.get(job, []):
        if kind == "env" and not os.getenv(target):
            problems.append(f"env {target} unset — {why}")
        elif kind == "file" and not (_ROOT / target).is_file():
            problems.append(f"missing file {target} — {why}")
        elif kind == "dir":
            d = _ROOT / target
            d.mkdir(parents=True, exist_ok=True)
            if not os.access(d, os.W_OK):
                problems.append(f"dir {target} not writable — {why}")
    if not problems:
        return

    msg = f"PREFLIGHT FAILED ({job}):\n" + "\n".join(f"  - {p}" for p in problems)
    print(msg, file=sys.stderr)
    try:  # best-effort: if the push channel is itself the problem, this no-ops
        from kalshi_weather.dashboard.notifications import notify_health
        notify_health(f"Preflight failed: {job}", msg)
    except Exception:
        pass
    sys.exit(78)      # EX_CONFIG — launchd records it; health_check flags it
