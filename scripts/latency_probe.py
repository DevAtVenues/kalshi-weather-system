"""
Market reaction-latency probe: WHEN do day-ahead mids move?

Builds mid-price series from the logged orderbook depth snapshots and profiles
mean |Δmid| per 5-min interval by UTC hour, day-ahead contracts only (same-day
moves are obs-driven and would swamp the signal), live-priced books only
(10–90¢; near-resolved books don't reprice).

First run (2026-07-12, full archive): 14 UTC (10 AM ET) mean move = 1.76¢/5min
vs 0.38¢ afternoon baseline — 4.6×. Overnight 00z/06z model output gets priced
HOURS after publication → the morning stale-market window (playbook trade #3).
Caveat: 05–13 UTC has zero snapshots (laptop asleep) — pre-wave books unobserved
until the machine stays awake overnight; venue trading hours unverified there.

Usage: .venv/bin/python scripts/latency_probe.py
"""
from __future__ import annotations

import glob
import re
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
ORDERBOOK = ROOT / "data" / "logger" / "orderbook"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def contract_date(ticker: str):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    return pd.Timestamp(2000 + int(m.group(1)), MONTHS[m.group(2)],
                        int(m.group(3))).date()


def hour_profile() -> dict[int, tuple[float, int]]:
    prof: dict[int, list] = {}
    for f in sorted(glob.glob(str(ORDERBOOK / "*.parquet"))):
        df = pd.read_parquet(f)
        best = (df.groupby(["snapshot_utc", "ticker", "side"])["price_cents"]
                  .max().unstack("side").dropna(subset=["yes", "no"]))
        best["mid"] = (best["yes"] + (100 - best["no"])) / 2
        best = best.reset_index()
        cdate = {t: contract_date(t) for t in best["ticker"].unique()}
        best["cdate"] = best["ticker"].map(cdate)
        best["et_date"] = (best["snapshot_utc"] - timedelta(hours=5)).dt.date
        best = best[best["cdate"] > best["et_date"]]          # day-ahead only
        best = best[(best["mid"] >= 10) & (best["mid"] <= 90)]
        best = best.sort_values("snapshot_utc")
        best["dmid"] = best.groupby("ticker")["mid"].diff().abs()
        gap = best.groupby("ticker")["snapshot_utc"].diff().dt.total_seconds() / 60
        ok = best[(gap > 2) & (gap < 15)]                     # true 5-min steps only
        for h, g in ok.groupby(ok["snapshot_utc"].dt.hour):
            a = prof.setdefault(int(h), [0.0, 0])
            a[0] += g["dmid"].sum()
            a[1] += len(g)
    return {h: (s / n if n else float("nan"), n) for h, (s, n) in prof.items()}


if __name__ == "__main__":
    prof = hour_profile()
    print("UTC hr | mean |Δmid| ¢/5min | n      (day-ahead, mid 10–90¢)")
    for h in range(24):
        mu, n = prof.get(h, (float("nan"), 0))
        bar = "#" * int(mu * 40) if n else ""
        print(f"  {h:02d}   |  {mu:5.2f}" if n else f"  {h:02d}   |    —  ",
              f"          | {n:>6} {bar}")
    if not any(h in prof for h in range(5, 14)):
        print("\n⚠ 05–13 UTC unobserved (machine asleep overnight) — pre-wave "
              "books unverified; keep the laptop awake one night to close this.",
              file=sys.stderr)
