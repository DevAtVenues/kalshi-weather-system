#!/usr/bin/env python
"""Timing & edge-realization report over graded signal snapshots.

Read-only analysis of data/outcomes/signal_outcomes.jsonl. Each graded row is a
(run_ts, ticker) observation carrying the market mid at observation time, our
model probability, hours_to_settle, and the final settled outcome — so we can
ask, for any slice: "had we traded the signal's direction at that moment's mid,
what would one contract have made after taker fees?"

Sections:
  1. Realized P&L by model-edge bucket        (is edge monotone -> real info?)
  2. Realized P&L by lead-time bucket         (WHEN does firing pay?)
  3. Realized P&L by direction x strike_type  (which species carry the P&L?)
  4. Convergence: on >=5c model-vs-market disagreements, does the market
     subsequently move toward us (we front-run) or do we capitulate?
  5. Stop-rule check: hold-to-settlement vs adverse-move exits on the
     BUY_NO-between species (history says stops destroy this edge).

All confidence intervals are block-bootstrapped over settlement_date — weather
days are autocorrelated, snapshots within a day are heavily correlated (HARD
RULE: never report an edge point-estimate without its interval).

Usage:  .venv/bin/python scripts/timing_report.py [--days N]  (default: all)
Run ad hoc or from daily_review; writes nothing, pushes nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"

LEAD_BINS = [-24.1, -18, -12, -6, 0, 6, 12, 18]
# hours_to_settle = hours until the local settlement window OPENS (negative =
# window already open). Labels below translate each bin to local wall-clock.
LEAD_LABELS = [
    "day 18-24h local (evening, near-dead)",   # (-24,-18]
    "day 12-18h local (AFTERNOON)",            # (-18,-12]
    "day 06-12h local (morning)",              # (-12, -6]
    "day 00-06h local (overnight)",            # ( -6,  0]
    "prev-day 18-24h local",                   # (  0,  6]
    "prev-day 12-18h local (afternoon)",       # (  6, 12]
    "prev-day 06-12h local (morning)",         # ( 12, 18]
]

rng = np.random.default_rng(7)


def taker_fee(p: np.ndarray | float) -> np.ndarray | float:
    return 0.07 * p * (1 - p)


def block_ci(vals: np.ndarray, dates: np.ndarray, n_boot: int = 800):
    """Mean + 95% CI, bootstrapping whole settlement_date blocks."""
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0
    groups: dict = {}
    for v, d in zip(vals, dates):
        groups.setdefault(d, []).append(v)
    days = list(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(days), size=len(days), replace=True)
        means[b] = np.mean(np.concatenate([groups[days[i]] for i in pick]))
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(days)


def line(name: str, sub: pd.DataFrame, col: str = "pnl_taker") -> None:
    m, lo, hi, nd = block_ci(sub[col].to_numpy(), sub["settlement_date"].to_numpy())
    if nd == 0:
        print(f"   {name:44s} (no data)")
        return
    flag = "++" if lo > 0 else ("--" if hi < 0 else "  ")
    print(f"{flag} {name:44s} n={len(sub):6d} days={nd:2d} "
          f"mean={m * 100:+6.1f}c CI[{lo * 100:+6.1f},{hi * 100:+6.1f}] "
          f"win={sub['won'].mean() * 100:5.1f}% px={sub['price_paid'].mean() * 100:3.0f}c")


def load(days: int | None) -> pd.DataFrame:
    df = pd.read_json(OUTCOMES, lines=True)
    df = df[df["market_mid"].notna() & df["yes_settled"].notna() & df["hours_to_settle"].notna()].copy()
    if df.empty:
        sys.exit("no graded rows with market_mid — nothing to analyze")
    df["run_ts"] = pd.to_datetime(df["run_ts"])
    if days:
        cutoff = df["settlement_date"].sort_values().unique()[-days:]
        df = df[df["settlement_date"].isin(cutoff)]
    df["yes_settled"] = df["yes_settled"].astype(float)
    df["price_paid"] = np.where(df["direction"] == "BUY_YES", df["market_mid"], 1 - df["market_mid"])
    df["won"] = np.where(df["direction"] == "BUY_YES", df["yes_settled"], 1 - df["yes_settled"])
    df["pnl_taker"] = df["won"] - df["price_paid"] - taker_fee(df["price_paid"])
    df["model_edge"] = np.where(df["direction"] == "BUY_YES",
                                df["prob_estimate"] - df["market_mid"],
                                df["market_mid"] - df["prob_estimate"])
    df["lead_bucket"] = pd.cut(df["hours_to_settle"], bins=LEAD_BINS, labels=LEAD_LABELS)
    df["tradeable"] = (df["price_paid"] >= 0.03) & (df["price_paid"] <= 0.97)
    return df


def convergence(df: pd.DataFrame) -> None:
    rows = []
    for tick, g in df.groupby("ticker"):
        if len(g) < 4:
            continue
        g = g.sort_values("run_ts")
        first, last = g.iloc[0], g.iloc[-1]
        if (last.run_ts - first.run_ts).total_seconds() < 6 * 3600:
            continue
        d0 = first.prob_estimate - first.market_mid
        if abs(d0) < 0.05:
            continue
        rows.append(dict(date=first.settlement_date, src=first.prob_source, d0=d0,
                         dm=last.market_mid - first.market_mid,
                         right=(last.yes_settled == 1.0) if d0 > 0 else (last.yes_settled == 0.0)))
    c = pd.DataFrame(rows)
    if c.empty:
        print("   (not enough multi-snapshot tickers — check logging cadence/uptime)")
        return
    for src in ["ensemble", "rule"]:
        cs = c[c.src == src]
        if len(cs) < 10:
            continue
        toward = ((np.sign(cs.dm) == np.sign(cs.d0)) & (cs.dm.abs() >= 0.02)).astype(float)
        m1, lo1, hi1, _ = block_ci(toward.to_numpy(), cs.date.to_numpy())
        m2, lo2, hi2, nd = block_ci(cs.right.astype(float).to_numpy(), cs.date.to_numpy())
        print(f"   {src:9s} n={len(cs):5d} days={nd:2d}  "
              f"market->us {m1 * 100:4.1f}% [{lo1 * 100:.0f},{hi1 * 100:.0f}]   "
              f"outcome sided with model {m2 * 100:4.1f}% [{lo2 * 100:.0f},{hi2 * 100:.0f}]")


def stop_rule(df: pd.DataFrame) -> None:
    trades = []
    for tick, g in df[df.strike_type == "between"].groupby("ticker"):
        g = g.sort_values("run_ts")
        trig = g[(g.direction == "BUY_NO") & (g.model_edge >= 0.10)
                 & (g.prob_source == "ensemble") & g.market_mid.between(0.10, 0.90)]
        if trig.empty:
            continue
        e = trig.iloc[0]
        path = g[g.run_ts > e.run_ts]["market_mid"].to_numpy()
        outcome = float(g.yes_settled.iloc[-1])
        hold = (e.market_mid - outcome) - taker_fee(1 - e.market_mid)
        trades.append((e.settlement_date, e.market_mid, outcome, hold, path))
    if not trades:
        print("   (no qualifying ensemble BUY_NO-between entries)")
        return
    dates = np.array([t[0] for t in trades])
    holds = np.array([t[3] for t in trades])
    m, lo, hi, nd = block_ci(holds, dates)
    print(f"   HOLD to settle: n={len(trades)} days={nd} mean {m * 100:+.1f}c [{lo * 100:+.1f},{hi * 100:+.1f}]")
    for X in (0.15, 0.25):
        pnls = []
        for _, mid0, outcome, hold, path in trades:
            crossed = path[path - mid0 >= X]
            if len(crossed):
                pnls.append((mid0 - crossed[0]) - taker_fee(1 - mid0) - taker_fee(1 - crossed[0]))
            else:
                pnls.append(hold)
        m, lo, hi, _ = block_ci(np.array(pnls), dates)
        print(f"   STOP at +{int(X * 100)}c : mean {m * 100:+.1f}c [{lo * 100:+.1f},{hi * 100:+.1f}]"
              f"   (if this beats HOLD, history has changed — re-examine)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="restrict to last N settlement dates")
    args = ap.parse_args()

    df = load(args.days)
    t = df[df["tradeable"]]
    span = f"{df['settlement_date'].min()} -> {df['settlement_date'].max()}"
    print(f"graded snapshots with market price: {len(df)} ({span}); tradeable-px: {len(t)}")
    print("mid-fill minus taker fee; ++/-- = 95% block-bootstrap CI clear of zero\n")

    print("1) by model-edge bucket (should be monotone if edge is information):")
    for lo_e, hi_e in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 1.0)]:
        line(f"edge [{lo_e:.2f},{hi_e:.2f})", t[(t.model_edge >= lo_e) & (t.model_edge < hi_e)])

    e = t[t.model_edge >= 0.10]
    print("\n2) edge>=10c by lead window (local wall-clock at the settlement station):")
    for lb in LEAD_LABELS:
        line(lb, e[e.lead_bucket == lb])

    print("\n3) edge>=10c by direction x strike_type:")
    for d in ("BUY_NO", "BUY_YES"):
        for st in ("between", "greater", "less"):
            line(f"{d} {st}", e[(e.direction == d) & (e.strike_type == st)])

    print("\n4) convergence on >=5c disagreements held >=6h (do we front-run?):")
    convergence(df)

    print("\n5) stop-rule check, ensemble BUY_NO-between (hold has always won):")
    stop_rule(df)

    n_days_sparse = (df.groupby("settlement_date")["run_ts"].nunique() < 10).sum()
    if n_days_sparse:
        print(f"\nWARNING: {n_days_sparse} settlement dates have <10 distinct run "
              f"timestamps — machine-sleep gaps degrade every table above.")


if __name__ == "__main__":
    main()
