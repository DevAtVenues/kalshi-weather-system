"""
Skill-vs-market drift monitor: are we still valid THIS WEEK?

The market price is the strongest baseline this system has — if our calibrated
probability stops beating the mid, there is no edge, whatever the backtest said.
forward_report answers "how have we done"; this is a control chart on "has the
model gone stale", refreshed daily from graded outcomes:

  SKILL = mean Brier(market) − mean Brier(model), rolling window (default 14
  settlement days). Positive = we price settlement better than the market did
  at the same moment. CI via block bootstrap BY SETTLEMENT DAY (same-day
  outcomes across cities are one weather system, not independent samples).

  BIAS = mean(prob − outcome). The YES-over-prediction incident, as a number.

Hygiene (each rule guards a known artifact):
  • rows with hours_to_settle ≤ 0 are dropped — the market has effectively
    resolved (mid ≈ 0.99) and would flatter the baseline
  • one row per (ticker, lead bucket): the LAST read — cycle-level rows are the
    same bet logged 4x/day, not more evidence
  • leads bucketed same-day (h ≤ 14) vs day-ahead (14 < h ≤ 40); alarms fire
    ONLY on the ensemble day-ahead bucket (the tradeable engine — same-day is
    quarantined watch-only, rule-source is legacy)

Alarms (pushed via notify_health on status TRANSITIONS, state in logs/health/):
  RED   skill < 0 and the 90% block-bootstrap CI excludes 0 (market beats us,
        significantly), or |bias| > 0.05 with CI excluding 0
  WARN  skill point-estimate < 0 but CI spans 0, or effective n < 8 days
Runs inside daily_review; standalone: .venv/bin/python scripts/skill_monitor.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"
STATE    = ROOT / "logs" / "health" / "skill_state.json"

WINDOW_DAYS   = 14
MIN_EFF_DAYS  = 8       # below this, intervals are block-count artifacts
BOOT_ITERS    = 2000
BIAS_ALARM    = 0.05
LEAD_BUCKETS  = (("same-day", 0.0, 14.0), ("day-ahead", 14.0, 40.0))


def load_rows() -> pd.DataFrame:
    rows = []
    try:
        with open(OUTCOMES) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    need = ["prob_estimate", "market_mid", "yes_settled", "settlement_date",
            "prob_source", "hours_to_settle", "ticker", "run_ts"]
    df = df.dropna(subset=[c for c in need if c in df.columns])
    df = df[df["hours_to_settle"] > 0]                    # market not yet resolved
    df["lead"] = None
    for name, lo, hi in LEAD_BUCKETS:
        df.loc[(df["hours_to_settle"] > lo) & (df["hours_to_settle"] <= hi),
               "lead"] = name
    df = df.dropna(subset=["lead"])
    # last read per (ticker, lead): the most-informed row in that bucket
    df = df.sort_values("run_ts").groupby(["ticker", "lead"], as_index=False).last()
    df["y"] = df["yes_settled"].astype(float)
    df["brier_model"]  = (df["prob_estimate"] - df["y"]) ** 2
    df["brier_market"] = (df["market_mid"]   - df["y"]) ** 2
    return df


def block_bootstrap_ci(df: pd.DataFrame, col_fn, iters: int = BOOT_ITERS,
                       q: tuple[float, float] = (5, 95)) -> tuple[float, float]:
    """Resample SETTLEMENT DAYS with replacement (a heat wave is one draw)."""
    days = df["settlement_date"].unique()
    rng = np.random.default_rng(0)          # fixed seed: reproducible reports
    groups = {d: g for d, g in df.groupby("settlement_date")}
    stats = []
    for _ in range(iters):
        sample = pd.concat([groups[d] for d in rng.choice(days, len(days))])
        stats.append(col_fn(sample))
    return float(np.percentile(stats, q[0])), float(np.percentile(stats, q[1]))


def analyse(df: pd.DataFrame, source: str, lead: str,
            window_days: int = WINDOW_DAYS) -> dict | None:
    sub = df[(df["prob_source"] == source) & (df["lead"] == lead)]
    if sub.empty:
        return None
    days = sorted(sub["settlement_date"].unique())[-window_days:]
    sub = sub[sub["settlement_date"].isin(days)]
    skill_fn = lambda d: float((d["brier_market"] - d["brier_model"]).mean())
    bias_fn  = lambda d: float((d["prob_estimate"] - d["y"]).mean())
    skill, bias = skill_fn(sub), bias_fn(sub)
    s_lo, s_hi = block_bootstrap_ci(sub, skill_fn)
    b_lo, b_hi = block_bootstrap_ci(sub, bias_fn)
    return {"source": source, "lead": lead, "n": len(sub), "days": len(days),
            "brier_model": float(sub["brier_model"].mean()),
            "brier_market": float(sub["brier_market"].mean()),
            "skill": skill, "skill_ci": (s_lo, s_hi),
            "bias": bias, "bias_ci": (b_lo, b_hi)}


def status_of(a: dict | None) -> tuple[str, str]:
    if a is None:
        return "WARN", "no graded rows for the tradeable bucket"
    if a["days"] < MIN_EFF_DAYS:
        return "WARN", f"only {a['days']} settlement days — interval not meaningful yet"
    if a["skill"] < 0 and a["skill_ci"][1] < 0:
        return "RED", (f"market beats the model: skill {a['skill']:+.4f} "
                       f"CI({a['skill_ci'][0]:+.4f},{a['skill_ci'][1]:+.4f})")
    if abs(a["bias"]) > BIAS_ALARM and (a["bias_ci"][0] > 0 or a["bias_ci"][1] < 0):
        return "RED", (f"systematic {'over' if a['bias'] > 0 else 'under'}-prediction: "
                       f"bias {a['bias']:+.3f} CI({a['bias_ci'][0]:+.3f},{a['bias_ci'][1]:+.3f})")
    if a["skill"] < 0:
        return "WARN", f"skill negative ({a['skill']:+.4f}) but CI spans 0 — watch"
    return "OK", f"skill {a['skill']:+.4f} CI({a['skill_ci'][0]:+.4f},{a['skill_ci'][1]:+.4f})"


def _fmt(a: dict | None, label: str) -> str:
    if a is None:
        return f"  {label:<22} (no rows)"
    return (f"  {label:<22} n={a['n']:>4} days={a['days']:>2}  "
            f"Brier us {a['brier_model']:.4f} vs mkt {a['brier_market']:.4f}  "
            f"skill {a['skill']:+.4f} CI({a['skill_ci'][0]:+.4f},{a['skill_ci'][1]:+.4f})  "
            f"bias {a['bias']:+.3f}")


def report(push: bool = True, window_days: int = WINDOW_DAYS) -> str:
    df = load_rows()
    if df.empty:
        return "(no graded outcomes yet — skill monitor idle)"
    ens_ahead = analyse(df, "ensemble", "day-ahead", window_days)
    status, why = status_of(ens_ahead)

    lines = [f"SKILL vs MARKET (last {window_days} settlement days, "
             f"block-bootstrap by day, fills-free Brier):",
             _fmt(ens_ahead, "ensemble day-ahead*"),
             _fmt(analyse(df, "ensemble", "same-day", window_days), "ensemble same-day"),
             _fmt(analyse(df, "rule", "day-ahead", window_days), "rule day-ahead (legacy)"),
             f"  * alarmed bucket → {status}: {why}"]

    if push:
        try:
            prev = json.loads(STATE.read_text()).get("status", "OK")
        except (OSError, json.JSONDecodeError):
            prev = "OK"
        if status != prev:
            from kalshi_weather.dashboard.notifications import notify_health
            if status == "RED":
                notify_health("Model skill alarm", f"ensemble day-ahead: {why}",
                              priority="high")
            elif prev == "RED":
                notify_health("Model skill recovered", why, priority="default",
                              tags="white_check_mark")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"status": status, "why": why, "ts": time.time()}))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(push="--no-push" not in sys.argv))
