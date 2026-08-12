"""
Grade Stage-2 candidate verdicts against settlements.

The candidate pipeline (scripts/candidate_pipeline.py) writes a board JSON per
cycle: the full top-N board plus TRADE_SMALL/WATCH/PASS verdicts for the vetted
subset. This grades every board row once its settlement label exists and answers
the question the vet exists for: **does the vetting add value over the raw
board, and what did the rows we suppressed actually return?**

Method:
  • One graded row per (ticker, direction): the LATEST board before settlement
    (the final read). Earlier cycles of the same contract are dropped — grading
    every cycle would double-count persistent picks.
  • Return per $1 at the board's market mid: BUY_YES → y − mid, BUY_NO → mid − y
    (y = 1 if the contract settled YES on the integer CLI high).
  • Aggregates by verdict bucket (unvetted board rows = UNVETTED), by push-gate
    annotation, and same-day vs day-ahead.

CAVEATS (printed with the report): fills are at the logged mid with no spread or
fees — OPTIMISTIC; same-day outcomes across cities are autocorrelated (one
synoptic system = many correlated rows), so treat n as inflated; this is an
advisory diagnostic, not a backtest result.

Usage:
  .venv/bin/python scripts/grade_candidates.py
Also imported by scripts/daily_review.py (report()).
Output: data/analysis/candidates/grades.jsonl (rewritten — fully derived)
      + data/analysis/candidates/forward_stats.json (per-(city,type,direction)
        forward record — read by candidate_pipeline's validation check so the
        "(city,T/B) not OOS-validated" flag shows its accumulating evidence
        instead of being unclearable) + stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kalshi_weather.live.runner import CITY_CONFIGS               # noqa: E402
from candidate_pipeline import _labels, _settles_yes, OUT_DIR     # noqa: E402

GRADES = OUT_DIR / "grades.jsonl"
FORWARD_STATS = OUT_DIR / "forward_stats.json"

# Graduation bar (ADVISORY — reported, never auto-enforced). A (city, type,
# direction) combo "meets the bar" when its graded forward record would justify
# a HUMAN promoting it into ENSEMBLE_CITIES / the OOS ceiling. Mid-fill optimism
# + same-day autocorrelation mean these stats overstate live returns, so the bar
# is deliberately stiff and the switch stays human: nothing reads `graduated`
# to change live gating.
GRAD_MIN_N    = 30
GRAD_MIN_DAYS = 10
GRAD_MIN_WIN  = 0.60


def _final_highs(station: str, cache: dict) -> dict:
    """date-string → settled integer high, from the label parquets."""
    if station not in cache:
        lab = _labels(station)
        cache[station] = {d.strftime("%Y-%m-%d"): h
                          for d, h in zip(lab["date"], lab["high"])}
    return cache[station]


def load_graded_rows() -> pd.DataFrame:
    """Latest board row per (ticker, direction), graded where a label exists."""
    latest: dict[tuple, dict] = {}
    for f in sorted(OUT_DIR.glob("*.json")):
        if f.name == GRADES.name:
            continue
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        run_ts = doc.get("run_ts", "")
        vmap = {(v["ticker"], v["direction"]): v["verdict"]
                for v in doc.get("verdicts", [])}
        for b in doc.get("board", []):
            key = (b["ticker"], b["direction"])
            prev = latest.get(key)
            if prev is not None and prev["run_ts"] >= run_ts:
                continue
            latest[key] = {**b, "run_ts": run_ts,
                           "verdict": vmap.get(key, "UNVETTED")}

    label_cache: dict = {}
    rows = []
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for r in latest.values():
        if str(r["settlement_date"]) >= today:
            continue    # same guard as grade_signals: never grade an unfinished day
        station = (CITY_CONFIGS.get(r.get("city")) or {}).get("nws_station")
        if not station:
            continue
        high = _final_highs(station, label_cache).get(str(r["settlement_date"]))
        if high is None:
            continue                                   # not settled/labeled yet
        floor = r.get("floor_strike")
        cap = r.get("cap_strike")
        y = _settles_yes(str(r["strike_type"]),
                         float(floor) if pd.notna(floor) else None,
                         float(cap) if pd.notna(cap) else None, float(high))
        mid = float(r["market_mid"])
        ret = (1.0 - mid if y else -mid) if r["direction"] == "BUY_YES" \
            else (mid - 1.0 if y else mid)
        rows.append({**r, "final_high": float(high), "settled_yes": bool(y),
                     "ret_per_dollar": round(ret, 4)})
    return pd.DataFrame(rows)


def forward_stats(df: pd.DataFrame) -> dict:
    """Per-(city, contract-type, direction) forward record from graded board rows.

    This is the feedback loop the validation flag was missing: the vet FLAGs
    "(city,B) not OOS-validated" on every board, but until now no artifact
    summarized what those exact combos actually RETURNED forward, so the flag
    could never be weighed or cleared. Distinct settlement days are counted
    because same-day rows are one weather system, not independent evidence.
    """
    combos: dict[str, dict] = {}
    d = df.assign(ctype=[("B" if s == "between" else "T") for s in df["strike_type"]])
    for (city, ctype, direction), g in d.groupby(["city", "ctype", "direction"]):
        n, days = int(len(g)), int(g["settlement_date"].nunique())
        mean_ret = float(g["ret_per_dollar"].mean())
        win = float((g["ret_per_dollar"] > 0).mean())
        combos[f"{city}|{ctype}|{direction}"] = {
            "n": n, "days": days,
            "mean_ret": round(mean_ret, 4), "win": round(win, 4),
            "graduated": bool(n >= GRAD_MIN_N and days >= GRAD_MIN_DAYS
                              and mean_ret > 0 and win >= GRAD_MIN_WIN),
        }
    return {"meta": {
                "generated_by": "grade_candidates.py (rewritten each grading run)",
                "caveats": "fills at board mid — OPTIMISTIC, no fees/spread; "
                           "same-day rows autocorrelated; advisory only — "
                           "graduation is a human decision",
                "bar": {"min_n": GRAD_MIN_N, "min_days": GRAD_MIN_DAYS,
                        "min_win": GRAD_MIN_WIN, "mean_ret": "> 0"}},
            "combos": combos}


def _bucket_lines(df: pd.DataFrame, col: str, title: str) -> list[str]:
    lines = [title]
    for key, g in df.groupby(col):
        lines.append(f"  {str(key):<14} n={len(g):>3}  mean ret {g['ret_per_dollar'].mean():+.3f}"
                     f"  win {100 * (g['ret_per_dollar'] > 0).mean():.0f}%")
    return lines


def report() -> str:
    df = load_graded_rows()
    if df.empty:
        return "(no settled candidate boards to grade yet)"
    GRADES.write_text("\n".join(json.dumps(r, default=str)
                                for r in df.to_dict("records")) + "\n")
    stats = forward_stats(df)
    FORWARD_STATS.write_text(json.dumps(stats, indent=2))

    same = df["is_same_day"].fillna(False).astype(bool) if "is_same_day" in df \
        else pd.Series(False, index=df.index)
    lines = [f"{len(df)} settled candidates graded (latest board per contract; "
             f"fills at mid = OPTIMISTIC, no fees/spread; same-day rows autocorrelated)"]
    lines += _bucket_lines(df, "verdict", "by Stage-2 verdict:")
    if "species" in df.columns:
        lines += _bucket_lines(df.assign(species=df["species"].fillna("unlabeled")),
                               "species", "by species (hypothesized edge source):")
    lines += _bucket_lines(df, "gate", "by push-gate annotation (what did suppression save/cost?):")
    lines += _bucket_lines(df.assign(day=same.map({True: "same-day", False: "day-ahead"})),
                           "day", "by horizon:")
    grads = {k: v for k, v in stats["combos"].items() if v["graduated"]}
    lines.append(f"graduation bar (advisory: n≥{GRAD_MIN_N}, days≥{GRAD_MIN_DAYS}, "
                 f"win≥{GRAD_MIN_WIN:.0%}, mean>0 — mid-fill optimistic, human decides):")
    if grads:
        for k, v in sorted(grads.items(), key=lambda kv: -kv[1]["mean_ret"]):
            lines.append(f"  {k:<18} n={v['n']:>3}/{v['days']}d  mean {v['mean_ret']:+.3f}"
                         f"  win {v['win']:.0%}  MEETS BAR — eligible for human promotion")
    else:
        lines.append("  (no combo meets it yet)")
    ts = df[df["verdict"] == "TRADE_SMALL"]
    if len(ts):
        lines.append("TRADE_SMALL detail (the rows the vet would trade):")
        for _, r in ts.iterrows():
            lines.append(f"  {r['ticker']} {r['direction']} mid {r['market_mid']:.2f} "
                         f"→ high {r['final_high']:.0f} ret {r['ret_per_dollar']:+.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    out = report()
    print(out)
    if not out.startswith("("):
        print(f"\nwrote {GRADES}")
