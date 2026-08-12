"""
Morning-edge tracker — does our forecast front-run the market?

The NYC-Jul-7 lesson: the edge was in the MORNING (our cool view vs a hot market), and by
the time the observation confirmed it the market had repriced to 99¢ — a real edge we
couldn't capture because we didn't trust our forecast enough to act early. This measures,
for every settled contract, whether:
  1. our EARLY read (max lead) disagreed with the market, and
  2. the market later CONVERGED toward our early view, and
  3. our early read BEAT the market (closer to the actual outcome), and
  4. how much edge was CAPTURABLE by entering early at the market's early price.

If we consistently win these, the model is trustworthy enough to front-run — and that's the
day P1.0/live-sizing turns on. Reads the continuous signals_log (every cycle logs our P +
market_mid per contract) + graded outcomes. Ensemble-sourced only. Blocked by settlement
DAY for the honest effective-n.

Usage: .venv/bin/python scripts/morning_edge.py [--min-dis 0.10] [--live]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
SIGNALS = ROOT / "data" / "signals" / "signals_log.jsonl"
OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"
LOG = ROOT / "data" / "analysis" / "morning_edge.jsonl"


def _load():
    series: dict[str, list[dict]] = defaultdict(list)
    for line in SIGNALS.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("prob_source") != "ensemble":
            continue
        if r.get("prob_estimate") is None or r.get("market_mid") is None:
            continue
        series[r["ticker"]].append(r)
    settled: dict[str, int] = {}
    for line in OUTCOMES.read_text().splitlines() if OUTCOMES.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("yes_settled") is not None:
            settled[r["ticker"]] = 1 if r["yes_settled"] else 0
    return series, settled


def _early_late(snaps: list[dict]) -> tuple[dict, dict]:
    """early = most lead-time (largest hours_to_settle, our cleanest forecast read);
    late = last snapshot before settlement (the market's final word)."""
    with_h = [s for s in snaps if s.get("hours_to_settle") is not None]
    early = max(with_h, key=lambda s: s["hours_to_settle"]) if with_h else snaps[0]
    late = min(snaps, key=lambda s: abs(s.get("hours_to_settle") or 0)) if with_h else snaps[-1]
    return early, late


def analyse(min_dis: float) -> list[dict]:
    series, settled = _load()
    cases = []
    for tk, snaps in series.items():
        if tk not in settled or len(snaps) < 2:
            continue
        early, late = _early_late(snaps)
        our_e = float(early["prob_estimate"]); mkt_e = float(early["market_mid"])
        mkt_l = float(late["market_mid"]); y = settled[tk]
        dis = our_e - mkt_e
        if abs(dis) < min_dis:
            continue
        converged = (mkt_l - mkt_e) * dis > 0                     # market moved toward our early view
        we_beat = (mkt_e - y) ** 2 > (our_e - y) ** 2            # our early read closer to truth (Brier)
        capturable = (1 if dis > 0 else -1) * (y - mkt_e)        # enter early at mkt_e on our side → realized P&L
        cases.append({
            "ticker": tk, "sdate": early.get("settlement_date"), "city": early.get("city"),
            "our_early": round(our_e, 2), "mkt_early": round(mkt_e, 2), "mkt_late": round(mkt_l, 2),
            "outcome": y, "dis": round(dis, 2), "converged": converged, "we_beat": we_beat,
            "capturable_c": round(capturable * 100, 1),
        })
    return cases


def report(cases: list[dict]) -> str:
    if not cases:
        return "  No settled morning-disagreement cases yet."
    by_day = defaultdict(list)
    for c in cases:
        by_day[c["sdate"]].append(c)
    days = len(by_day)
    conv = st.mean(1 if c["converged"] else 0 for c in cases)
    beat = st.mean(1 if c["we_beat"] else 0 for c in cases)
    # capturable edge, blocked by day (per-day mean, then mean of days) — honest effective-n
    day_cap = [st.mean(c["capturable_c"] for c in cs) for cs in by_day.values()]
    out = [f"  {len(cases)} disagreement-contracts over {days} distinct days"
           f"  (effective n = DAYS; need ≥10 to trust)",
           f"  market converged to our early view : {conv:5.0%}",
           f"  our early read beat the market      : {beat:5.0%}",
           f"  capturable edge (entered early)     : {st.mean(day_cap):+.1f}¢/contract  (per-day mean)"]
    trust = "✅ front-running looks REAL" if (days >= 10 and beat > 0.55 and st.mean(day_cap) > 0) \
        else ("⏳ promising, need more days" if beat > 0.5 else "❌ no front-run edge yet")
    out.append(f"  → {trust}")
    recent = sorted(cases, key=lambda c: c["sdate"], reverse=True)[:6]
    out.append("  recent cases (city sdate: our→mkt then market moved→ / outcome):")
    for c in recent:
        arrow = "converged✓" if c["converged"] else "diverged✗"
        win = "WON" if c["we_beat"] else "lost"
        out.append(f"    {c['city']:<4} {c['sdate']}  {c['our_early']:.2f}→mkt {c['mkt_early']:.2f}, "
                   f"mkt went {c['mkt_late']:.2f} [{arrow}], settled {c['outcome']} [{win}] "
                   f"{c['capturable_c']:+.0f}¢")
    return "\n".join(out)


def live_watch(min_dis: float) -> str:
    """Open contracts where our current ensemble read disagrees with the market — the
    morning edges to consider front-running NOW."""
    series, _ = _load()
    today = date.today().isoformat()
    rows = []
    for tk, snaps in series.items():
        latest = max(snaps, key=lambda s: s.get("run_ts", ""))
        sd = latest.get("settlement_date", "")
        if sd < today:
            continue
        our = float(latest["prob_estimate"]); mkt = float(latest["market_mid"])
        if not (0.02 < mkt < 0.98):        # skip already-resolved/locked markets (no room to front-run)
            continue
        if abs(our - mkt) >= min_dis:
            rows.append((abs(our - mkt), tk, latest.get("city"), our, mkt, sd))
    if not rows:
        return "  none right now."
    out = []
    for _, tk, city, our, mkt, sd in sorted(rows, reverse=True)[:12]:
        side = "YES" if our > mkt else "NO"
        out.append(f"    {city:<4} {sd} {tk:<24} market {mkt*100:3.0f}% vs our {our*100:3.0f}%  → lean {side}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dis", type=float, default=0.10)
    ap.add_argument("--live", action="store_true", help="also show current open disagreements to watch")
    args = ap.parse_args()

    cases = analyse(args.min_dis)
    print("=" * 70)
    print("  MORNING-EDGE TRACKER — does our forecast front-run the market?")
    print("=" * 70)
    print(report(cases))

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"run": date.today().isoformat(), "n_cases": len(cases),
                            "days": len({c["sdate"] for c in cases})}) + "\n")

    if args.live:
        print("\n  LIVE — open disagreements to watch (front-run candidates):")
        print(live_watch(args.min_dis))


if __name__ == "__main__":
    main()
