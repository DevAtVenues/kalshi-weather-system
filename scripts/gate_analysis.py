"""
Gate analysis — is each push gate saving us or costing us?

Reads the counterfactual gate log (data/analysis/gate_log.jsonl, written by
push_forecast_picks) and grades every HELD candidate against the actual outcome: if we
had fired it, would it have won? Per gate:
  • held-pick win-rate < ~45%  → the gate SAVED us (it filtered losers) ✅
  • held-pick win-rate > ~55%  → the gate COST us (it filtered winners) ❌ loosen it
  • ~50%                       → neutral (the gate isn't adding information)
This turns each eyeball-set threshold into a data-tunable decision — without waiting to
place the trades, because held picks grade for free. Blocked by settlement DAY (effective-n).

Usage: .venv/bin/python scripts/gate_analysis.py [--live]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
GATE_LOG = ROOT / "data" / "analysis" / "gate_log.jsonl"
OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"

FILTER_GATES = ["horizon", "edge_below_bar", "model_spread", "nws_disagree", "ens_wide", "ceiling", "dedup"]


def _load():
    # dedup to the LAST decision per (ticker, direction, settlement_date) — the engine's
    # final verdict for that contract-day.
    latest: dict[tuple, dict] = {}
    if GATE_LOG.exists():
        for line in GATE_LOG.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (d["ticker"], d["direction"], d["settlement_date"])
            if k not in latest or d["run_ts"] > latest[k]["run_ts"]:
                latest[k] = d
    settled: dict[str, int] = {}
    for line in OUTCOMES.read_text().splitlines() if OUTCOMES.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("yes_settled") is not None:
            settled[r["ticker"]] = 1 if r["yes_settled"] else 0
    return list(latest.values()), settled


def _would_win(d: dict, y: int) -> bool:
    return (d["direction"] == "BUY_YES" and y == 1) or (d["direction"] == "BUY_NO" and y == 0)


def report() -> str:
    decs, settled = _load()
    graded = [(d, settled[d["ticker"]]) for d in decs if d["ticker"] in settled]
    if not graded:
        return "  No graded gate decisions yet (engine populates gate_log each cycle; grades after settlement)."

    out = [f"  {len(graded)} graded gate-decisions.  Held-pick win-rate = would it have won if fired?"]
    # FIRED bucket first — our actual auto-fires
    for gate in ["FIRED"] + FILTER_GATES:
        g = [(d, y) for d, y in graded if d["gate"] == gate]
        if not g:
            continue
        days = len({d["settlement_date"] for d, _ in g})
        # block by day: per-day win-rate, then mean
        by_day = defaultdict(list)
        for d, y in g:
            by_day[d["settlement_date"]].append(1 if _would_win(d, y) else 0)
        wr = st.mean(st.mean(v) for v in by_day.values())
        if gate == "FIRED":
            verdict = "✅ good" if wr > 0.55 else ("⚠ weak" if wr > 0.45 else "❌ losing")
            label = "fired & won"
        else:
            verdict = ("✅ SAVED us (filtered losers)" if wr < 0.45
                       else "❌ COST us (filtered winners) — loosen" if wr > 0.55
                       else "• neutral (no info)")
            label = "would-have-won"
        out.append(f"    {gate:<14} n={len(g):<4} days={days:<3} {label} {wr:5.0%}   {verdict}")
    out.append("  (need ≥10 days per gate before acting on it; early reads are noisy.)")
    return "\n".join(out)


def live() -> str:
    decs, _ = _load()
    today = date.today().isoformat()
    held = [d for d in decs if d["gate"] in FILTER_GATES and d["settlement_date"] >= today]
    if not held:
        return "  nothing currently held."
    out = []
    for d in sorted(held, key=lambda d: d["settlement_date"]):
        p = d.get("prob_estimate"); m = d.get("market_mid")
        pm = f"our {p*100:.0f}% vs mkt {m*100:.0f}%" if (p is not None and m is not None) else ""
        out.append(f"    {d['city']:<4} {d['settlement_date']} {d['ticker']:<24} "
                   f"HELD by [{d['gate']}]  edge {d['edge_raw']:+.2f}  {pm}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    print("=" * 70)
    print("  GATE ANALYSIS — is each push gate saving us or costing us?")
    print("=" * 70)
    print(report())
    if args.live:
        print("\n  LIVE — currently held candidates (why the board is quiet):")
        print(live())


if __name__ == "__main__":
    main()
