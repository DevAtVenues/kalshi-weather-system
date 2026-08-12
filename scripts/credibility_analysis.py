"""
Credibility gate analysis — P0.4.

Reads data/logger/picks/*.jsonl (written by dashboard store._log_pick_credibility)
and grades each dashboard pick's credibility verdict against the actual outcome.

Key question: are REJECT picks actually losing more than WARN/OK picks?
If REJECT picks are winning at > 55%, the gate is FILTERING WINNERS — loosen it.
If REJECT picks win at < 45%, the gate is SAVING US — keep or tighten it.

Usage: .venv/bin/python scripts/credibility_analysis.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT   = Path(__file__).parents[1]
PICKS  = ROOT / "data" / "logger" / "picks"
OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"


def _load_outcomes() -> dict[str, int]:
    """Map ticker → yes_settled (1/0) from signal_outcomes."""
    settled: dict[str, int] = {}
    if not OUTCOMES.exists():
        return settled
    for line in OUTCOMES.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("yes_settled") is not None:
            settled[r["ticker"]] = 1 if r["yes_settled"] else 0
    return settled


def _load_picks() -> list[dict]:
    """Load all credibility pick-log entries, deduped to last verdict per ticker-day."""
    if not PICKS.exists():
        return []
    latest: dict[tuple, dict] = {}
    for path in sorted(PICKS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (d.get("ticker"), d.get("settlement_date"), d.get("action"))
            if k not in latest or d.get("log_ts", "") > latest[k].get("log_ts", ""):
                latest[k] = d
    return list(latest.values())


def _would_win(d: dict, y: int) -> bool:
    action = d.get("action", "")
    return (action == "BUY YES" and y == 1) or (action == "BUY NO" and y == 0)


def report() -> str:
    picks   = _load_picks()
    settled = _load_outcomes()

    graded = [(p, settled[p["ticker"]]) for p in picks if p.get("ticker") in settled]
    if not graded:
        return ("  No graded credibility picks yet.\n"
                "  (picks are logged when dashboard refreshes; grades after settlement.)")

    out = [f"  {len(graded)} graded credibility-pick decisions."]
    out.append("  Held-pick win-rate = what % would have won if we had ignored the credibility gate?\n")

    tiers = ["ok", "warn", "reject"]
    for tier in tiers:
        g = [(p, y) for p, y in graded if p.get("credibility_tier") == tier]
        if not g:
            continue
        days = len({p.get("settlement_date") for p, _ in g})
        by_day: dict[str, list[int]] = defaultdict(list)
        for p, y in g:
            by_day[p.get("settlement_date", "?")].append(1 if _would_win(p, y) else 0)
        wr = st.mean(st.mean(v) for v in by_day.values())

        if tier == "ok":
            verdict = "✅ good" if wr > 0.55 else ("⚠ weak" if wr > 0.45 else "❌ losing")
        elif tier == "warn":
            verdict = ("✅ SAVED us (filtered losers)" if wr < 0.45
                       else "❌ COST us (filtered winners) — loosen" if wr > 0.55
                       else "• neutral (no info)")
        else:  # reject
            verdict = ("✅ SAVED us (correct rejects)" if wr < 0.45
                       else "❌ COST us (false rejects) — gate too tight" if wr > 0.55
                       else "• neutral")

        out.append(f"    {tier.upper():<6}  n={len(g):<4}  days={days:<3}  win% {wr:5.0%}   {verdict}")

    # Per-reject-reason breakdown
    reason_wins: dict[str, list[int]] = defaultdict(list)
    for p, y in graded:
        for reason in p.get("credibility_reasons", []):
            key = reason.split(" ")[0]   # first word as category
            reason_wins[key].append(1 if _would_win(p, y) else 0)

    if reason_wins:
        out.append("\n  REJECT reason breakdown (what's each sub-gate catching?):")
        for reason, ws in sorted(reason_wins.items(), key=lambda x: st.mean(x[1])):
            if len(ws) < 3:
                continue
            wr = st.mean(ws)
            verdict = ("✅ catching losers" if wr < 0.45
                       else "❌ catching winners" if wr > 0.55
                       else "• neutral")
            out.append(f"    {reason:<20} n={len(ws):<3}  win% {wr:5.0%}   {verdict}")

    out.append("\n  (need ≥10 graded days per tier before acting on it — current numbers are early reads.)")
    return "\n".join(out)


def main() -> None:
    print("=" * 70)
    print("  CREDIBILITY GATE ANALYSIS (P0.4)")
    print("=" * 70)
    print(report())


if __name__ == "__main__":
    main()
