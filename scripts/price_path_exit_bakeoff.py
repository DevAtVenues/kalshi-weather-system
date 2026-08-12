"""
Step 4 — exit-policy bake-off for the price-path book (DORMANT PILOT).

Runs every exit policy in the engine's library head-to-head over the full
Step-1 panel: fixed horizons, target-price ("convergence complete"),
trailing-peak, stop-loss (expected worst — the empirical receipt for
hold-to-settlement), and hold-to-end. Reports per-policy net P&L under BOTH
fill models with day-block bootstrapped CIs.

Context: Step 2 returned NO-GO (2026-08-11) — no policy is expected to clear
costs on current data. This harness exists so that verdict is re-checkable for
free as the logged panel grows, and so activation (if ever) starts from a
tested engine + a named champion policy rather than a fresh build.

Reads : data/analysis/price_path/panel.parquet   (build_price_path_panel.py)
Writes: data/analysis/price_path/exit_bakeoff.csv
Run   : .venv/bin/python scripts/price_path_exit_bakeoff.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.pricepath.engine import (  # noqa: E402
    POLICIES, day_block_ci, gated_entries, simulate,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "analysis" / "price_path" / "panel.parquet"
OUT = ROOT / "data" / "analysis" / "price_path" / "exit_bakeoff.csv"


def c(x: float) -> str:
    return f"{100 * x:+6.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel).reset_index(drop=True)
    entries = gated_entries(panel)
    print("=" * 76)
    print("PRICE-PATH EXIT-POLICY BAKE-OFF  (Step 4 — dormant pilot harness)")
    print("=" * 76)
    print(f"panel rows {len(panel):,} | gated entries {len(entries):,} | "
          f"days {entries['settlement_date'].nunique()}")
    print(f"\n  {'policy':>14} {'n':>6} {'days':>4} {'held(med)':>9} "
          f"{'PESS net':>9} {'ci95':>18} {'OPT net':>9}")

    rows = []
    for name, pol in POLICIES.items():
        trades = simulate(panel, pol, entries=entries)
        if trades.empty:
            continue
        p = day_block_ci(trades, "pnl_pess")
        o = day_block_ci(trades, "pnl_opt")
        held = trades["held_min"].median()
        flag = "  <== CI>0" if p[1] > 0 else ""
        print(f"  {name:>14} {p[3]:>6,} {p[4]:>4} {held:>8.0f}m "
              f"{c(p[0]):>9} [{c(p[1])},{c(p[2])}] {c(o[0]):>9}{flag}")
        rows.append({"policy": name, "n": p[3], "days": p[4],
                     "held_med_min": held,
                     "pess": p[0], "pess_lo": p[1], "pess_hi": p[2],
                     "opt": o[0]})

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    best = res.loc[res["pess_lo"].idxmax()]
    print("\n" + "=" * 76)
    if best["pess_lo"] > 0:
        print(f"CHAMPION: {best['policy']} clears the pessimistic bar "
              f"({c(best['pess'])}c [{c(best['pess_lo'])},{c(best['pess_hi'])}]) "
              "— re-run the Step-2 study and escalate for an activation decision.")
    else:
        print(f"NO policy clears the pessimistic bar (best lower bound: "
              f"{best['policy']} at {c(best['pess_lo'])}c) — consistent with the "
              "Step-2 NO-GO. Pilot stays dormant; the shadow book keeps accruing.")
    print(f"(wrote {OUT})")
    print("=" * 76)


if __name__ == "__main__":
    main()
