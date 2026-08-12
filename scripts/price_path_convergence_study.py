"""
Step 2 of the price-path (mispricing) book — the GO/NO-GO gate.
See docs/STATE_OF_THE_MODEL.md Part V.

NO TRADING. This is a purely descriptive study on the Step-1 panel. It answers the
one question that can kill the whole strategy in an afternoon:

    When the ensemble model and the market disagree, does the market move toward the
    model FAST and FAR enough that the realized move clears the round-trip cost
    (2x spread + 2x fees) on our actual, thin weather books?

Method (deliberately pessimistic):
  * Entry signal at time t: divergence = model_P - yes_mid. Positive => model thinks
    YES is underpriced => we'd BUY YES and profit if the mid RISES; negative => BUY NO
    / profit if the mid FALLS. So the directional move is  sign(divergence) * dMid.
  * Forward move over horizon D: the SAME ticker's yes_mid at t+D (nearest logged
    snapshot within a tolerance), minus yes_mid at t. Measured in MID terms.
  * Pessimistic P&L accounting: you enter by crossing the spread (buy at ask) and exit
    by crossing it back (sell at bid). Entry-at-ask + exit-at-bid on a constant book ==
    mid-to-mid move minus the full spread minus fees both sides. So:
        net_pnl(D) = sign(divergence) * dMid(D)  -  round_trip_taker(t)
    where round_trip_taker = spread + taker_fee(ask) + taker_fee(bid). This is the
    same pessimistic-fill bar the settlement book must clear.
  * Autocorrelation: intraday snapshots within a ticker-day are massively correlated,
    and same-day contracts across cities share weather. We resample WHOLE SETTLEMENT
    DAYS (block bootstrap), so the effective sample size is distinct days (~tens), not
    the ~1M overlapping snapshots. This mirrors outcome_tracker._block_bootstrap_edge.

Reads : data/analysis/price_path/panel.parquet   (build_price_path_panel.py first)
Writes: data/analysis/price_path/convergence_summary.csv  (+ a stdout memo)
Run   : .venv/bin/python scripts/price_path_convergence_study.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "analysis" / "price_path" / "panel.parquet"
OUT_CSV = ROOT / "data" / "analysis" / "price_path" / "convergence_summary.csv"

# Forward horizons to test (minutes) and how close a real snapshot must land to t+D.
HORIZONS_MIN = [30, 60, 120, 240]
TOL_MIN = 20

# Only study a genuinely two-sided, tradeable book. Locked/degenerate prices near
# 0 or 1 have no room and no liquidity; they are not where this strategy lives.
BAND_LO, BAND_HI = 0.05, 0.95
MAX_SIGNAL_AGE_MIN = 90     # model view must be reasonably fresh at entry


def block_bootstrap(df: pd.DataFrame, value_col: str, block_col: str = "settlement_date",
                    n_boot: int = 5000, seed: int = 0):
    """Mean of value_col with a 95% CI, resampling whole blocks (settlement days)
    with replacement. Returns (point, lo95, hi95, n_obs, n_blocks)."""
    blocks = [g[value_col].to_numpy() for _, g in df.groupby(block_col, sort=False)]
    blocks = [b for b in blocks if len(b)]
    if not blocks:
        return None
    allv = np.concatenate(blocks)
    point = float(allv.mean())
    rng = random.Random(seed)
    k = len(blocks)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([blocks[rng.randrange(k)] for _ in range(k)])
        means[i] = pick.mean()
    means.sort()
    lo = float(means[int(0.025 * n_boot)])
    hi = float(means[min(int(0.975 * n_boot), n_boot - 1)])
    return point, lo, hi, int(allv.size), k


def forward_mids(panel: pd.DataFrame, horizon_min: int) -> pd.Series:
    """For each row, the SAME ticker's yes_mid at snapshot_utc + horizon (nearest
    logged snapshot within TOL_MIN). NaN if no snapshot lands in the window."""
    book = (panel[["ticker", "snapshot_utc", "yes_mid"]]
            .dropna(subset=["yes_mid"])
            .sort_values("snapshot_utc")
            .rename(columns={"yes_mid": "yes_mid_future",
                             "snapshot_utc": "future_utc"}))
    left = panel[["ticker", "snapshot_utc"]].copy()
    # CRITICAL: pd.merge_asof RESETS the index to a fresh RangeIndex, so we cannot
    # recover row alignment with .reindex(panel.index) afterward (that silently
    # scrambles the join). Carry the original row id through as a column instead.
    left["_oid"] = panel.index
    left["target_utc"] = left["snapshot_utc"] + pd.Timedelta(minutes=horizon_min)
    left = left.sort_values("target_utc")
    merged = pd.merge_asof(
        left, book,
        left_on="target_utc", right_on="future_utc",
        by="ticker", direction="nearest",
        tolerance=pd.Timedelta(minutes=TOL_MIN),
    )
    # Require the matched snapshot to be strictly in the future (not the entry itself).
    ahead = merged["future_utc"] > merged["snapshot_utc"]
    merged.loc[~ahead, "yes_mid_future"] = np.nan
    return merged.set_index("_oid")["yes_mid_future"].reindex(panel.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    panel = panel.reset_index(drop=True)

    # --- Universe: a fresh model view on a tradeable, non-degenerate book. ---
    base = panel[
        panel["has_model"]
        & ~panel["crossed"]
        & panel["yes_mid"].between(BAND_LO, BAND_HI)
        & (panel["signal_age_min"] <= MAX_SIGNAL_AGE_MIN)
        & panel["divergence"].notna()
        & panel["settlement_date"].notna()
    ].copy()

    print("=" * 72)
    print("PRICE-PATH CONVERGENCE STUDY  (Step 2 GO/NO-GO gate)")
    print("=" * 72)
    print(f"panel rows                : {len(panel):,}")
    print(f"study universe            : {len(base):,} "
          f"(fresh model, tradeable book {BAND_LO}-{BAND_HI}, non-crossed)")
    print(f"distinct settlement days  : {base['settlement_date'].nunique()}")
    print(f"distinct tickers          : {base['ticker'].nunique()}")
    print(f"median round-trip (taker) : {base['round_trip_taker'].median():.3f}")
    print(f"median |divergence|       : {base['divergence'].abs().median():.3f}")

    sign = np.sign(base["divergence"]).replace(0, np.nan)
    # Entry gate the live strategy would actually use: only bother when the expected
    # move (proxied by the divergence we could capture if the market fully converged)
    # exceeds the round-trip cost. Reported alongside the unconditional view.
    gated_mask = base["divergence"].abs() > base["round_trip_taker"]

    rows = []
    for h in HORIZONS_MIN:
        fut = forward_mids(base, h)
        dmid = fut - base["yes_mid"]
        dir_move = sign * dmid                      # raw directional move (pre-cost)
        net = dir_move - base["round_trip_taker"]   # pessimistic taker net P&L

        for label, mask in (("all", pd.Series(True, index=base.index)),
                            ("gated|div|>cost", gated_mask)):
            sub = base.loc[mask].copy()
            sub["dir_move"] = dir_move.loc[mask]
            sub["net"] = net.loc[mask]
            sub = sub.dropna(subset=["dir_move", "net"])
            if sub.empty:
                continue

            raw = block_bootstrap(sub, "dir_move", n_boot=args.n_boot)
            npnl = block_bootstrap(sub, "net", n_boot=args.n_boot)
            hit = float((sub["dir_move"] > sub["round_trip_taker"]).mean())
            rows.append({
                "horizon_min": h, "cohort": label,
                "n_obs": npnl[3], "n_days": npnl[4],
                "raw_move": raw[0], "raw_lo": raw[1], "raw_hi": raw[2],
                "net_pnl": npnl[0], "net_lo": npnl[1], "net_hi": npnl[2],
                "pct_move_beats_cost": hit,
            })

    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    def fmt(x):  # cents
        return f"{100 * x:+5.2f}"

    print("\n--- CONVERGENCE (raw directional mid-move toward the model, pre-cost) ---")
    print("  positive => market moves toward the model. cents per contract.")
    print(f"  {'horizon':>8} {'cohort':>16} {'n_days':>6} {'n_obs':>9} "
          f"{'raw_move':>9} {'ci95':>16}")
    for r in rows:
        print(f"  {r['horizon_min']:>6}m {r['cohort']:>16} {r['n_days']:>6} "
              f"{r['n_obs']:>9,} {fmt(r['raw_move']):>9} "
              f"[{fmt(r['raw_lo'])},{fmt(r['raw_hi'])}]")

    print("\n--- NET P&L after round-trip cost (pessimistic taker both sides) ---")
    print("  THE GATE: is the lower CI bound > 0 ? cents per contract.")
    print(f"  {'horizon':>8} {'cohort':>16} {'n_days':>6} {'n_obs':>9} "
          f"{'net_pnl':>9} {'ci95':>16} {'%move>cost':>10}")
    for r in rows:
        flag = "  <== CI>0" if r["net_lo"] > 0 else ""
        print(f"  {r['horizon_min']:>6}m {r['cohort']:>16} {r['n_days']:>6} "
              f"{r['n_obs']:>9,} {fmt(r['net_pnl']):>9} "
              f"[{fmt(r['net_lo'])},{fmt(r['net_hi'])}] "
              f"{100 * r['pct_move_beats_cost']:>9.1f}%{flag}")

    # --- Verdict heuristic ---
    gated = [r for r in rows if r["cohort"] == "gated|div|>cost"]
    best = max(gated, key=lambda r: r["net_lo"]) if gated else None
    print("\n" + "=" * 72)
    if best and best["net_lo"] > 0:
        print(f"VERDICT: GO-ish. Best cohort net P&L CI clears zero at "
              f"{best['horizon_min']}m: {fmt(best['net_pnl'])}c "
              f"[{fmt(best['net_lo'])},{fmt(best['net_hi'])}]. "
              f"Proceed to Steps 3-4 (MTM engine + exit bake-off) and a live "
              f"exit-fill sanity check before any careful trading.")
    elif best and best["net_pnl"] > 0:
        print(f"VERDICT: MARGINAL. Positive mean net P&L "
              f"({fmt(best['net_pnl'])}c at {best['horizon_min']}m) but the CI "
              f"crosses zero. Not dead, but not a clean GO — Step 3/4 exit modeling "
              f"decides whether a smarter exit than fixed-horizon rescues it.")
    else:
        print("VERDICT: NO-GO on this evidence. Realized moves do not clear the "
              "round trip even at the best horizon. Strategy stops here — an "
              "afternoon spent, settlement book unaffected.")
    print(f"(wrote {OUT_CSV})")
    print("=" * 72)


if __name__ == "__main__":
    main()
