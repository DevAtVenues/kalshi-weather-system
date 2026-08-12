"""
Stress test for price-path Steps 1-2 — decide whether the GO/NO-GO "GO-ish" survives
honest scrutiny before we build the exit machinery (Steps 3-4).
See docs/STATE_OF_THE_MODEL.md Part V and memory project_pricepath_step12.

The Step-2 study used a shortcut: net = sign(div)*dMid - entry_round_trip, which
ASSUMES the exit spread equals the entry spread. Here we drop that assumption and
compute TRUE executable fills against the actual future book:

  BUY_YES (divergence>0): enter by lifting the ask -> yes_ask(t);
                          exit by hitting the future bid -> yes_bid(t+D).
      pnl = yes_bid(t+D) - yes_ask(t) - taker_fee(yes_ask) - taker_fee(yes_bid_fut)
  BUY_NO  (divergence<0): buy NO at (1 - yes_bid(t)); exit sell NO at (1 - yes_ask(t+D)).
      pnl = yes_bid(t) - yes_ask(t+D) - fees        (algebra of the NO side)

That is the pessimistic-taker bar the settlement book must clear. We also report an
optimistic maker bracket (rest a bid in, rest an offer out — capture the spread).

Tests run:
  T0  Step-1 validation: orderbook-derived yes_mid vs the logger's own market_mid.
  T1  Executable pessimistic net P&L by horizon (all vs gated) with day-block CI.
  T2  Adverse selection: executable net split by hours-to-settle.
  T3  Depth realism: require real size at BOTH touches, recompute.
  T4  Capturability: does the BID (what we sell into) actually move, or just the ask?

Reads : data/analysis/price_path/panel.parquet
Run   : .venv/bin/python scripts/price_path_stress_test.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_weather.backtest import maker_fee, taker_fee

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "analysis" / "price_path" / "panel.parquet"

HORIZONS_MIN = [30, 60, 120]
TOL_MIN = 20
BAND_LO, BAND_HI = 0.05, 0.95
MAX_AGE = 90
N_BOOT = 5000


def block_ci(df, col, block="settlement_date", n_boot=N_BOOT, seed=0):
    blocks = [g[col].to_numpy() for _, g in df.groupby(block, sort=False)]
    blocks = [b for b in blocks if len(b)]
    if not blocks:
        return None
    allv = np.concatenate(blocks)
    rng = random.Random(seed)
    k = len(blocks)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = np.concatenate([blocks[rng.randrange(k)] for _ in range(k)]).mean()
    means.sort()
    return (float(allv.mean()), float(means[int(0.025 * n_boot)]),
            float(means[min(int(0.975 * n_boot), n_boot - 1)]), int(allv.size), k)


def forward_book(panel, horizon_min):
    """Future yes_bid / yes_ask for the same ticker at snapshot+horizon (nearest
    within TOL_MIN, strictly ahead). Returns a frame aligned to panel.index."""
    book = (panel[["ticker", "snapshot_utc", "yes_bid", "yes_ask"]]
            .dropna(subset=["yes_bid", "yes_ask"])
            .sort_values("snapshot_utc")
            .rename(columns={"snapshot_utc": "future_utc",
                             "yes_bid": "bid_fut", "yes_ask": "ask_fut"}))
    left = panel[["ticker", "snapshot_utc"]].copy()
    # CRITICAL: pd.merge_asof resets the index; carry the original row id as a column
    # so we can realign, rather than the broken .reindex(panel.index) after the reset.
    left["_oid"] = panel.index
    left["target_utc"] = left["snapshot_utc"] + pd.Timedelta(minutes=horizon_min)
    left = left.sort_values("target_utc")
    m = pd.merge_asof(left, book, left_on="target_utc", right_on="future_utc",
                      by="ticker", direction="nearest",
                      tolerance=pd.Timedelta(minutes=TOL_MIN))
    m.loc[~(m["future_utc"] > m["snapshot_utc"]), ["bid_fut", "ask_fut"]] = np.nan
    return m.set_index("_oid").reindex(panel.index)[["bid_fut", "ask_fut"]]


def executable_pnl(row_ask, row_bid, ask_fut, bid_fut, is_yes):
    """Pessimistic taker P&L per contract, on the 0-1 scale. is_yes: bool Series."""
    fee = lambda p: p.map(taker_fee)
    # BUY_YES: buy ask now, sell future bid.
    pnl_yes = bid_fut - row_ask - fee(row_ask) - fee(bid_fut)
    # BUY_NO: buy NO at (1-yes_bid) now, sell NO at (1-yes_ask_fut) => yes_bid - ask_fut.
    no_entry = 1 - row_bid
    no_exit = 1 - ask_fut
    pnl_no = (no_exit - no_entry) - fee(no_entry) - fee(no_exit)
    return pnl_yes.where(is_yes, pnl_no)


def optimistic_pnl(row_ask, row_bid, ask_fut, bid_fut, is_yes):
    """Optimistic maker bracket: rest a bid in, rest an offer out (capture spread),
    maker fees only. OPTIMISTIC because it assumes both resting orders fill —
    charter: edge that survives only this model is NOT edge."""
    fee = lambda p: p.map(maker_fee)
    # BUY_YES maker: buy at bid now, sell at ask future.
    pnl_yes = ask_fut - row_bid - fee(row_bid) - fee(ask_fut)
    # BUY_NO maker: buy NO at (1-yes_ask) now, sell NO at (1-yes_bid_fut)
    #   = row_ask - bid_fut on the YES scale.
    pnl_no = row_ask - bid_fut - fee(1 - row_ask) - fee(1 - bid_fut)
    return pnl_yes.where(is_yes, pnl_no)


def c(x):
    return f"{100 * x:+5.2f}"


def main():
    panel = pd.read_parquet(PANEL).reset_index(drop=True)

    # ---------- T0: Step-1 validation ----------
    print("=" * 74)
    print("T0  STEP-1 VALIDATION: orderbook yes_mid  vs  logger market_mid")
    print("=" * 74)
    v = panel[panel["has_model"] & ~panel["crossed"]
              & panel["market_mid"].notna() & panel["yes_mid"].notna()
              & (panel["signal_age_min"] <= 10)]  # fresh: book ~ contemporaneous w/ signal
    diff = (v["yes_mid"] - v["market_mid"]).abs()
    print(f"  n compared (signal age <=10m): {len(v):,}")
    print(f"  corr(orderbook_mid, logger_mid): {v['yes_mid'].corr(v['market_mid']):.4f}")
    print(f"  median |diff|: {diff.median():.4f}   p90 |diff|: {diff.quantile(.9):.4f}")
    print(f"  min signal_age_min (must be >=0, no leakage): "
          f"{panel['signal_age_min'].min():.2f}")

    # ---------- universe ----------
    base = panel[
        panel["has_model"] & ~panel["crossed"]
        & panel["yes_mid"].between(BAND_LO, BAND_HI)
        & (panel["signal_age_min"] <= MAX_AGE)
        & panel["divergence"].notna() & panel["settlement_date"].notna()
    ].copy()
    base["is_yes"] = base["divergence"] > 0
    gate = base["divergence"].abs() > base["round_trip_taker"]

    print("\n" + "=" * 74)
    print("T1  EXECUTABLE pessimistic-taker net P&L (buy ask now, sell FUTURE bid)")
    print("     -- no constant-spread shortcut. cents/contract, day-block 95% CI.")
    print("=" * 74)
    print(f"  {'hz':>4} {'cohort':>14} {'days':>4} {'n':>8} "
          f"{'PESS net':>9} {'ci95':>16} {'OPT net':>9}")
    per_h = {}
    for h in HORIZONS_MIN:
        fb = forward_book(base, h)
        pess = executable_pnl(base["yes_ask"], base["yes_bid"],
                              fb["ask_fut"], fb["bid_fut"], base["is_yes"])
        opt = optimistic_pnl(base["yes_ask"], base["yes_bid"],
                             fb["ask_fut"], fb["bid_fut"], base["is_yes"])
        d = base.assign(pess=pess, opt=opt, bid_fut=fb["bid_fut"],
                        ask_fut=fb["ask_fut"]).dropna(subset=["pess"])
        per_h[h] = d
        for label, mask in (("all", pd.Series(True, index=d.index)),
                            ("gated", gate.reindex(d.index).fillna(False))):
            sub = d.loc[mask]
            if sub.empty:
                continue
            ci = block_ci(sub, "pess")
            oci = block_ci(sub, "opt")
            flag = " <==CI>0" if ci[1] > 0 else ""
            print(f"  {h:>3}m {label:>14} {ci[4]:>4} {ci[3]:>8,} "
                  f"{c(ci[0]):>9} [{c(ci[1])},{c(ci[2])}] {c(oci[0]):>9}{flag}")

    print("\n" + "=" * 74)
    print("T2  ADVERSE SELECTION: executable pess net by hours-to-settle (60m, gated)")
    print("     -- is the edge only in the <6h danger zone (settlement resolving)?")
    print("=" * 74)
    d = per_h[60]
    dg = d.loc[gate.reindex(d.index).fillna(False)].copy()
    dg["htier"] = pd.cut(dg["hours_to_settle"], [-1e9, 6, 12, 24, 1e9],
                         labels=["<6h", "6-12h", "12-24h", ">24h"])
    print(f"  {'tier':>8} {'days':>4} {'n':>8} {'PESS net':>9} {'ci95':>16}")
    for tier, sub in dg.groupby("htier", observed=True):
        if len(sub) < 20:
            continue
        ci = block_ci(sub, "pess")
        flag = " <==CI>0" if ci[1] > 0 else ""
        print(f"  {tier:>8} {ci[4]:>4} {ci[3]:>8,} {c(ci[0]):>9} "
              f"[{c(ci[1])},{c(ci[2])}]{flag}")

    print("\n" + "=" * 74)
    print("T3  DEPTH REALISM: require >=25 contracts at BOTH touches (60m, gated)")
    print("=" * 74)
    deep = dg[(dg["yes_ask_depth"] >= 25) & (dg["yes_bid_depth"] >= 25)]
    if len(deep) >= 20:
        ci = block_ci(deep, "pess")
        print(f"  n={ci[3]:,} days={ci[4]}  PESS net {c(ci[0])} "
              f"[{c(ci[1])},{c(ci[2])}]  ({100*len(deep)/len(dg):.0f}% of gated survive depth>=25)")
    else:
        print(f"  too few deep-book entries ({len(deep)}) to bootstrap.")

    print("\n" + "=" * 74)
    print("T4  CAPTURABILITY: of the mid-move, how much shows up in the BID we sell into?")
    print("=" * 74)
    d = per_h[60].copy()
    dgg = d.loc[gate.reindex(d.index).fillna(False)]
    dmid = (dgg["bid_fut"] + dgg["ask_fut"]) / 2 - dgg["yes_mid"]
    dbid = dgg["bid_fut"] - dgg["yes_bid"]
    sgn = np.where(dgg["is_yes"], 1, -1)
    # Means, not medians: the median snapshot sees zero move (books mostly sit
    # still over 60m), so a median ratio is 0/0. The mean is the tradeable claim.
    mean_mid, mean_bid = float(np.mean(sgn * dmid)), float(np.mean(sgn * dbid))
    print(f"  mean directional mid-move : {c(mean_mid)}c   "
          f"(median {c(float(np.median(sgn * dmid)))}c — most snapshots don't move)")
    print(f"  mean directional bid-move : {c(mean_bid)}c  "
          f"(the part a resting exit actually captures)")
    if abs(mean_mid) > 1e-9:
        print(f"  bid-move / mid-move ratio : {mean_bid / mean_mid:.2f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
