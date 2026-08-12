"""
Step 3 — mark-to-market engine for the price-path book (DORMANT PILOT).

Opens simulated positions from divergence signals on the Step-1 panel, walks
each position forward along the same ticker's logged book, closes it under an
explicit exit policy, and prices the round trip under BOTH fill models
(charter: edge that survives only the optimistic model is not edge).

Entry rule (identical to the Step-2 gated cohort):
    fresh model view (age <= 90m), two-sided non-crossed book, mid in
    [0.05, 0.95], and |divergence| > round_trip_taker. One position per
    (ticker, direction, settlement_date) — the first qualifying snapshot.

Fill models per trade:
    pessimistic  taker both ways: BUY_YES enters at the ask, exits at the
                 future bid (BUY_NO mirrors on the NO side); taker fees both
                 sides. The bar any activation decision must clear.
    optimistic   maker at mid both ways, maker fees. Reported for the bracket,
                 never for the verdict.

Exit policies are pure functions over the position's forward path (mid-based
triggers, strictly after entry — no look-ahead by construction). A policy that
never triggers exits at the ticker's LAST logged snapshot ("end of book", the
hold-to-settlement proxy that stays inside the panel).

No network, no orders, no imports from live/exchange or live/executor — see
package docstring for the guardrails.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from kalshi_weather.backtest import maker_fee, taker_fee

BAND_LO, BAND_HI = 0.05, 0.95
MAX_SIGNAL_AGE_MIN = 90.0


# ── exit policies ─────────────────────────────────────────────────────────────
# Each policy: (path, entry) -> positional index into `path` of the exit
# snapshot, or None for "never triggered" (=> exit at the last snapshot).
# `path` is the ticker's snapshots strictly AFTER the entry snapshot, ordered;
# `entry` is the entry row. Triggers read mids only (what we can observe);
# fills are then priced at the exit snapshot's bid/ask by the engine.

def _fixed_horizon(minutes: float) -> "ExitPolicy":
    def fn(path: pd.DataFrame, entry: pd.Series) -> int | None:
        target = entry["snapshot_utc"] + pd.Timedelta(minutes=minutes)
        hit = np.nonzero((path["snapshot_utc"] >= target).to_numpy())[0]
        return int(hit[0]) if len(hit) else None
    return ExitPolicy(f"fixed_{int(minutes)}m", fn)


def _target_model(path: pd.DataFrame, entry: pd.Series) -> int | None:
    """Convergence complete: the market mid reaches the model probability that
    justified the entry. The canonical 'sell into the rise' exit."""
    p = float(entry["prob_estimate"])
    mids = path["yes_mid"].to_numpy()
    hit = np.nonzero(mids >= p)[0] if entry["divergence"] > 0 else \
        np.nonzero(mids <= p)[0]
    return int(hit[0]) if len(hit) else None


def _trailing_peak(retrace: float) -> "ExitPolicy":
    def fn(path: pd.DataFrame, entry: pd.Series) -> int | None:
        sgn = 1.0 if entry["divergence"] > 0 else -1.0
        fav = sgn * (path["yes_mid"].to_numpy() - float(entry["yes_mid"]))
        best = np.maximum.accumulate(fav)
        hit = np.nonzero((best - fav) >= retrace)[0]
        return int(hit[0]) if len(hit) else None
    return ExitPolicy(f"trail_{int(round(100 * retrace))}c", fn)


def _stop_loss(loss: float) -> "ExitPolicy":
    """Included for completeness of the bake-off; the timing study showed stops
    destroy this family of edge — expected to grade WORST, and its presence in
    the results is the empirical receipt for the hold-to-settlement design."""
    def fn(path: pd.DataFrame, entry: pd.Series) -> int | None:
        sgn = 1.0 if entry["divergence"] > 0 else -1.0
        fav = sgn * (path["yes_mid"].to_numpy() - float(entry["yes_mid"]))
        hit = np.nonzero(fav <= -loss)[0]
        return int(hit[0]) if len(hit) else None
    return ExitPolicy(f"stop_{int(round(100 * loss))}c", fn)


def _hold_to_end(path: pd.DataFrame, entry: pd.Series) -> int | None:
    return None                     # never triggers => engine exits at the end


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    fn: Callable[[pd.DataFrame, pd.Series], int | None]


POLICIES: dict[str, ExitPolicy] = {p.name: p for p in (
    _fixed_horizon(30), _fixed_horizon(60), _fixed_horizon(120),
    _fixed_horizon(240),
    ExitPolicy("target_model", _target_model),
    _trailing_peak(0.03),
    _stop_loss(0.05),
    ExitPolicy("hold_to_end", _hold_to_end),
)}


# ── P&L ───────────────────────────────────────────────────────────────────────

def trade_pnl(entry: pd.Series, exit_row: pd.Series) -> tuple[float, float]:
    """(pessimistic, optimistic) net P&L per contract on the 0-1 scale."""
    is_yes = entry["divergence"] > 0
    b0, a0 = float(entry["yes_bid"]), float(entry["yes_ask"])
    be, ae = float(exit_row["yes_bid"]), float(exit_row["yes_ask"])
    m0 = (b0 + a0) / 2.0
    me = (be + ae) / 2.0
    if is_yes:
        pess = be - a0 - taker_fee(a0) - taker_fee(be)
        opt = me - m0 - maker_fee(m0) - maker_fee(me)
    else:  # BUY_NO: buy NO at (1-bid), sell NO at (1-ask_future)
        pess = b0 - ae - taker_fee(1 - b0) - taker_fee(1 - ae)
        opt = m0 - me - maker_fee(1 - m0) - maker_fee(1 - me)
    return pess, opt


# ── engine ────────────────────────────────────────────────────────────────────

def gated_entries(panel: pd.DataFrame) -> pd.DataFrame:
    """First qualifying snapshot per (ticker, direction, settlement_date)."""
    u = panel[
        panel["has_model"] & ~panel["crossed"]
        & panel["yes_mid"].between(BAND_LO, BAND_HI)
        & (panel["signal_age_min"] <= MAX_SIGNAL_AGE_MIN)
        & panel["divergence"].notna() & panel["settlement_date"].notna()
        & (panel["divergence"].abs() > panel["round_trip_taker"])
    ].copy()
    if u.empty:
        return u
    u["direction"] = np.where(u["divergence"] > 0, "BUY_YES", "BUY_NO")
    u = u.sort_values("snapshot_utc")
    return u.drop_duplicates(["ticker", "direction", "settlement_date"],
                             keep="first")


def simulate(panel: pd.DataFrame, policy: ExitPolicy,
             entries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run every gated entry through `policy`. Returns one row per trade with
    both fill models, exit reason, and holding time."""
    if entries is None:
        entries = gated_entries(panel)
    books = {t: g.sort_values("snapshot_utc").reset_index(drop=True)
             for t, g in panel.groupby("ticker", sort=False)}
    rows = []
    for _, e in entries.iterrows():
        book = books.get(e["ticker"])
        if book is None:
            continue
        path = book[book["snapshot_utc"] > e["snapshot_utc"]]
        path = path[path["yes_bid"].notna() & path["yes_ask"].notna()]
        if path.empty:
            continue                # entered on the ticker's last print: no trade
        path = path.reset_index(drop=True)
        idx = policy.fn(path, e)
        reason = policy.name if idx is not None else "end_of_book"
        x = path.iloc[idx if idx is not None else len(path) - 1]
        pess, opt = trade_pnl(e, x)
        rows.append({
            "ticker": e["ticker"], "direction": e["direction"],
            "settlement_date": str(e["settlement_date"]),
            "entry_utc": e["snapshot_utc"].isoformat(),
            "exit_utc": x["snapshot_utc"].isoformat(),
            "held_min": float((x["snapshot_utc"] - e["snapshot_utc"])
                              .total_seconds() / 60.0),
            "divergence": float(e["divergence"]),
            "policy": policy.name, "exit_reason": reason,
            "pnl_pess": float(pess), "pnl_opt": float(opt),
        })
    return pd.DataFrame(rows)


def day_block_ci(df: pd.DataFrame, col: str, n_boot: int = 5000,
                 seed: int = 0) -> tuple[float, float, float, int, int] | None:
    """(mean, lo95, hi95, n_obs, n_days) resampling whole settlement days —
    the same convention as outcome_tracker / the Step-2 study."""
    blocks = [g[col].to_numpy() for _, g in df.groupby("settlement_date",
                                                       sort=False)]
    blocks = [b for b in blocks if len(b)]
    if not blocks:
        return None
    allv = np.concatenate(blocks)
    rng = random.Random(seed)
    k = len(blocks)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = np.concatenate(
            [blocks[rng.randrange(k)] for _ in range(k)]).mean()
    means.sort()
    return (float(allv.mean()), float(means[int(0.025 * n_boot)]),
            float(means[min(int(0.975 * n_boot), n_boot - 1)]),
            int(allv.size), k)
