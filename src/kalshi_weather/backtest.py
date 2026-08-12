"""
One-decision-per-contract backtest loop for Kalshi weather markets.

PRE-REGISTERED PARAMETERS — do not change after any backtest result has
been recorded. Commit methodology before touching the test set.
Parameters registered: 2026-06-03

Decision-time prices now use the last candle before the settlement window opens
(pre-settlement overnight price) rather than the first candle at market open.
This removes the ATM/OOT spread bias that was noted in earlier runs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

# ── Pre-registered parameters ─────────────────────────────────────────────────
EDGE_THRESHOLD       = 0.05    # |prob - market_price| must exceed this to trade
ASSUMED_HALF_SPREAD  = 0.01    # worst-case half-spread added to pessimistic fills
TAKER_SLIPPAGE       = 0.003   # extra slippage on taker fills (pessimistic model)
MAKER_SLIPPAGE       = 0.001   # extra slippage on maker fills (optimistic model)
BLOCK_BOOTSTRAP_DAYS = 30      # days per bootstrap block (captures heat-wave autocorr)
N_BOOTSTRAP          = 2_000   # bootstrap iterations
# ─────────────────────────────────────────────────────────────────────────────


def taker_fee(p: float) -> float:
    """Kalshi taker fee: 7% × p × (1-p). Symmetric across YES/NO sides."""
    return 0.07 * float(p) * (1.0 - float(p))


def maker_fee(p: float) -> float:
    return taker_fee(p) * 0.25


class Rule(Protocol):
    def __call__(self, row: pd.Series) -> float:
        """
        Return P(YES) ∈ (0, 1), or float('nan') to abstain.

        Safe columns: settlement_date, contract_type, threshold_f, t_direction,
                      tmax_f_fcst, tmin_f_fcst, high_normal, market_price.
        FORBIDDEN (look-ahead): actual_high_f, result, label_consistent.
        """
        ...


@dataclass
class BacktestResult:
    trades: pd.DataFrame     # one row per traded contract
    estimates: pd.DataFrame  # one row per contract where rule returned non-NaN (for calibration)
    metrics_pess: dict       # metrics under pessimistic fill model
    metrics_opt: dict        # metrics under optimistic fill model
    rule_name: str = "unnamed"

    def summary(self) -> None:
        _print_summary(self)


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    aligned: pd.DataFrame,
    rule: Rule,
    edge_threshold: float = EDGE_THRESHOLD,
    rule_name: str = "unnamed",
    verbose: bool = True,
    atm_min: float = 0.0,
    atm_max: float = 1.0,
) -> BacktestResult:
    """
    For each settled contract in aligned (chronological):
      1. rule(row) → prob_estimate (or NaN → skip)
      2. edge = |prob - market_price|; pick direction with positive edge
      3. If edge ≥ edge_threshold: record trade under both fill models
    """
    df = aligned.copy()
    if "settlement_date" in df.columns:
        df = df.sort_values("settlement_date").reset_index(drop=True)

    if atm_min > 0.0 or atm_max < 1.0:
        price_col = df["last_price_dollars"] if "last_price_dollars" in df.columns else pd.Series(dtype=float)
        atm_mask = price_col.between(atm_min, atm_max)
        df = df[atm_mask].reset_index(drop=True)
        if verbose:
            print(f"  ATM filter [{atm_min:.2f}, {atm_max:.2f}]: {len(df)} of {len(aligned)} contracts")

    trade_rows: list[dict] = []
    estimate_rows: list[dict] = []

    for _, row in df.iterrows():
        result = row.get("result")
        if not isinstance(result, str) or result not in ("yes", "no"):
            continue

        prob = rule(row)
        if not np.isfinite(prob):
            continue

        market_price = _safe_float(row.get("last_price_dollars"))
        if not np.isfinite(market_price):
            continue

        prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
        outcome = 1 if result == "yes" else 0

        estimate_rows.append({
            "ticker":          row.get("ticker"),
            "settlement_date": row.get("settlement_date"),
            "prob_estimate":   prob,
            "outcome":         outcome,
            "market_price":    market_price,
        })

        # Edge: positive for whichever side we should take
        # edge_buy_yes = prob - market_price
        # edge_buy_no  = (1 - prob) - (1 - market_price) = -(edge_buy_yes)
        edge_signed = prob - market_price
        if edge_signed >= 0:
            direction = "buy_yes"
            edge_raw  = edge_signed
        else:
            direction = "buy_no"
            edge_raw  = -edge_signed

        if edge_raw < edge_threshold:
            continue

        # Fill prices
        bid = _safe_float(row.get("yes_bid_dollars"))
        ask = _safe_float(row.get("yes_ask_dollars"))
        mid = ((bid + ask) / 2) if (np.isfinite(bid) and np.isfinite(ask)) else market_price

        if direction == "buy_yes":
            yes_fill_pess = float(np.clip((ask if np.isfinite(ask) else mid) + ASSUMED_HALF_SPREAD, 0.01, 0.99))
            yes_fill_opt  = float(np.clip(mid, 0.01, 0.99))
            fee_pess = taker_fee(yes_fill_pess) + TAKER_SLIPPAGE
            fee_opt  = maker_fee(yes_fill_opt)  + MAKER_SLIPPAGE
            pnl_pess = float(outcome)   - yes_fill_pess - fee_pess
            pnl_opt  = float(outcome)   - yes_fill_opt  - fee_opt
            fill_pess, fill_opt = yes_fill_pess, yes_fill_opt
        else:
            # Buy NO: cost = 1 - yes_bid (NO ask = 1 - YES bid)
            no_ask = 1.0 - (bid if np.isfinite(bid) else mid)
            no_fill_pess = float(np.clip(no_ask + ASSUMED_HALF_SPREAD, 0.01, 0.99))
            no_fill_opt  = float(np.clip(1.0 - mid, 0.01, 0.99))
            fee_pess = taker_fee(no_fill_pess) + TAKER_SLIPPAGE
            fee_opt  = maker_fee(no_fill_opt)  + MAKER_SLIPPAGE
            no_outcome = 1 - outcome
            pnl_pess = float(no_outcome) - no_fill_pess - fee_pess
            pnl_opt  = float(no_outcome) - no_fill_opt  - fee_opt
            fill_pess, fill_opt = no_fill_pess, no_fill_opt

        trade_rows.append({
            "ticker":          row.get("ticker"),
            "settlement_date": row.get("settlement_date"),
            "contract_type":   row.get("contract_type"),
            "threshold_f":     row.get("threshold_f"),
            "direction":       direction,
            "prob_estimate":   prob,
            "market_price":    market_price,
            "edge_raw":        edge_raw,
            "fill_pess":       fill_pess,
            "fill_opt":        fill_opt,
            "fee_pess":        fee_pess,
            "fee_opt":         fee_opt,
            "result":          result,
            "outcome":         outcome,
            "pnl_pess":        pnl_pess,
            "pnl_opt":         pnl_opt,
        })

    trades    = pd.DataFrame(trade_rows)
    estimates = pd.DataFrame(estimate_rows)

    metrics_pess = _compute_metrics(trades, estimates, pnl_col="pnl_pess")
    metrics_opt  = _compute_metrics(trades, estimates, pnl_col="pnl_opt")

    result_obj = BacktestResult(
        trades=trades,
        estimates=estimates,
        metrics_pess=metrics_pess,
        metrics_opt=metrics_opt,
        rule_name=rule_name,
    )

    if verbose:
        _print_summary(result_obj)

    return result_obj


# ── Metrics ───────────────────────────────────────────────────────────────────

def _safe_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _compute_metrics(
    trades: pd.DataFrame,
    estimates: pd.DataFrame,
    pnl_col: str,
) -> dict:
    n_est = len(estimates)
    brier = float(np.mean((estimates["prob_estimate"].values - estimates["outcome"].values) ** 2)) if n_est else float("nan")

    if trades.empty:
        return {"n_estimates": n_est, "n_trades": 0, "brier_score": brier}

    pnl   = trades[pnl_col].to_numpy(dtype=float)
    dates = trades["settlement_date"].to_numpy()

    cumulative  = np.cumsum(pnl)
    peak        = np.maximum.accumulate(cumulative)
    max_drawdown = float((cumulative - peak).min())

    ci_low, ci_high = _block_bootstrap_ci(pnl, dates)
    n_eff = _effective_n(pnl)

    return {
        "n_estimates":  n_est,
        "n_trades":     int(len(trades)),
        "brier_score":  round(brier, 5),
        "win_rate":     round(float((pnl > 0).mean()), 4),
        "total_return": round(float(pnl.sum()), 4),
        "avg_return":   round(float(pnl.mean()), 5),
        "max_drawdown": round(max_drawdown, 4),
        "edge_mean":    round(float(trades["edge_raw"].mean()), 4),
        "edge_ci_low":  round(ci_low, 4),
        "edge_ci_high": round(ci_high, 4),
        "effective_n":  n_eff,
    }


def _block_bootstrap_ci(
    pnl: np.ndarray,
    dates: np.ndarray,
    block_days: int = BLOCK_BOOTSTRAP_DAYS,
    n_iter: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """
    Block bootstrap CI on mean P&L per trade.
    Blocks are formed from consecutive settlement dates to capture
    heat-wave / cold-snap autocorrelation across adjacent days.
    """
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    if n_dates < 2:
        m = float(pnl.mean())
        return m, m

    # Map date → pnl indices
    date_to_idx: dict = defaultdict(list)
    for i, d in enumerate(dates):
        date_to_idx[d].append(i)

    rng = np.random.default_rng(42)
    boot_means: list[float] = []

    for _ in range(n_iter):
        n_blocks = max(1, n_dates // block_days)
        starts = rng.integers(0, n_dates, size=n_blocks)
        sampled: list[int] = []
        for s in starts:
            for k in range(block_days):
                idx = int(s + k) % n_dates
                for trade_idx in date_to_idx[unique_dates[idx]]:
                    sampled.append(trade_idx)
        if sampled:
            boot_means.append(pnl[sampled].mean())

    if not boot_means:
        return float("nan"), float("nan")

    arr = np.array(boot_means)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _effective_n(pnl: np.ndarray) -> int:
    """Rough effective sample size via lag-1 autocorrelation adjustment."""
    n = len(pnl)
    if n < 4:
        return n
    centered = pnl - pnl.mean()
    var = np.mean(centered ** 2)
    if var < 1e-12:
        return n
    lag1 = float(np.mean(centered[1:] * centered[:-1]) / var)
    lag1 = np.clip(lag1, -0.99, 0.99)
    return max(1, int(n / (1.0 + 2.0 * max(0.0, lag1))))


def reliability_curve(estimates: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute reliability diagram data: predicted probability vs actual frequency.
    Returns a DataFrame with columns: bin_center, mean_pred, mean_actual, count.
    Bins with fewer than 5 observations are dropped.
    """
    probs   = estimates["prob_estimate"].to_numpy()
    outcomes = estimates["outcome"].to_numpy()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < 5:
            continue
        rows.append({
            "bin_center":  round((lo + hi) / 2, 3),
            "mean_pred":   round(float(probs[mask].mean()), 4),
            "mean_actual": round(float(outcomes[mask].mean()), 4),
            "count":       int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_summary(result: BacktestResult) -> None:
    p, o = result.metrics_pess, result.metrics_opt
    print(f"\n{'═'*60}")
    print(f"  Backtest: {result.rule_name}")
    print(f"{'═'*60}")
    print(f"  Estimates (all non-NaN):     {p.get('n_estimates', 0)}")
    print(f"  Brier score:                 {p.get('brier_score', '—')}")
    print()
    print(f"  {'Metric':<25} {'Pessimistic':>12} {'Optimistic':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for key, label in [
        ("n_trades",    "Trades"),
        ("win_rate",    "Win rate"),
        ("total_return","Total return ($)"),
        ("avg_return",  "Avg return/trade ($)"),
        ("max_drawdown","Max drawdown ($)"),
        ("edge_mean",   "Mean edge"),
        ("edge_ci_low", "Edge CI low (95%)"),
        ("edge_ci_high","Edge CI high (95%)"),
        ("effective_n", "Effective N"),
    ]:
        pv = p.get(key, "—")
        ov = o.get(key, "—")
        print(f"  {label:<25} {str(pv):>12} {str(ov):>12}")
    print(f"{'═'*60}\n")

    if not result.estimates.empty:
        rel = reliability_curve(result.estimates)
        if not rel.empty:
            print("  Reliability curve (predicted → actual):")
            print(f"  {'Bin':>6} {'Pred':>8} {'Actual':>8} {'N':>6}")
            for _, r in rel.iterrows():
                calib_flag = " ←" if abs(r["mean_pred"] - r["mean_actual"]) > 0.10 else ""
                print(f"  {r['bin_center']:>6.2f} {r['mean_pred']:>8.3f} {r['mean_actual']:>8.3f} {r['count']:>6}{calib_flag}")
            print()
