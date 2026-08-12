"""
Baseline trading rules for the Kalshi weather backtester.

Each rule is a callable (pd.Series -> float) that returns P(YES) ∈ (0, 1)
or float('nan') to abstain. See backtest.Rule protocol.

Safe columns available: settlement_date, contract_type, threshold_f,
                         t_direction, tmax_f_fcst, tmin_f_fcst, high_normal,
                         market_price.
FORBIDDEN (look-ahead):  actual_high_f, result, label_consistent.
"""
from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd


def make_climo_rule(
    labels: pd.DataFrame,
    train_years: range | list[int],
    min_samples: int = 5,
    rule_name: str = "climo_baseline",
):
    """
    Build a climatological baseline rule from historical label data.

    For each (month, day) combination, the empirical fraction of historical
    daily-high values that met or exceeded a given threshold is used as P(YES).
    No forecast data is required — this is a pure climatological prior.

    Parameters
    ----------
    labels      : DataFrame from ingest.labels.fetch_labels(); must include
                  columns: date (Python date), high (°F), station.
    train_years : years to include in the training distribution.
    min_samples : minimum historical observations required to return a
                  non-NaN probability. Dates/thresholds with fewer data
                  points abstain (return NaN).

    Returns
    -------
    rule : callable matching the backtest.Rule protocol.
    """
    year_set = set(train_years)
    train = labels[labels["date"].apply(lambda d: d.year in year_set)].copy()

    # Build (month, day) → sorted array of historical tmax values
    md_hist: dict[tuple[int, int], list[float]] = {}
    for _, row in train.iterrows():
        d = row["date"]
        if isinstance(d, date) and pd.notna(row.get("high")):
            key = (d.month, d.day)
            md_hist.setdefault(key, []).append(float(row["high"]))

    md_sorted: dict[tuple[int, int], np.ndarray] = {
        k: np.sort(v) for k, v in md_hist.items() if len(v) >= min_samples
    }

    def rule(row: pd.Series) -> float:
        d = row.get("settlement_date")
        threshold = row.get("threshold_f")
        direction = row.get("t_direction")
        if d is None or threshold is None or pd.isna(threshold):
            return float("nan")
        # settlement_date may be a Python date or a Timestamp
        if hasattr(d, "date"):
            d = d.date()
        key = (d.month, d.day)
        hist = md_sorted.get(key)
        if hist is None:
            return float("nan")
        # "greater": YES if tmax >= threshold; "less": YES if tmax < threshold
        if direction == "less":
            return float(np.mean(hist < float(threshold)))
        return float(np.mean(hist >= float(threshold)))

    rule.__name__ = rule_name
    return rule


def make_forecast_rule(
    labels: pd.DataFrame,
    forecasts: pd.DataFrame,
    train_years: list[int],
    station: str = "KNYC",
    model: str = "gfs_seamless",
    min_samples: int = 20,
    distribution: str = "gaussian",
    rule_name: str = "forecast_bias_corrected",
):
    """
    Build a GFS-forecast-informed rule using historical bias correction.

    For each calendar month, we fit a distribution of residuals
    (actual_high − GFS_tmax_forecast) from the training years. Given a
    new contract, we read off the tail probability:

        P(actual_high ≥ T | GFS_tmax) = P(residual ≥ T − GFS_tmax)

    Parameters
    ----------
    labels       : from ingest.labels.fetch_labels(); must have date, high, station.
    forecasts    : from ingest.forecasts.fetch_forecasts(); must have date,
                   tmax_f_fcst, model.
    train_years  : years to use for bias estimation (must precede the test window).
    station      : NWS station to filter labels (default: KNYC).
    model        : Open-Meteo model to use (default: gfs_seamless).
    min_samples  : min observations per month; falls back to global when below.
    distribution : "gaussian" (default) fits N(mu, sigma) per month — proper
                   tail behaviour with limited training data. "empirical" uses
                   the raw ECDF — exact for in-distribution gaps but assigns
                   probability 0/1 beyond the training range.
    """
    from scipy import stats as scipy_stats

    year_set = set(train_years)

    # ── Filter labels ──────────────────────────────────────────────────────────
    lbl = labels.copy()
    if "station" in lbl.columns:
        lbl = lbl[lbl["station"] == station]
    lbl["date"] = lbl["date"].apply(
        lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
    )
    lbl = lbl[lbl["date"].apply(lambda d: d.year in year_set)]
    lbl = lbl[lbl["high"].notna()][["date", "high"]].copy()

    # ── Filter forecasts ───────────────────────────────────────────────────────
    fcst = forecasts[forecasts["model"] == model].copy()
    fcst["date"] = fcst["date"].apply(
        lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
    )
    fcst = fcst[fcst["date"].apply(lambda d: d.year in year_set)]
    fcst = fcst[fcst["tmax_f_fcst"].notna()][["date", "tmax_f_fcst"]].copy()

    # ── Compute residuals: actual − GFS ────────────────────────────────────────
    # Positive residual → actual warmer than forecast (GFS underestimated)
    merged = lbl.merge(fcst, on="date", how="inner")
    merged["residual"] = merged["high"] - merged["tmax_f_fcst"]
    merged["month"] = merged["date"].apply(lambda d: d.month)

    # ── Fit per-month parameters ───────────────────────────────────────────────
    # gaussian: (mu, sigma) tuples; empirical: sorted residual arrays
    month_params: dict[int, object] = {}
    for month_val, grp in merged.groupby("month"):
        res = grp["residual"].dropna().to_numpy()
        if len(res) >= min_samples:
            if distribution == "gaussian":
                month_params[int(month_val)] = (float(res.mean()), float(res.std(ddof=1)))
            else:
                month_params[int(month_val)] = np.sort(res)

    # Global fallback
    all_res = merged["residual"].dropna().to_numpy()
    if distribution == "gaussian":
        global_params: object = (float(all_res.mean()), float(all_res.std(ddof=1)))
    else:
        global_params = np.sort(all_res)

    n_train  = len(merged)
    bias     = float(merged["residual"].mean()) if n_train else float("nan")
    spread   = float(merged["residual"].std())  if n_train else float("nan")
    n_months = len(month_params)
    print(
        f"[{rule_name}] Training: {n_train} days, {n_months}/12 months "
        f"with ≥{min_samples} samples | dist={distribution} | "
        f"GFS bias={bias:+.2f}°F, spread={spread:.2f}°F"
    )
    if distribution == "gaussian":
        print("  Monthly (mu, sigma) by month:")
        for m in sorted(month_params):
            mu, sigma = month_params[m]  # type: ignore[misc]
            print(f"    {m:02d}: mu={mu:+.2f}°F  sigma={sigma:.2f}°F")

    def rule(row: pd.Series) -> float:
        d         = row.get("settlement_date")
        threshold = row.get("threshold_f")
        direction = row.get("t_direction")
        gfs_tmax  = row.get("tmax_f_fcst")

        if d is None or threshold is None or pd.isna(threshold):
            return float("nan")
        if gfs_tmax is None or pd.isna(gfs_tmax):
            return float("nan")
        if direction not in ("greater", "less", "between"):
            return float("nan")

        if hasattr(d, "date"):
            d = d.date()

        params = month_params.get(d.month, global_params)

        if distribution == "gaussian":
            mu, sigma = params  # type: ignore[misc]
            if sigma < 1e-6:
                return float("nan")
            if direction == "between":
                floor_f = float(threshold)
                cap_f   = row.get("cap_f")
                if cap_f is None or pd.isna(cap_f):
                    cap_f = floor_f + 2.0  # standard Kalshi 2°F bracket
                calibrated = float(gfs_tmax) + mu
                p_below_cap = scipy_stats.norm.cdf(float(cap_f),   loc=calibrated, scale=sigma)
                p_below_flr = scipy_stats.norm.cdf(float(floor_f), loc=calibrated, scale=sigma)
                return float(p_below_cap - p_below_flr)
            # gap = T − GFS_tmax; P(actual ≥ T | GFS) = P(residual ≥ gap)
            gap = float(threshold) - float(gfs_tmax)
            if direction == "less":
                return float(scipy_stats.norm.cdf(gap, loc=mu, scale=sigma))
            return float(scipy_stats.norm.sf(gap, loc=mu, scale=sigma))
        else:
            residuals = params  # type: ignore[assignment]
            if len(residuals) == 0:
                return float("nan")
            if direction == "between":
                floor_f = float(threshold)
                cap_f   = float(row.get("cap_f") or floor_f + 2.0)
                # residual = actual − GFS; actual = GFS + residual
                return float(np.mean((residuals >= floor_f - float(gfs_tmax)) &
                                     (residuals <  cap_f   - float(gfs_tmax))))
            gap = float(threshold) - float(gfs_tmax)
            if direction == "less":
                return float(np.mean(residuals < gap))
            return float(np.mean(residuals >= gap))

    rule.__name__ = rule_name
    return rule


def make_isotonic_calibrated_rule(
    base_rule: Callable[[pd.Series], float],
    calibration_df: pd.DataFrame,
    rule_name: str | None = None,
) -> Callable[[pd.Series], float]:
    """
    Wrap a base rule with isotonic regression calibration.

    Isotonic regression (sklearn) fits a monotone mapping
    predicted_prob → actual_outcome_frequency using the calibration set.
    Because it is monotone, it preserves the ranking of the base rule's
    predictions while correcting systematic over- or under-confidence.

    Parameters
    ----------
    base_rule       : any rule function matching the backtest.Rule protocol.
    calibration_df  : aligned DataFrame for the calibration period. Must
                      include columns `actual_high_f`, `result`, and all
                      columns the base_rule reads. Outcomes are extracted
                      from `result` ("yes" → 1, "no" → 0).
    rule_name       : name for the returned rule (defaults to
                      f"isotonic({base_rule.__name__})").

    Returns
    -------
    A new rule that passes each row through base_rule, then maps the
    raw probability through the isotonic calibration curve.
    """
    from sklearn.isotonic import IsotonicRegression

    # Score every row in the calibration set
    cal_rows, cal_outcomes = [], []
    for _, row in calibration_df.iterrows():
        result = row.get("result")
        if result not in ("yes", "no"):
            continue
        p = base_rule(row)
        if not np.isfinite(p):
            continue
        cal_rows.append(float(p))
        cal_outcomes.append(1 if result == "yes" else 0)

    if len(cal_rows) < 20:
        raise ValueError(
            f"Too few calibration samples ({len(cal_rows)}); need ≥ 20."
        )

    X = np.array(cal_rows)
    y = np.array(cal_outcomes, dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)

    n_cal = len(X)
    raw_brier = float(np.mean((X - y) ** 2))
    cal_pred  = ir.predict(X)
    cal_brier = float(np.mean((cal_pred - y) ** 2))
    print(
        f"[isotonic] Calibration: n={n_cal}, "
        f"Brier before={raw_brier:.5f}, after={cal_brier:.5f}"
    )

    name = rule_name or f"isotonic({base_rule.__name__})"

    def calibrated_rule(row: pd.Series) -> float:
        p = base_rule(row)
        if not np.isfinite(p):
            return float("nan")
        result = ir.predict([float(p)])[0]
        return float(np.clip(result, 1e-6, 1.0 - 1e-6))

    calibrated_rule.__name__ = name
    return calibrated_rule


def make_b_isotonic_rule(
    labels: pd.DataFrame,
    forecasts: pd.DataFrame,
    cal_aligned_b: pd.DataFrame,
    station: str,
    model: str = "gfs_seamless",
    train_years: list[int] | None = None,
    min_samples: int = 20,
    min_yes: int = 100,
    rule_name: str = "b_isotonic",
) -> Callable[[pd.Series], float]:
    """
    Build an isotonic-calibrated rule for B-type (between) Kalshi contracts.

    B-type contracts settle YES if actual_high_f falls in a 2-integer-degree bracket
    [threshold_f - 0.5, threshold_f + 1.5), where threshold_f is the ticker .5 midpoint.

    Design notes
    ------------
    - Fits its own per-month (mu, sigma) from train_years — does NOT use load_bias_table()
      to avoid any look-ahead from calibration-year data being included in the bias table.
    - Isotonic calibration is trained against actual outcomes (result="yes"/"no"), not
      market prices.  Requires min_yes YES outcomes for a reliable calibration curve.
    - The T-type isotonic rule is NOT reused: T-type P(YES) is a tail probability (domain
      [0,1]) while B-type P(YES) is a narrow-band probability (typically [0.01, 0.35]).
      Sharing an isotonic map across these structurally different domains would corrupt both.
    - Uses increasing=True isotonic regression: higher gaussian B-prob → higher calibrated
      prob.  This holds because higher gaussian prob means the forecast is closer to the
      band center → more likely to land in the band.

    Parameters
    ----------
    labels        : from fetch_labels(); must include date, high, station.
    forecasts     : from fetch_forecasts(); must include date, tmax_f_fcst, model.
    cal_aligned_b : B-type rows from build_aligned_dataset() for the calibration year.
                    Must include: settlement_date, threshold_f, tmax_f_fcst, result.
    station       : NWS station ID (e.g. "KNYC").
    model         : forecast model to use.
    train_years   : years for fitting gaussian parameters (default [2022, 2023]).
    min_samples   : min observations per month for a per-month fit; falls back to global.
    min_yes       : minimum number of YES outcomes required; raises ValueError if not met.
    rule_name     : name for the returned callable.
    """
    from scipy import stats as scipy_stats
    from sklearn.isotonic import IsotonicRegression

    year_set = set(train_years or [2022, 2023])

    # ── 1. Fit per-month (mu, sigma) from training data ────────────────────────
    lbl = labels.copy()
    if "station" in lbl.columns:
        lbl = lbl[lbl["station"] == station]
    lbl["date"] = lbl["date"].apply(
        lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
    )
    lbl = lbl[lbl["date"].apply(lambda d: d.year in year_set)]
    lbl = lbl[lbl["high"].notna()][["date", "high"]].copy()

    fcst = forecasts[forecasts["model"] == model].copy()
    fcst["date"] = fcst["date"].apply(
        lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
    )
    fcst = fcst[fcst["date"].apply(lambda d: d.year in year_set)]
    fcst = fcst[fcst["tmax_f_fcst"].notna()][["date", "tmax_f_fcst"]].copy()

    merged = lbl.merge(fcst, on="date", how="inner")
    merged["residual"] = merged["high"] - merged["tmax_f_fcst"]
    merged["month"] = merged["date"].apply(lambda d: d.month)

    month_params: dict[int, tuple[float, float]] = {}
    for month_val, grp in merged.groupby("month"):
        res = grp["residual"].dropna().to_numpy()
        if len(res) >= min_samples:
            month_params[int(month_val)] = (float(res.mean()), float(res.std(ddof=1)))

    all_res = merged["residual"].dropna().to_numpy()
    global_params = (float(all_res.mean()), float(all_res.std(ddof=1)))

    n_train = len(merged)
    print(
        f"[{rule_name}] Gaussian training: {n_train} days, "
        f"{len(month_params)}/12 months with ≥{min_samples} samples"
    )

    # ── 2. Inner: compute gaussian B-type P(YES) for one row ──────────────────
    def _gauss_b(row: pd.Series) -> float:
        d         = row.get("settlement_date")
        threshold = row.get("threshold_f")
        gfs_tmax  = row.get("tmax_f_fcst")
        if d is None or threshold is None or pd.isna(threshold):
            return float("nan")
        if gfs_tmax is None or pd.isna(gfs_tmax):
            return float("nan")
        if hasattr(d, "date"):
            d = d.date()
        mu, sigma = month_params.get(d.month, global_params)
        if sigma < 1e-6:
            return float("nan")
        # Bracket is 2 integers wide: [threshold_f - 0.5, threshold_f + 1.5)
        floor_f    = float(threshold) - 0.5
        cap_f      = float(threshold) + 1.5
        calibrated = float(gfs_tmax) + mu
        return float(
            scipy_stats.norm.cdf(cap_f,   loc=calibrated, scale=sigma) -
            scipy_stats.norm.cdf(floor_f, loc=calibrated, scale=sigma)
        )

    # ── 3. Build calibration arrays from cal_aligned_b ────────────────────────
    cal_probs: list[float] = []
    cal_outcomes: list[int] = []

    for _, row in cal_aligned_b.iterrows():
        result = row.get("result")
        if result not in ("yes", "no"):
            continue
        p = _gauss_b(row)
        if not np.isfinite(p):
            continue
        cal_probs.append(p)
        cal_outcomes.append(1 if result == "yes" else 0)

    n_yes = sum(cal_outcomes)
    n_cal = len(cal_probs)

    if n_yes < min_yes:
        raise ValueError(
            f"[{rule_name}] Only {n_yes} YES outcomes in calibration data "
            f"(need ≥ {min_yes}). Skipping B-type isotonic."
        )

    X = np.array(cal_probs)
    y = np.array(cal_outcomes, dtype=float)

    # ── 4. Diagnostics before fitting ─────────────────────────────────────────
    raw_brier = float(np.mean((X - y) ** 2))

    # Monotonicity check: bin probs, verify actual YES rate is roughly increasing
    n_bins = 6
    bin_edges = np.percentile(X, np.linspace(0, 100, n_bins + 1))
    mono_ok = True
    prev_rate = -1.0
    print(f"[{rule_name}] Calibration data: n={n_cal}, YES={n_yes} ({n_yes/n_cal:.1%})")
    print(f"[{rule_name}] Gaussian prob range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"[{rule_name}] Monotonicity check (should be increasing):")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (X >= lo) & (X <= hi)
        n_b  = mask.sum()
        rate = float(y[mask].mean()) if n_b > 0 else float("nan")
        mid  = (lo + hi) / 2
        flag = "✓" if np.isnan(rate) or rate >= prev_rate - 0.02 else "✗ NOT MONOTONE"
        if "✗" in flag:
            mono_ok = False
        print(f"    gauss [{lo:.3f},{hi:.3f}] → actual {rate:.3f}  (n={n_b}) {flag}")
        if not np.isnan(rate):
            prev_rate = rate

    if not mono_ok:
        print(f"  WARNING: non-monotone region detected. Isotonic regression will "
              f"pool these bins — calibration may be coarser in this region.")

    # ── 5. Fit isotonic regression ─────────────────────────────────────────────
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(X, y)

    cal_pred  = ir.predict(X)
    cal_brier = float(np.mean((cal_pred - y) ** 2))
    improvement = (1 - cal_brier / raw_brier) * 100
    print(
        f"[{rule_name}] Brier: raw={raw_brier:.5f} → isotonic={cal_brier:.5f} "
        f"({improvement:.1f}% improvement)"
    )
    if improvement < 0:
        raise ValueError(
            f"[{rule_name}] Isotonic calibration made Brier WORSE "
            f"({raw_brier:.5f} → {cal_brier:.5f}). This should not happen. "
            f"Check calibration data for contamination."
        )

    # ── 6. Return closure ─────────────────────────────────────────────────────
    def rule(row: pd.Series) -> float:
        p = _gauss_b(row)
        if not np.isfinite(p):
            return float("nan")
        return float(np.clip(ir.predict([p])[0], 1e-6, 1.0 - 1e-6))

    rule.__name__ = rule_name
    return rule


def make_b_gaussian_rule(
    labels: pd.DataFrame,
    forecasts: pd.DataFrame,
    station: str,
    model: str = "gfs_seamless",
    train_years: list[int] | None = None,
    min_samples: int = 20,
    rule_name: str = "b_gaussian",
) -> Callable[[pd.Series], float]:
    """
    Gaussian-only B-type bracket rule (no isotonic calibration).

    P(YES) = P(actual ∈ [threshold_f-0.5, threshold_f+1.5) | GFS_tmax)
    where threshold_f is the ticker .5 midpoint (e.g. 92.5 for bracket 92–93°F).

    Used as a baseline for comparison with make_b_isotonic_rule.
    """
    from scipy import stats as scipy_stats

    year_set = set(train_years or [2022, 2023])

    lbl = labels.copy()
    if "station" in lbl.columns:
        lbl = lbl[lbl["station"] == station]
    lbl["date"] = lbl["date"].apply(
        lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
    )
    lbl = lbl[lbl["date"].apply(lambda d: d.year in year_set)]
    lbl = lbl[lbl["high"].notna()][["date", "high"]].copy()

    fcst = forecasts[forecasts["model"] == model].copy()
    fcst["date"] = fcst["date"].apply(
        lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
    )
    fcst = fcst[fcst["date"].apply(lambda d: d.year in year_set)]
    fcst = fcst[fcst["tmax_f_fcst"].notna()][["date", "tmax_f_fcst"]].copy()

    merged = lbl.merge(fcst, on="date", how="inner")
    merged["residual"] = merged["high"] - merged["tmax_f_fcst"]
    merged["month"] = merged["date"].apply(lambda d: d.month)

    month_params: dict[int, tuple[float, float]] = {}
    for month_val, grp in merged.groupby("month"):
        res = grp["residual"].dropna().to_numpy()
        if len(res) >= min_samples:
            month_params[int(month_val)] = (float(res.mean()), float(res.std(ddof=1)))

    all_res = merged["residual"].dropna().to_numpy()
    global_params = (float(all_res.mean()), float(all_res.std(ddof=1)))

    print(
        f"[{rule_name}] Gaussian training: {len(merged)} days, "
        f"{len(month_params)}/12 months with ≥{min_samples} samples"
    )

    def rule(row: pd.Series) -> float:
        d         = row.get("settlement_date")
        threshold = row.get("threshold_f")
        gfs_tmax  = row.get("tmax_f_fcst")
        if d is None or threshold is None or pd.isna(threshold):
            return float("nan")
        if gfs_tmax is None or pd.isna(gfs_tmax):
            return float("nan")
        if hasattr(d, "date"):
            d = d.date()
        mu, sigma = month_params.get(d.month, global_params)
        if sigma < 1e-6:
            return float("nan")
        floor_f    = float(threshold) - 0.5
        cap_f      = float(threshold) + 1.5
        calibrated = float(gfs_tmax) + mu
        return float(
            scipy_stats.norm.cdf(cap_f,   loc=calibrated, scale=sigma) -
            scipy_stats.norm.cdf(floor_f, loc=calibrated, scale=sigma)
        )

    rule.__name__ = rule_name
    return rule
