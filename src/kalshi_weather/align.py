"""
Join the three data sources into a single flat DataFrame for backtesting.

Each output row = one settled Kalshi contract, with:
  - settlement metadata (ticker, date, threshold, contract_type)
  - IEM settlement label (actual high temperature, Kalshi result)
  - Open-Meteo forecast feature (tmax for that day, issued before the day)
  - Kalshi market price at close (coerced to float)

No look-ahead: the Historical Forecast API returns the forecast that existed
*before* each settlement date, so joining on settlement_date is safe.
The leakage test suite in tests/test_leakage.py enforces the general invariant.

Contract types in the KXHIGHNY (formerly HIGHNY) series:
  "T"  — threshold contract: pays YES if tmax >= threshold_f
  "B"  — bracket contract:   pays YES if tmax falls in a 2°F bracket
           [threshold_f, threshold_f + 2°F)
  "?"  — unknown / could not parse ticker

For Milestone 1 baseline, use threshold_contracts() to restrict to T contracts.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from kalshi_weather.tz import lst_date_for_utc

# Pattern: <SERIES>-<YYMONDD>-<TYPE><VALUE>
# TYPE is T (threshold) or B (bracket).  VALUE is numeric (int or N.N).
_TICKER_RE = re.compile(r"^[A-Z]+-\d{2}[A-Z]{3}\d{2}-([TB])([\d.]+)$")


def _to_utc(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _to_date(val: object) -> date:
    return pd.Timestamp(val).date()


def _parse_contract_type(ticker: str) -> tuple[str, float | None]:
    """
    Return (contract_type, threshold_f) parsed from a ticker string.

    contract_type: "T" (threshold), "B" (bracket), or "?" (unknown)
    threshold_f  : the numeric value after the type letter, or None if not parseable

    For T contracts: threshold_f is the temperature that must be reached for YES.
    For B contracts: threshold_f is the lower bound of the 2°F bracket.
    """
    m = _TICKER_RE.match(ticker)
    if not m:
        return "?", None
    ctype = m.group(1)
    try:
        val = float(m.group(2))
    except ValueError:
        val = None
    return ctype, val


def build_aligned_dataset(
    markets: pd.DataFrame,
    labels: pd.DataFrame,
    forecasts: pd.DataFrame,
    station: str = "KNYC",
    model: str = "gfs_seamless",
) -> pd.DataFrame:
    """
    Produce one row per settled Kalshi contract.

    Parameters
    ----------
    markets   : from ingest.kalshi.fetch_all_markets()
    labels    : from ingest.labels.fetch_labels()
    forecasts : from ingest.forecasts.fetch_forecasts()
    station   : NWS station ID
    model     : Open-Meteo model to use as forecast feature

    Returns
    -------
    Columns:
        ticker, settlement_date, contract_type, threshold_f,
        actual_high_f, actual_low_f, result, label_consistent,
        tmax_f_fcst, tmin_f_fcst, high_normal,
        last_price_dollars, yes_bid_dollars, yes_ask_dollars,
        volume_fp, expiration_value
    """
    # ── 1. Markets: settlement_date, contract_type, threshold ─────────────────
    mkt = markets.copy()

    mkt["settlement_date"] = mkt["close_time"].apply(
        lambda t: lst_date_for_utc(_to_utc(t), station)
    )

    parsed = mkt["ticker"].apply(_parse_contract_type)
    mkt["contract_type"] = parsed.apply(lambda x: x[0])
    mkt["threshold_f"]   = parsed.apply(lambda x: x[1])

    # For T contracts, prefer floor_strike (whole-number precision is fine for tails).
    # For B contracts, keep the ticker-parsed value — it carries the .5 suffix that
    # defines the exact bracket lower bound [T, T+2); floor_strike truncates to int.
    if "floor_strike" in mkt.columns:
        fs = pd.to_numeric(mkt["floor_strike"], errors="coerce")
        is_t = mkt["contract_type"] == "T"
        mkt["threshold_f"] = mkt["threshold_f"].where(~(is_t & fs.notna()), fs)

    # T contract direction from Kalshi strike_type field:
    #   "greater" → YES if tmax >= floor_strike (warm/upper tail)
    #   "less"    → YES if tmax <  cap_strike   (cold/lower tail)
    # Expose as t_direction so rules can compute the correct P(YES).
    if "strike_type" in mkt.columns:
        mkt["t_direction"] = mkt["strike_type"]
    else:
        mkt["t_direction"] = pd.NA

    # Coerce price / volume fields to float (API returns strings)
    for col in ("last_price_dollars", "yes_bid_dollars", "yes_ask_dollars",
                "volume_fp", "expiration_value"):
        if col in mkt.columns:
            mkt[col] = pd.to_numeric(mkt[col], errors="coerce")

    market_cols = [
        "ticker", "settlement_date", "contract_type", "threshold_f", "t_direction",
        "result", "expiration_value", "last_price_dollars",
        "yes_bid_dollars", "yes_ask_dollars", "volume_fp",
    ]
    mkt_slim = mkt[[c for c in market_cols if c in mkt.columns]].copy()

    # ── 2. Labels ─────────────────────────────────────────────────────────────
    lbl = labels.copy()
    if "station" in lbl.columns:
        lbl = lbl[lbl["station"] == station]
    lbl["date"] = lbl["date"].apply(_to_date)
    lbl = lbl.rename(columns={"high": "actual_high_f", "low": "actual_low_f"})
    keep_lbl = ["date", "actual_high_f", "actual_low_f", "high_normal"]
    lbl_slim = lbl[[c for c in keep_lbl if c in lbl.columns]].copy()

    # ── 3. Forecasts ──────────────────────────────────────────────────────────
    fcst = forecasts[forecasts["model"] == model].copy()
    fcst["date"] = fcst["date"].apply(_to_date)
    keep_fcst = ["date", "tmax_f_fcst", "tmin_f_fcst"]
    fcst_slim = fcst[[c for c in keep_fcst if c in fcst.columns]].copy()

    # ── 4. Join ───────────────────────────────────────────────────────────────
    aligned = mkt_slim.merge(lbl_slim, left_on="settlement_date", right_on="date", how="left")
    if "date" in aligned.columns:
        aligned = aligned.drop(columns=["date"])

    aligned = aligned.merge(fcst_slim, left_on="settlement_date", right_on="date", how="left")
    if "date" in aligned.columns:
        aligned = aligned.drop(columns=["date"])

    # ── 5. Label consistency ──────────────────────────────────────────────────
    # B contracts: YES iff actual_high_f ∈ [threshold_f, threshold_f + 2).
    #   Brackets are 2°F wide; threshold_f is the lower bound (e.g., 54.5).
    # T contracts: tail contracts with ambiguous direction (lower vs upper tail
    #   determined by position relative to the day's bracket range). Skip check.
    # label_consistent=False when data is missing or contract type is unknown.
    both_present = aligned["actual_high_f"].notna() & aligned["threshold_f"].notna()
    result_yes   = aligned["result"] == "yes"
    is_b = aligned["contract_type"] == "B"

    # B contract bracket: the ticker value x.5 is the bracket MIDPOINT.
    # Actual boundaries are integer: [x-0.5, x+1.5) i.e. high ∈ {x, x+1}.
    # NWS CLI temperatures are whole degrees, so this covers exactly two integers.
    in_bracket = (
        (aligned["actual_high_f"] >= aligned["threshold_f"] - 0.5) &
        (aligned["actual_high_f"] <  aligned["threshold_f"] + 1.5)
    )
    aligned["label_consistent"] = both_present & is_b & (result_yes == in_bracket)

    # ── 6. Column order ───────────────────────────────────────────────────────
    col_order = [
        "ticker", "settlement_date", "contract_type", "threshold_f", "t_direction",
        "actual_high_f", "actual_low_f", "result", "label_consistent",
        "tmax_f_fcst", "tmin_f_fcst", "high_normal",
        "last_price_dollars", "yes_bid_dollars", "yes_ask_dollars",
        "volume_fp", "expiration_value",
    ]
    col_order = [c for c in col_order if c in aligned.columns]
    return aligned[col_order].sort_values("settlement_date").reset_index(drop=True)


def threshold_contracts(aligned: pd.DataFrame) -> pd.DataFrame:
    """Return only T-type (threshold) contracts with a valid threshold, direction, and price."""
    direction_ok = (
        aligned["t_direction"].isin(["greater", "less"])
        if "t_direction" in aligned.columns
        else True
    )
    mask = (
        (aligned["contract_type"] == "T")
        & aligned["threshold_f"].notna()
        & direction_ok
        & aligned["last_price_dollars"].notna()
    )
    return aligned[mask].reset_index(drop=True)


def alignment_summary(aligned: pd.DataFrame) -> None:
    """Print a diagnostic summary of an aligned dataset."""
    n = len(aligned)

    by_type = aligned["contract_type"].value_counts().to_dict() if "contract_type" in aligned.columns else {}
    n_label  = aligned["actual_high_f"].notna().sum()
    n_fcst   = aligned["tmax_f_fcst"].notna().sum() if "tmax_f_fcst" in aligned.columns else 0
    n_price  = aligned["last_price_dollars"].notna().sum() if "last_price_dollars" in aligned.columns else 0

    b_rows = aligned[aligned.get("contract_type", pd.Series()) == "B"] if "contract_type" in aligned.columns else pd.DataFrame()
    b_with_label = (b_rows["actual_high_f"].notna() & b_rows["threshold_f"].notna()).sum() if not b_rows.empty else 0
    b_consistent = b_rows["label_consistent"].sum() if not b_rows.empty else 0

    print(f"Aligned dataset: {n} contracts total")
    print(f"  By type:           {by_type}")
    print(f"  Label coverage:    {n_label}/{n} ({100*n_label/n:.1f}%)")
    print(f"  Forecast coverage: {n_fcst}/{n} ({100*n_fcst/n:.1f}%)")
    print(f"  Price coverage:    {n_price}/{n} ({100*n_price/n:.1f}%)")
    if not b_rows.empty and b_with_label > 0:
        print(f"  B-contract label consistent: {b_consistent}/{b_with_label} "
              f"({100*b_consistent/b_with_label:.1f}%)  [T-contracts skipped: tail logic complex]")
    if "settlement_date" in aligned.columns:
        dates = aligned["settlement_date"].dropna()
        print(f"  Date range:        {dates.min()} → {dates.max()}")
