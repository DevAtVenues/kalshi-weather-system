"""
Historical backtest on the settled Kalshi candle archive (the statistically-powered
answer to "is there edge" — uses data we already have, not 5 forward days).

Signal   = calibrated DETERMINISTIC forecast (HRRR/gfs_seamless) — ensemble members
           aren't archived historically, so this tests the deterministic+calibration
           strategy, an honest lower bound on what the live ensemble can do.
Price    = real historical Kalshi candle close, taken DAY-AHEAD (last candle strictly
           before the settlement date) — the only executable historical price.
Label    = CLI settlement high → yes_won().

HARD RULES honored:
  • No look-ahead: calibration (per-station bias+std) is fit ONLY on years < the
    settlement year (walk-forward). The price is strictly pre-settlement-day.
  • Fills MODELED both ways: optimistic (maker at close, no fee) and pessimistic
    (taker: adverse slippage + Kalshi taker fee 0.07·p·(1−p)). Edge that only
    survives the optimistic model is not edge.
  • Effective sample size: CI via block bootstrap resampling whole DAYS, not
    contracts (weather days are autocorrelated).

Usage: .venv/bin/python scripts/historical_backtest.py
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from kalshi_weather.outcome_tracker import parse_ticker, yes_won   # noqa: E402
from kalshi_weather.calibration.bias import contract_probability   # noqa: E402

SERIES_STATION = {
    "KXHIGHNY": "KNYC", "KXHIGHCHI": "KMDW", "KXHIGHMIA": "KMIA", "KXHIGHTPHX": "KPHX",
    "KXHIGHTBOS": "KBOS", "KXHIGHTSFO": "KSFO", "KXHIGHLAX": "KLAX", "KXHIGHDEN": "KDEN",
    "KXHIGHTATL": "KATL", "KXHIGHTHOU": "KHOU", "KXHIGHTDC": "KDCA", "KXHIGHTNOLA": "KMSY",
    "KXHIGHPHIL": "KPHL", "KXHIGHTSEA": "KSEA", "KXHIGHTLV": "KLAS", "KXHIGHTMIN": "KMSP",
    "KXHIGHTSATX": "KSAT", "KXHIGHTDAL": "KDFW", "KXHIGHAUS": "KAUS", "KXHIGHTOKC": "KOKC",
}
SIGMA_FLOOR, SIGMA_CAP = 1.2, 4.0
EDGE_THRESH = 0.07      # only "trade" when |P − price| clears fees+half-spread
SLIP = 0.02             # pessimistic adverse slippage (taker)


def _station_for(ticker: str) -> str | None:
    t = ticker if ticker.startswith("KX") else "KX" + ticker
    for series in sorted(SERIES_STATION, key=len, reverse=True):
        if t.startswith(series + "-"):
            return SERIES_STATION[series]
    return None


def _load_meta() -> dict[str, tuple]:
    """ticker -> (strike_type, floor_strike, cap_strike) from REAL Kalshi metadata.
    parse_ticker only GUESSES (assumes every T is 'less'), which inverts ~half of them."""
    mk = pd.read_parquet("data/raw/kalshi/markets_all_cities.parquet")
    out = {}
    for r in mk.itertuples():
        st = getattr(r, "strike_type", None)
        if st in ("less", "greater", "between"):
            out[r.ticker] = (st,
                             float(r.floor_strike) if pd.notna(r.floor_strike) else None,
                             float(r.cap_strike) if pd.notna(r.cap_strike) else None)
    return out


def _load_fc_lab():
    fc = pd.concat([pd.read_parquet(f) for f in
                    glob.glob("data/raw/forecasts/gfs_seamless/*.parquet")], ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    fc = fc.dropna(subset=["tmax_f_fcst"]).drop_duplicates(["station", "date"])
    fmap = {(r.station, r.date): float(r.tmax_f_fcst) for r in fc.itertuples()}
    lab = []
    for st in os.listdir("data/raw/labels"):
        for f in glob.glob(f"data/raw/labels/{st}/*.parquet"):
            d = pd.read_parquet(f); d["station"] = st; lab.append(d)
    lab = pd.concat(lab, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["date"]).dt.date
    lmap = {(r.station, r.date): float(r.high) for r in lab.itertuples() if pd.notna(r.high)}
    # walk-forward calibration: per-station (bias, std) fit on years < Y
    err = lab.merge(
        pd.DataFrame([(s, d, v) for (s, d), v in fmap.items()], columns=["station", "date", "fcst"]),
        on=["station", "date"], how="inner")
    err["err"] = err["high"] - err["fcst"]
    err["year"] = pd.to_datetime(err["date"]).dt.year
    calib: dict[tuple[str, int], tuple[float, float]] = {}
    for st in err.station.unique():
        sub = err[err.station == st]
        for Y in range(2023, 2027):
            past = sub[sub.year < Y]["err"].dropna()
            if len(past) >= 60:
                calib[(st, Y)] = (float(past.mean()), float(np.clip(past.std(), SIGMA_FLOOR, SIGMA_CAP)))
    return fmap, lmap, calib


def _decision_price(df: pd.DataFrame, sdate: date) -> float | None:
    """Last candle CLOSE strictly before the settlement calendar day (day-ahead)."""
    ts = pd.to_datetime(df["end_period_ts"], unit="s", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df["end_period_ts"], errors="coerce")
    pre = df[ts.dt.date < sdate]
    pre = pre[pre.get("volume", 0).fillna(0) > 0] if "volume" in pre else pre
    if pre.empty:
        return None
    row = pre.iloc[-1]
    p = float(row["price_close"])
    oi = float(row.get("open_interest", 0) or 0)
    vol = float(pre["volume"].sum()) if "volume" in pre else 0.0
    return (p / 100.0 if p > 1.5 else p), oi, vol   # candles are cents; normalize


def run() -> None:
    fmap, lmap, calib = _load_fc_lab()
    meta = _load_meta()
    files = glob.glob("data/raw/kalshi/candles/*.parquet")
    print(f"Scanning {len(files)} settled Kalshi markets ({len(meta)} have real strike metadata)…")

    trades = []   # (day, strike_type, dir, pnl_opt, pnl_pess, oi, vol)
    uncond = []   # (day, pnl_pess) — buy NO on EVERY bracket, no model
    n_join = 0
    for i, f in enumerate(files):
        tk = os.path.basename(f)[:-8]
        tk_kx = tk if tk.startswith("KX") else "KX" + tk
        p = parse_ticker(tk_kx)
        if not p or tk_kx not in meta:      # need REAL strike direction, not a guess
            continue
        strike_type, floor_k, cap_k = meta[tk_kx]
        st = _station_for(tk)
        sdate = p["settlement_date"]; Y = sdate.year
        if st is None or (st, sdate) not in fmap or (st, sdate) not in lmap or (st, Y) not in calib:
            continue
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "price_close" not in df or df.empty:
            continue
        dp = _decision_price(df, sdate)
        if dp is None:
            continue
        price, oi, vol = dp
        if not (0.02 <= price <= 0.98):
            continue
        n_join += 1
        bias, sd = calib[(st, Y)]
        mu = fmap[(st, sdate)] + bias
        prob = contract_probability(mu, sd, floor=floor_k, cap=cap_k, strike_type=strike_type)
        if prob is None:
            continue
        parsed = {"strike_type": strike_type, "floor_strike": floor_k, "cap_strike": cap_k}
        outcome = 1 if yes_won(parsed, lmap[(st, sdate)]) else 0
        # Attribution baseline: what does UNCONDITIONALLY buying NO on every bracket
        # earn (no model, no edge filter)? If ≈ the model-selected edge, the "edge" is
        # structural bracket-overpricing, not our model.
        if strike_type == "between":
            fee0 = 0.07 * price * (1 - price)
            uncond.append((sdate, (1 - outcome) - (1 - max(0.01, price - SLIP)) - fee0))
        edge = prob - price
        if abs(edge) < EDGE_THRESH:
            continue
        p = {"strike_type": strike_type}   # for the type label below
        fee = 0.07 * price * (1 - price)
        if edge > 0:   # BUY YES
            pnl_opt = outcome - price
            pnl_pess = outcome - min(0.99, price + SLIP) - fee
            d = "YES"
        else:          # BUY NO
            pnl_opt = (1 - outcome) - (1 - price)
            pnl_pess = (1 - outcome) - (1 - max(0.01, price - SLIP)) - fee
            d = "NO"
        trades.append((sdate, p["strike_type"], d, pnl_opt, pnl_pess, oi, vol))

    print(f"Joined {n_join} markets with forecast+price+label; {len(trades)} cleared the "
          f"{EDGE_THRESH:.2f} edge bar.\n")
    if not trades:
        return
    tr = pd.DataFrame(trades, columns=["day", "type", "dir", "opt", "pess", "oi", "vol"])

    def _summary(sub: pd.DataFrame, label: str) -> None:
        if sub.empty:
            print(f"  {label:<22} (none)"); return
        # block bootstrap by DAY on the pessimistic pnl
        days = sub.groupby("day")["pess"].mean()
        boot = [np.random.choice(days.values, len(days), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:<22} n={len(sub):<5} days={sub.day.nunique():<4} "
              f"win={(sub.pess>0).mean()*100:4.0f}%  "
              f"opt={sub.opt.mean()*100:+5.1f}¢  pess={sub.pess.mean()*100:+5.1f}¢  "
              f"CI[{lo*100:+.1f},{hi*100:+.1f}]¢")

    print("Realized edge per contract (¢), optimistic=maker@close / pessimistic=taker+slip+fee:")
    print("  block-bootstrap 95% CI is on the PESSIMISTIC pnl, resampled by DAY.\n")
    _summary(tr, "ALL")
    _summary(tr[tr.dir == "NO"], "BUY_NO only")
    _summary(tr[tr.dir == "YES"], "BUY_YES only")
    _summary(tr[tr.type == "between"], "B-type (between)")
    _summary(tr[tr.type == "less"], "T-type (less)")
    _summary(tr[(tr.type == "between") & (tr.dir == "NO")], "B-type BUY_NO")

    print("\nStress test — the +edge should SHRINK toward 0 if it's thin-market price-noise selection:")
    bno = tr[(tr.type == "between") & (tr.dir == "NO")]
    for oi_min in (0, 50, 200, 500):
        _summary(bno[bno.oi >= oi_min], f"B-type NO, OI≥{oi_min}")

    print("\nAttribution — is it the MODEL, or just 'short every bracket'?")
    ub = pd.DataFrame(uncond, columns=["day", "pess"])
    if not ub.empty:
        d = ub.groupby("day")["pess"].mean()
        boot = [np.random.choice(d.values, len(d), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {'UNCOND short brackets':<22} n={len(ub):<5} days={ub.day.nunique():<4} "
              f"win={(ub.pess>0).mean()*100:4.0f}%  pess={ub.pess.mean()*100:+5.1f}¢  "
              f"CI[{lo*100:+.1f},{hi*100:+.1f}]¢")
        print(f"  {'MODEL-selected NO':<22} (see B-type BUY_NO row above) — model adds value only if > uncond")

    print("\nYear-by-year robustness (B-type BUY_NO, pessimistic) — must hold EVERY year:")
    bno = bno.assign(year=bno.day.map(lambda d: d.year))
    for y in sorted(bno.year.unique()):
        s = bno[bno.year == y]
        print(f"  {y}: n={len(s):<4} days={s.day.nunique():<4} win={(s.pess>0).mean()*100:3.0f}%  pess={s.pess.mean()*100:+5.1f}¢")
    print("\n(Deterministic strategy, day-ahead candle price. Edge that shows only in the")
    print(" optimistic column is not edge. CI crossing 0 = not proven.)")


if __name__ == "__main__":
    np.random.seed(0)
    run()
