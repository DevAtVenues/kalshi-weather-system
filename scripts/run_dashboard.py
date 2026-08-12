"""
Terminal dashboard for Kalshi weather contracts.

Displays live METAR temps, today's GFS forecast high/low, intraday lock status,
and current Kalshi bid/ask for the at-the-money contract — one row per city.

Run:
  .venv/bin/python scripts/run_dashboard.py

Options (env vars):
  DASHBOARD_POLL_SECONDS   — METAR + Kalshi price refresh (default: 600 = 10 min)
  DASHBOARD_CITIES         — comma-separated city keys to show (default: all)
                             e.g.  DASHBOARD_CITIES=NYC,CHI,MIA
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import requests
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from kalshi_weather.config import load_stations
from kalshi_weather.ingest.kalshi import _get
from kalshi_weather.ingest.metar import fetch_metar
from kalshi_weather.monitor.lock import DailyTracker

POLL_INTERVAL = int(os.getenv("DASHBOARD_POLL_SECONDS", "600"))

_CITY_FILTER = os.getenv("DASHBOARD_CITIES", "")
CITY_FILTER  = set(_CITY_FILTER.upper().split(",")) if _CITY_FILTER else set()

console = Console()


# ── GFS forecast ──────────────────────────────────────────────────────────────

def fetch_gfs_forecasts(stations_cfg: dict) -> dict[str, dict]:
    """Today's GFS predicted high/low (°F) per METAR station. One call per city."""
    result = {}
    for cfg in stations_cfg.values():
        station = cfg["metar_station"]
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":         cfg["lat"],
                    "longitude":        cfg["lon"],
                    "daily":            "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "timezone":         cfg["timezone"],
                    "forecast_days":    1,
                    "models":           "gfs_seamless",
                },
                timeout=10,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            highs = daily.get("temperature_2m_max", [None])
            lows  = daily.get("temperature_2m_min", [None])
            result[station] = {
                "gfs_h": round(highs[0], 0) if highs and highs[0] is not None else None,
                "gfs_l": round(lows[0],  0) if lows  and lows[0]  is not None else None,
            }
        except Exception:
            result[station] = {"gfs_h": None, "gfs_l": None}
        time.sleep(0.1)
    return result


# ── Kalshi contracts ──────────────────────────────────────────────────────────

def fetch_contracts_for_series(series_ticker: str, city: str) -> list[dict]:
    try:
        data = _get("/markets", {"series_ticker": series_ticker, "status": "open", "limit": 200})
    except Exception:
        return []
    out = []
    for m in data.get("markets", []):
        if m.get("strike_type") not in ("greater", "less"):
            continue
        out.append({
            "ticker":       m["ticker"],
            "city":         city,
            "strike_type":  m["strike_type"],
            "floor_strike": m.get("floor_strike"),
            "cap_strike":   m.get("cap_strike"),
            "yes_bid":      m.get("yes_bid"),
            "yes_ask":      m.get("yes_ask"),
        })
    return out


def load_contracts(stations_cfg: dict) -> dict[str, list[dict]]:
    by_station: dict[str, list[dict]] = defaultdict(list)
    for cfg in stations_cfg.values():
        series = cfg.get("kalshi_series", "")
        if not series:
            continue
        by_station[cfg["metar_station"]].extend(
            fetch_contracts_for_series(series, cfg["city"])
        )
        time.sleep(0.15)
    return dict(by_station)


# ── Lock status (read-only — does not consume _alerted) ───────────────────────

def current_lock(tracker: DailyTracker, contracts: list[dict]) -> tuple[str, str]:
    """Return (label, rich_style) for the lock column without side effects."""
    if not tracker._readings:
        return "—", "dim"

    rhi = tracker.running_high_f

    # Walk contracts in strike order so we report the most relevant one
    for c in sorted(contracts, key=lambda x: float(x.get("floor_strike") or x.get("cap_strike") or 0)):
        if c["strike_type"] == "greater" and c.get("floor_strike") is not None:
            if rhi >= float(c["floor_strike"]):
                return f"YES ≥{float(c['floor_strike']):.0f}°F", "bold green"
        elif c["strike_type"] == "less" and c.get("cap_strike") is not None:
            if rhi >= float(c["cap_strike"]):
                return f"NO  >{float(c['cap_strike']):.0f}°F", "bold red"

    clo = tracker.confirmed_low_f
    if clo is not None:
        return f"LOW {clo:.0f}°F confirmed", "bold cyan"

    return "—", "dim"


def atm_contract(contracts: list[dict], temp_f: float | None) -> dict | None:
    """At-the-money 'greater' contract — floor_strike closest to current temp."""
    greaters = [
        c for c in contracts
        if c["strike_type"] == "greater" and c.get("floor_strike") is not None
    ]
    if not greaters or temp_f is None:
        return None
    return min(greaters, key=lambda c: abs(float(c["floor_strike"]) - temp_f))


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(
    stations_cfg: dict,
    obs: dict,
    forecasts: dict,
    contracts_by_station: dict,
    trackers: dict,
    utc_now: datetime,
    metar_age_s: float,
    price_age_s: float,
) -> Table:
    metar_age = f"{int(metar_age_s // 60)}m" if metar_age_s < 3600 else f"{metar_age_s/3600:.1f}h"
    price_age = f"{int(price_age_s // 60)}m" if price_age_s < 3600 else f"{price_age_s/3600:.1f}h"
    title = (
        f"[bold white]Kalshi Weather Monitor[/bold white]  "
        f"[dim]{utc_now:%H:%M UTC}[/dim]  "
        f"[dim]METAR {metar_age} ago  ·  Prices {price_age} ago[/dim]"
    )

    t = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        show_footer=False,
        expand=False,
        title_justify="left",
    )
    t.add_column("City",       style="bold white", min_width=14)
    t.add_column("Temp",       justify="right",    min_width=7)
    t.add_column("GFS H/L",    justify="center",   min_width=9)
    t.add_column("Day H/L",    justify="center",   min_width=9)
    t.add_column("Lock",       min_width=22)
    t.add_column("ATM Bid/Ask",justify="center",   min_width=14)

    for cfg in stations_cfg.values():
        station = cfg["metar_station"]
        city    = cfg["city"]

        reading  = obs.get(station)
        temp_f   = reading["temp_f"] if reading else None
        temp_str = f"{temp_f:.0f}°F" if temp_f is not None else "[dim]—[/dim]"

        fc      = forecasts.get(station, {})
        gfs_h   = fc.get("gfs_h")
        gfs_l   = fc.get("gfs_l")
        gfs_str = f"{gfs_h:.0f}/{gfs_l:.0f}" if gfs_h is not None else "[dim]—[/dim]"

        tracker = trackers.get(station)
        if tracker and tracker._readings:
            rhi = tracker.running_high_f
            rlo = tracker.running_low_f
            day_str = f"{rhi:.0f}/{rlo:.0f}"
        else:
            day_str = "[dim]—[/dim]"

        contracts = contracts_by_station.get(station, [])
        lock_txt, lock_style = current_lock(tracker, contracts) if tracker else ("—", "dim")

        atm = atm_contract(contracts, temp_f)
        if atm:
            bid = atm.get("yes_bid") or 0
            ask = atm.get("yes_ask") or 0
            fs  = float(atm["floor_strike"])
            price_str = f"T{fs:.0f}  {bid}¢/{ask}¢"
        else:
            price_str = "[dim]—[/dim]"

        temp_display = f"[{lock_style}]{temp_str}[/{lock_style}]" if lock_txt != "—" else temp_str

        t.add_row(
            city,
            temp_display,
            gfs_str,
            day_str,
            f"[{lock_style}]{lock_txt}[/{lock_style}]",
            price_str,
        )

    return t


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    stations_cfg = load_stations()
    if CITY_FILTER:
        stations_cfg = {k: v for k, v in stations_cfg.items() if k in CITY_FILTER}

    trackers: dict[str, DailyTracker] = {
        cfg["metar_station"]: DailyTracker(
            station=cfg["metar_station"],
            lst_offset=cfg["lst_offset_hours"],
        )
        for cfg in stations_cfg.values()
    }
    all_stations = [cfg["metar_station"] for cfg in stations_cfg.values()]
    track_dates: dict[str, object] = {}

    console.print("[bold]Loading GFS forecasts…[/bold]  (one call per city)")
    forecasts = fetch_gfs_forecasts(stations_cfg)
    last_forecast_refresh = time.monotonic()

    console.print("[bold]Loading Kalshi contracts…[/bold]")
    contracts_by_station = load_contracts(stations_cfg)
    last_price_refresh = time.monotonic()

    console.print("[bold]Fetching METAR…[/bold]")
    obs = fetch_metar(all_stations)
    last_metar_fetch = time.monotonic()
    for station, tracker in trackers.items():
        r = obs.get(station)
        if r:
            tracker.update(r["temp_f"])

    console.print(f"[dim]Ready. Refreshing every {POLL_INTERVAL // 60} min. Ctrl+C to quit.[/dim]\n")

    with Live(console=console, refresh_per_second=0.5, screen=False) as live:
        while True:
            utc_now  = datetime.now(timezone.utc)
            now_mono = time.monotonic()

            # Day rollover per station
            for station, tracker in trackers.items():
                new_date = tracker.settlement_date(utc_now)
                if track_dates.get(station) not in (None, new_date):
                    tracker.reset()
                track_dates[station] = new_date

            # GFS: refresh every 6 h (model updates 4× / day)
            if now_mono - last_forecast_refresh >= 6 * 3600:
                forecasts = fetch_gfs_forecasts(stations_cfg)
                last_forecast_refresh = now_mono

            # METAR + Kalshi prices: every POLL_INTERVAL
            if now_mono - last_metar_fetch >= POLL_INTERVAL:
                try:
                    obs = fetch_metar(all_stations)
                    last_metar_fetch = now_mono
                    for station, tracker in trackers.items():
                        r = obs.get(station)
                        if r:
                            tracker.update(r["temp_f"])
                except Exception as e:
                    console.log(f"[red]METAR error:[/red] {e}")

            if now_mono - last_price_refresh >= POLL_INTERVAL:
                try:
                    contracts_by_station = load_contracts(stations_cfg)
                    last_price_refresh = now_mono
                except Exception as e:
                    console.log(f"[red]Kalshi error:[/red] {e}")

            metar_age = now_mono - last_metar_fetch
            price_age = now_mono - last_price_refresh

            live.update(build_table(
                stations_cfg, obs, forecasts, contracts_by_station,
                trackers, utc_now, metar_age, price_age,
            ))

            time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")
