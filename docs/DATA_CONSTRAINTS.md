# DATA_CONSTRAINTS.md
# Human-verified facts about each data source. Updated 2026-06-03.
# Read this before writing any ingestion code.

> **Note:** documents the initial data audit, framed around the original NYC-only
> vertical slice. The API/settlement facts remain accurate and useful; current
> system scope and architecture are in
> [`STATE_OF_THE_MODEL.md`](STATE_OF_THE_MODEL.md).

## Vertical Slice Config
- **City:** New York City
- **Settlement station:** KNYC (NWS Central Park — NOT JFK or LGA)
- **Kalshi ticker prefix:** `WEATHER-HIGHTEMP-NYC-*`
- **Day boundary:** Local Standard Time (UTC-5 year-round for settlement purposes)
  - During DST, this means the measurement window is 1:00 AM–12:59 AM local clock time

---

## Source 1: Kalshi (Market Prices)

**What it provides:** Candlestick OHLC + trade prints for each settled contract.
No historical orderbook depth — only fills and candles.

**History depth:** Weather markets started ~2022. Exact earliest date must be
verified empirically by querying the oldest known NYC temperature ticker.

**API architecture (as of Feb 2026):**
- Markets settled before a rolling cutoff → `/historical/markets/{ticker}/candlesticks`
- Markets still active → `/markets/{ticker}/candlesticks`
- Bridge: `GET /trade-api/v2/historical/cutoff` returns the partition timestamp

**BREAKING CHANGES (Feb 2026 restructuring):**
- Price fields: now `yes_bid_dollars`, `yes_ask_dollars` (previously cent-denominated)
- `tick_size` field removed
- Timestamps: use `ts_ms` (milliseconds); legacy `ts` and `time` fields deprecated April 2026
- Any code written against pre-2026 documentation or community examples WILL break

**Rate limits:** Varies by account tier. Default public tier is low (20 req/sec).
Use exponential backoff; cache all responses as raw JSON alongside parquet.

**Authentication:** API key in header: `Authorization: Bearer {KALSHI_API_KEY}`

**Survivorship note:** Must enumerate ALL settled weather markets, not just
currently listed ones. Start from the historical endpoint to avoid missing
delisted contracts.

---

## Source 2: Open-Meteo (Forecast Features)

**What it provides:** Archived weather model forecasts, keyed by model run init time.
Two relevant products:

### Historical Forecast API
- Reconstructs "what was forecast at time T for time T+k" as a continuous series
- **GFS seamless (gfs_seamless):** available from 2021-01-01 (covers full Kalshi history)
- **ECMWF IFS 0.25° (ecmwf_ifs025):** available from ~2024-02-15 (verified empirically)
- **Individual ensemble members:** NOT available via this API
- Use for: daily tmax/tmin forecast; GFS is the primary model for M1 (covers 2021+)

### Ensemble API (live only)
- Individual member histories retained ~3 days only
- Members available from Sept 2025 forward via the Single Runs API archive
- Use for: live logger going forward; NOT usable for historical backtest pre-Sept 2025

**Implication for Milestone 1 baseline:**
Cannot use "fraction of GEFS members above threshold" for dates before Sept 2025.
Instead use ensemble mean + spread to approximate probability via normal CDF:
  P(T_max > threshold) ≈ 1 - Φ((threshold - ensemble_mean) / ensemble_spread)
Or use a pure climatological baseline (see below).

**Rate limits (free tier):** 10,000 calls/day; 5,000/hr; 600/min.
Requests over 10 variables OR longer than 2 weeks count as MULTIPLE calls.
**Pull once, cache to parquet. Never re-fetch inside the backtest loop.**

**Licensing:** CC-BY for non-commercial use. Commercial use requires paid plan.

**NEVER USE:** ERA5 / Open-Meteo Historical Weather API as a feature.
Reanalysis is after-the-fact truth and leaks the future into the model.

---

## Source 3: IEM / NWS CLI (Settlement Labels)

**What it provides:** Final NWS Daily Climate Reports (CLI/CF6) per station.
This is the exact source Kalshi uses for temperature contract settlement.

**API:** `https://mesonet.agron.iastate.edu/json/cli.py?station=KNYC&year=2026`
No authentication required.

**History depth:** KNYC records available from well before 2000. No gaps expected
for the 2022-present backtest window, but verify programmatically.

**Climatological baseline source:** IEM also provides the historical distribution
of daily highs by station and calendar date — usable as the null hypothesis
baseline for Milestone 1 without any forecast data at all.

**Important:**
- Use the FINAL CLI value, not preliminary reports
- Settlement can be delayed if the reported high is inconsistent with METAR highs
- The "day" boundary is LST midnight-to-midnight; during DST the clock shifts but
  the measurement window does not

**NOAA CDO note:** The legacy NOAA CDO API v2 endpoint is deprecated.
Do not use it. IEM is the correct primary source for CLI data.

---

## Known Gaps and Decisions

| Gap | Decision |
|-----|----------|
| Individual GEFS/ECMWF member history pre-Sept 2025 | Use ensemble mean+spread → normal CDF approximation for M1 baseline |
| Kalshi data start date unknown | Query empirically; document exact earliest ticker in DATA_AUDIT.md |
| No historical orderbook depth | Model fills from candle OHLC + trade prints only; dual fill model required |
| IEM has no SLA | Validate against NWS raw text products at api.weather.gov/products/types/CLI if needed |
