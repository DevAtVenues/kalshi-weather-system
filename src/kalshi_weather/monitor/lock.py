"""
Intraday lock detection for Kalshi temperature contracts.

A "lock" is when METAR observations make the contract outcome certain
before the NWS CLI report is published the next day:

  LOCK YES (greater contracts): running daily high >= floor_strike
      The temperature has already been physically observed at or above
      the threshold — the contract WILL settle YES.

  LOCK NO  (greater contracts): running daily high is well below
      floor_strike and it is past the typical daily peak hour.
      (soft signal — conservative threshold required)

  LOCK NO  (less contracts):    running daily high >= cap_strike
      The temperature has exceeded the cap — the "YES if tmax < cap"
      contract WILL settle NO. Complement: LOCK YES for the paired
      "greater" contract at the same threshold.

The 2°F confirmed-low rule (NowCast): once the observed temperature
has risen 2°F above today's running minimum, the daily low is
confirmed at that minimum value.  Used for low-temperature contracts
when they exist.

Each station gets a DailyTracker that resets at midnight LST.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from kalshi_weather.tz import UTC
from typing import NamedTuple

# METAR reports temperature in whole °C; max rounding error = 0.5°C = 0.9°F.
# A hard-lock comparison must exceed the contract boundary by this margin —
# otherwise the NWS actual could still land on the other side.
METAR_MARGIN_F = 0.9


# ---------------------------------------------------------------------------
# Lock event
# ---------------------------------------------------------------------------

class LockEvent(NamedTuple):
    ticker:      str
    city:        str
    lock_type:   str    # "YES_locked" | "NO_locked" | "LOW_confirmed"
    condition:   str    # human-readable description
    running_high_f: float
    running_low_f:  float
    threshold_f:    float | None


# ---------------------------------------------------------------------------
# Per-station daily state
# ---------------------------------------------------------------------------

@dataclass
class DailyTracker:
    """Accumulates intraday METAR observations for one station-day."""
    station:       str
    lst_offset:    int          # hours from UTC (e.g. -5 for EST)
    _readings:     list[float]  = field(default_factory=list)
    _alerted:      set[str]     = field(default_factory=set)  # tickers already fired

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def running_high_f(self) -> float:
        return max(self._readings) if self._readings else float("-inf")

    @property
    def running_low_f(self) -> float:
        return min(self._readings) if self._readings else float("inf")

    @property
    def confirmed_low_f(self) -> float | None:
        """
        Daily minimum is 'confirmed' once the temperature has risen
        2°F above the running minimum (NowCast 2°F reversal rule).
        """
        if len(self._readings) < 2:
            return None
        rmin = self.running_low_f
        if self._readings[-1] >= rmin + 2.0:
            return rmin
        return None

    # ── settlement day ──────────────────────────────────────────────────────

    def settlement_date(self, utc_now: datetime) -> date:
        """The LST date this tracker is accumulating for."""
        lst = utc_now + timedelta(hours=self.lst_offset)
        return lst.date()

    # ── update ──────────────────────────────────────────────────────────────

    def update(self, temp_f: float) -> None:
        self._readings.append(temp_f)

    def reset(self) -> None:
        self._readings.clear()
        self._alerted.clear()

    # ── lock check ──────────────────────────────────────────────────────────

    def check_locks(
        self,
        contracts: list[dict],
        utc_now: datetime | None = None,
    ) -> list[LockEvent]:
        """
        Evaluate all contracts for lock conditions.
        Returns only NEW locks (not previously alerted).
        """
        if not self._readings:
            return []

        utc_now = utc_now or datetime.now(UTC)
        events: list[LockEvent] = []

        rhi = self.running_high_f
        rlo = self.running_low_f
        confirmed_lo = self.confirmed_low_f

        for c in contracts:
            ticker    = c.get("ticker", "")
            direction = c.get("strike_type", "")
            threshold: float | None = None
            lock_type: str | None   = None
            condition: str | None   = None

            # Canonical rules (kalshi_weather.settlement): greater YES iff
            # h >= floor+1 (true max >= floor+0.5); less NO iff h >= cap
            # (true max >= cap-0.5). Lock = true boundary + METAR margin.
            if direction == "greater":
                fs = c.get("floor_strike")
                if fs is None:
                    continue
                threshold = float(fs)
                if rhi >= threshold + 0.5 + METAR_MARGIN_F:
                    lock_type = "YES_locked"
                    condition = (
                        f"running high {rhi:.1f}°F ≥ floor {threshold:.0f}°F"
                        f"+0.5 (+{METAR_MARGIN_F}°F margin)"
                    )

            elif direction == "less":
                cs = c.get("cap_strike")
                if cs is None:
                    continue
                threshold = float(cs)
                if rhi >= threshold - 0.5 + METAR_MARGIN_F:
                    # The day's high has exceeded the cap → YES is now impossible
                    lock_type = "NO_locked"
                    condition = (
                        f"running high {rhi:.1f}°F ≥ cap_strike {threshold:.0f}°F "
                        f"(+{METAR_MARGIN_F}°F margin)"
                        f" → YES (tmax < {threshold:.0f}°F) is now impossible"
                    )

            if lock_type and ticker not in self._alerted:
                self._alerted.add(ticker)
                events.append(LockEvent(
                    ticker=ticker,
                    city=c.get("city", self.station),
                    lock_type=lock_type,
                    condition=condition or "",
                    running_high_f=rhi,
                    running_low_f=rlo,
                    threshold_f=threshold,
                ))

        # Confirmed-low event (fires once per day per station)
        low_key = f"__low_{self.station}"
        if confirmed_lo is not None and low_key not in self._alerted:
            self._alerted.add(low_key)
            events.append(LockEvent(
                ticker=low_key,
                city=self.station,
                lock_type="LOW_confirmed",
                condition=(
                    f"confirmed daily low = {confirmed_lo:.1f}°F "
                    f"(current {self._readings[-1]:.1f}°F, reversed ≥2°F)"
                ),
                running_high_f=rhi,
                running_low_f=confirmed_lo,
                threshold_f=confirmed_lo,
            ))

        return events
