"""NBM NBP bulletin ingest: station-block parsing (pinned against the live
2026-07-21 01Z file format, NBM V5.0), UTC→local settlement-day mapping,
block-boundary isolation, cache-only reads, and the quantile→P(YES) CDF."""
import pytest
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.ingest import nbm

# Verbatim shape of the live file (KSAT block, trimmed to 3 day-groups).
BLOCK = """\
 KSAT    NBM V5.0 NBP GUIDANCE    7/21/2026  0100 UTC
        WED 22| THU 23| FRI 24
 UTC    00  12| 00  12| 00  12
 FHR    23  35| 47  59| 71  83
 TXNMN  97  79| 99  83| 98  82
 TXNSD   2   5|  2   5|  2   4
 TXNP1  95  73| 97  77| 94  77
 TXNP2  96  75| 97  81| 96  78
 TXNP5  97  79| 99  84| 98  82
 TXNP7  98  82|100  86| 99  86
 TXNP9  99  85|102  89|100  88
"""

# A station whose header must still CLOSE the tracked block even though we
# don't want it (ids that aren't K***: buoys, P/T sites).
OTHER = """\
 PADK    NBM V5.0 NBP GUIDANCE    7/21/2026  0100 UTC
        WED 22| THU 23| FRI 24
 UTC    00  12| 00  12| 00  12
 FHR    23  35| 47  59| 71  83
 TXNP1  10  10| 10  10| 10  10
 TXNP2  10  10| 10  10| 10  10
 TXNP5  10  10| 10  10| 10  10
 TXNP7  10  10| 10  10| 10  10
 TXNP9  10  10| 10  10| 10  10
"""

Q = {"p10": 95, "p25": 96, "p50": 97, "p75": 98, "p90": 99}


def test_parse_block_maps_00z_to_prior_local_day():
    parsed = nbm._parse_block(BLOCK.splitlines(), "KSAT")
    # 00Z Jul 22 UTC = evening of Jul 21 in San Antonio (CST) → local day 07-21
    assert parsed["2026-07-21"] == {"p10": 95, "p25": 96, "p50": 97,
                                    "p75": 98, "p90": 99, "mean": 97, "sd": 2}
    assert parsed["2026-07-22"]["p50"] == 99      # FHR 47 → 00Z Jul 23 → local 22nd
    assert set(parsed) == {"2026-07-21", "2026-07-22", "2026-07-23"}


def test_extract_isolates_blocks_and_ignores_unwanted_station():
    text = (OTHER + BLOCK + OTHER).splitlines()
    found = nbm._extract_stations(iter(text), {"KSAT"})
    assert set(found) == {"KSAT"}
    # PADK's all-10 rows must not have bled into KSAT's parse
    assert found["KSAT"]["2026-07-21"]["p50"] == 97


def test_latest_quantiles_reads_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(nbm, "CACHE_DIR", tmp_path)
    doc = {"fetched_at": "x", "complete": True,
           "stations": {"KSAT": {"2026-07-21": dict(Q)}}}
    (tmp_path / "nbp_20260721_01z.json").write_text(json.dumps(doc))
    now = datetime(2026, 7, 21, 5, 20, tzinfo=timezone.utc)   # board time, 01Z newest full cycle
    q = nbm.latest_quantiles("KSAT", "2026-07-21", stations={"KSAT"}, now=now)
    assert q["p50"] == 97 and q["cycle"] == "20260721_01Z"
    # unknown day → None (walks lookback, finds nothing, no crash)
    assert nbm.latest_quantiles("KSAT", "2026-08-01",
                                stations={"KSAT"}, now=now) is None


def test_nbm_prob_interior_matches_linear_cdf():
    # Canonical rules: greater strict > → P(int high >= 99) = 1 - cdf(98.5)
    assert nbm.nbm_prob("greater", 98.0, None, Q) == pytest.approx(0.175, abs=2e-3)
    # bracket {98, 99} INCLUSIVE: cdf(99.5) - cdf(97.5) (upper end in normal tail)
    assert nbm.nbm_prob("between", 98.0, 99.0, Q) == pytest.approx(0.320, abs=3e-3)
    assert nbm.nbm_prob("less", None, 98.0, Q) == 0.625
    # ladder identity: less(98) | {98,99} | greater(99) tiles all outcomes
    total = (nbm.nbm_prob("less", None, 98.0, Q)
             + nbm.nbm_prob("between", 98.0, 99.0, Q)
             + nbm.nbm_prob("greater", 99.0, None, Q))
    assert abs(total - 1.0) < 3e-3


def test_nbm_prob_tail_uses_normal_extrapolation():
    p = nbm.nbm_prob("greater", 101.0, None, Q)   # beyond p90
    assert 0.0 < p < 0.10                          # small but not zero
    assert nbm.nbm_prob("greater", 101.0, None, {"p10": 95}) is None  # incomplete


def test_degenerate_spread_guarded():
    flat = {"p10": 97, "p25": 97, "p50": 97, "p75": 97, "p90": 97}
    p = nbm.nbm_prob("greater", 98.0, None, flat)
    assert p is not None and 0.0 <= p < 0.5        # sigma floor prevents div-by-zero
