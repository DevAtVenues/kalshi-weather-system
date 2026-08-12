"""Minimax-family EV advisory: per-family P(YES) from raw member parquets,
settlement-convention lockstep with the pipeline, coverage guard, and the
worst-family EV / direction-unanimity semantics proven by hand on 2026-07-20
(MSP split → stand down; SAT unanimous → tradeable)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kalshi_weather.calibration import family_ev as fe
import candidate_pipeline as cp

WIN = (pd.Timestamp("2026-07-21 06:00", tz="UTC"),
       pd.Timestamp("2026-07-22 06:00", tz="UTC"))


def _write_family(root: Path, subdir: str, init: str, station: str,
                  member_peaks: list[float], hours: int = 24):
    """Synthetic hourly member file: each member ramps to its peak mid-window."""
    rows = []
    for m, peak in enumerate(member_peaks, start=1):
        for h in range(hours):
            t = WIN[0] + pd.Timedelta(hours=h)
            # triangular profile peaking at hour 12
            frac = 1.0 - abs(h - 12) / 12.0
            rows.append({"valid_time": t, "member": m,
                         "temp_f": peak - 15.0 + 15.0 * frac})
    d = root / subdir / init
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / f"{station}.parquet")


def test_settlement_convention_locksteps_with_pipeline():
    for st, fl, cap in (("greater", 87.0, None), ("less", None, 91.0),
                        ("between", 84.0, 85.0)):
        for h in (83.4, 83.6, 84.0, 84.4, 84.6, 85.0, 86.4, 86.6, 87.0, 90.4, 91.0):
            assert fe.settles_yes(st, fl, cap, h) == cp._settles_yes(st, fl, cap, h)


def test_family_probs_and_split_detection(tmp_path):
    # GFS warm (all members settle >= 99), ICON cool (all < 99) → SPLIT on T99
    _write_family(tmp_path, "gfs025", "20260720_18Z", "KTST", [99.6, 100.2, 101.0])
    _write_family(tmp_path, "icon_seamless", "20260720_18Z", "KTST", [96.0, 97.1, 97.9])
    fp = fe.family_probs("KTST", "2026-07-21", "greater", 99.0, None,
                         root=tmp_path, window=WIN)
    assert fp["gfs"]["p_yes"] == 1.0 and fp["icon"]["p_yes"] == 0.0
    assert fp["gfs"]["n"] == 3

    # BUY_NO at mid 0.5: hostile family is gfs (p_yes=1 → EV = 0.5 − 1 = −0.5)
    ev = fe.minimax_family_ev(fp, "BUY_NO", 0.5)
    assert ev["worst_family"] == "gfs" and ev["worst_ev"] == -0.5
    assert ev["per_family_ev"]["icon"] == 0.5
    assert not ev["direction_unanimous"]


def test_unanimous_direction_positive_worst_ev(tmp_path):
    # All families price YES below 50% → unanimous NO, +EV under every family
    _write_family(tmp_path, "gfs025", "20260720_18Z", "KTST", [97.0, 98.4, 99.6])
    _write_family(tmp_path, "ecmwf_ifs025", "20260720_12Z", "KTST", [96.0, 97.0, 98.2])
    fp = fe.family_probs("KTST", "2026-07-21", "between", 98.0, 99.0,
                         root=tmp_path, window=WIN)
    assert max(d["p_yes"] for d in fp.values()) < 0.5
    ev = fe.minimax_family_ev(fp, "BUY_NO", 0.45)
    assert ev["direction_unanimous"]
    assert ev["worst_ev"] > 0


def test_int_rounding_settlement_boundary(tmp_path):
    # greater is STRICT: T99-greater pays on int high >= 100. 99.6 rounds to
    # 100 (YES); 98.6 rounds to 99 (NO — equals the floor); 98.4 -> 98 (NO).
    _write_family(tmp_path, "gfs025", "20260720_18Z", "KTST", [99.6, 98.6, 98.4])
    fp = fe.family_probs("KTST", "2026-07-21", "greater", 99.0, None,
                         root=tmp_path, window=WIN)
    assert abs(fp["gfs"]["p_yes"] - 1 / 3) < 1e-3   # p_yes rounded to 3 dp


def test_newest_covering_init_wins(tmp_path):
    _write_family(tmp_path, "gfs025", "20260720_06Z", "KTST", [90.0, 90.0, 90.0])
    _write_family(tmp_path, "gfs025", "20260720_18Z", "KTST", [100.0, 100.0, 100.0])
    fams = fe.family_maxes("KTST", "2026-07-21", root=tmp_path, window=WIN)
    assert fams["gfs"]["init"] == "20260720_18Z"
    assert float(np.mean(fams["gfs"]["maxes"])) > 95


def test_truncated_horizon_falls_back_to_older_init(tmp_path):
    # Newest init stops 8h before window end (misses the afternoon peak) —
    # must fall back to the older init that fully covers the day.
    _write_family(tmp_path, "gfs025", "20260720_06Z", "KTST", [90.0, 90.0, 90.0])
    _write_family(tmp_path, "gfs025", "20260720_18Z", "KTST",
                  [100.0, 100.0, 100.0], hours=16)
    fams = fe.family_maxes("KTST", "2026-07-21", root=tmp_path, window=WIN)
    assert fams["gfs"]["init"] == "20260720_06Z"


def test_fewer_than_two_families_is_inconclusive(tmp_path):
    _write_family(tmp_path, "gfs025", "20260720_18Z", "KTST", [97.0, 98.0, 99.0])
    fp = fe.family_probs("KTST", "2026-07-21", "greater", 99.0, None,
                         root=tmp_path, window=WIN)
    assert fe.minimax_family_ev(fp, "BUY_NO", 0.5) is None
    assert fe.minimax_family_ev({}, "BUY_NO", 0.5) is None
