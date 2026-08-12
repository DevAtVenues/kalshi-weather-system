"""Market-shrinkage posterior: log-odds blend semantics, fit recovery on
synthetic data, population discipline (dedup, degenerate quotes), and the
None-means-unknown contract of shrink_prob."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kalshi_weather.calibration.shrinkage import blend, shrink_prob
import build_shrinkage as bs


def test_blend_endpoints_recover_inputs():
    assert abs(blend(0.30, 0.60, 1.0) - 0.30) < 1e-6   # w=1 → model
    assert abs(blend(0.30, 0.60, 0.0) - 0.60) < 1e-6   # w=0 → market
    mid = blend(0.30, 0.60, 0.5)
    assert 0.30 < mid < 0.60                            # in between, not average
    # log-odds blend is symmetric: blend(p, m, w) = 1 - blend(1-p, 1-m, w)
    assert abs(blend(0.30, 0.60, 0.4) - (1 - blend(0.70, 0.40, 0.4))) < 1e-9


def test_fit_recovers_true_weight():
    rng = np.random.default_rng(7)
    pop = []
    for i in range(4000):
        pm = float(rng.uniform(0.05, 0.95))
        mk = float(np.clip(pm + rng.normal(0, 0.15), 0.02, 0.98))
        true_p = blend(pm, mk, 0.7)
        pop.append({"x": pm, "m": mk,
                    "y": 1.0 if rng.random() < true_p else 0.0,
                    "d": f"2026-06-{(i % 20) + 1:02d}"})
    w = bs.fit_w(pop)
    assert 0.55 <= w <= 0.85, f"fit {w} should recover ~0.7"


def test_population_dedups_and_drops_degenerate_quotes(tmp_path, monkeypatch):
    lines = [
        # two reads of one ticker — only the LAST counts
        {"run_ts": "a", "ticker": "T1", "prob_source": "ensemble",
         "hours_to_settle": 20.0, "prob_estimate": 0.3, "market_mid": 0.5,
         "yes_settled": True, "settlement_date": "2026-07-01"},
        {"run_ts": "b", "ticker": "T1", "prob_source": "ensemble",
         "hours_to_settle": 18.0, "prob_estimate": 0.4, "market_mid": 0.55,
         "yes_settled": True, "settlement_date": "2026-07-01"},
        # degenerate quote — dropped
        {"run_ts": "c", "ticker": "T2", "prob_source": "ensemble",
         "hours_to_settle": 20.0, "prob_estimate": 0.4, "market_mid": 0.999,
         "yes_settled": False, "settlement_date": "2026-07-01"},
        # wrong lead bucket — dropped
        {"run_ts": "d", "ticker": "T3", "prob_source": "ensemble",
         "hours_to_settle": 5.0, "prob_estimate": 0.4, "market_mid": 0.5,
         "yes_settled": False, "settlement_date": "2026-07-01"},
        # ungraded — dropped
        {"run_ts": "e", "ticker": "T4", "prob_source": "ensemble",
         "hours_to_settle": 20.0, "prob_estimate": 0.4, "market_mid": 0.5,
         "yes_settled": None, "settlement_date": "2026-07-02"},
    ]
    p = tmp_path / "outcomes.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    monkeypatch.setattr(bs, "OUTCOMES", p)
    pop = bs.load_population("ensemble", "day-ahead")
    assert len(pop) == 1
    assert pop[0]["x"] == 0.4 and pop[0]["m"] == 0.55


def test_shrink_prob_none_contract():
    maps = {"weights": {"ensemble@day-ahead": {"w": 0.4}}}
    assert shrink_prob(0.3, 0.6, "ensemble", "day-ahead", maps) is not None
    assert shrink_prob(0.3, 0.6, "ensemble", "same-day", maps) is None  # no key
    assert shrink_prob(0.3, 0.6, "rule", "day-ahead", maps) is None
    assert shrink_prob(None, 0.6, "ensemble", "day-ahead", maps) is None
    assert shrink_prob(0.3, None, "ensemble", "day-ahead", maps) is None
    assert shrink_prob(0.3, 0.6, "ensemble", None, maps) is None       # no lead
    assert shrink_prob(0.3, 0.6, "ensemble", "day-ahead", {}) is None  # no artifact


def test_shrunk_prob_moves_toward_market():
    maps = {"weights": {"ensemble@day-ahead": {"w": 0.4}}}
    p = shrink_prob(0.10, 0.50, "ensemble", "day-ahead", maps)
    assert 0.10 < p < 0.50            # strictly between model and market
    # exact log-odds blend: sigmoid(0.4·logit(0.10) + 0.6·logit(0.50)) ≈ 0.293
    expected = 1 / (1 + np.exp(-(0.4 * np.log(0.1 / 0.9) + 0.6 * 0.0)))
    assert abs(p - expected) < 1e-9
