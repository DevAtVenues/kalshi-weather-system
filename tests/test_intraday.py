"""Intraday high-distribution model — the observation-conditioned twin of the ensemble."""
import numpy as np

from kalshi_weather.live.intraday import warming_fraction, intraday_adjust, intraday_prob


def test_warming_fraction_bounds():
    assert warming_fraction(3.0) == 1.0          # pre-dawn: full rise ahead
    assert warming_fraction(6.0) == 1.0          # at sunrise
    assert warming_fraction(15.0) == 0.0         # at peak: no rise left
    assert warming_fraction(20.0) == 0.0         # evening
    mid = warming_fraction(11.0)                 # late morning: partial
    assert 0.0 < mid < 1.0


def test_adjust_floor_never_below_observed():
    highs = np.array([80.0, 85.0, 90.0])
    # even with full remaining warming, nothing drops below the running high
    adj = intraday_adjust(highs, running_high=88.0, frac=1.0)
    assert (adj >= 88.0).all()
    # a member that under-forecast (80 < 88) is pulled up to the floor
    assert adj[0] == 88.0


def test_adjust_collapses_to_observed_at_peak():
    highs = np.array([80.0, 85.0, 92.0])
    adj = intraday_adjust(highs, running_high=87.0, frac=0.0)
    assert np.allclose(adj, 87.0)                # peak: high is whatever we've seen


def test_early_morning_is_forecast_like():
    # Members forecasting an afternoon high ~90; pre-dawn running high 70.
    members = np.array([88.0, 90.0, 91.0, 92.0])
    p = intraday_prob(members, running_high=70.0, local_hour=5.0,
                      strike_type="greater", floor=89.0, cap=None, nws=90.0, sigma=3.0)
    # obs don't bind yet → meaningful chance the 89°+ bucket hits
    assert p is not None and p > 0.4


def test_afternoon_peak_becomes_a_lock():
    members = np.array([88.0, 90.0, 91.0, 92.0])
    # by peak the high is essentially in at 91 → "≥89" is near-certain YES...
    p_yes = intraday_prob(members, running_high=91.0, local_hour=15.5,
                          strike_type="greater", floor=89.0, cap=None, nws=90.0, sigma=3.0)
    assert p_yes is not None and p_yes > 0.9
    # ...and "≥94" is near-certain NO (can't get there post-peak)
    p_no = intraday_prob(members, running_high=91.0, local_hour=15.5,
                         strike_type="greater", floor=94.0, cap=None, nws=90.0, sigma=3.0)
    assert p_no is not None and p_no < 0.1


def test_upside_bucket_prob_decays_through_the_day():
    # An above-current bucket gets less likely as the window to reach it closes.
    members = np.array([90.0, 92.0, 94.0, 96.0])
    common = dict(members=members, running_high=88.0, strike_type="greater",
                  floor=93.0, cap=None, nws=93.0, sigma=3.0)
    p_morning = intraday_prob(local_hour=9.0, **common)
    p_afternoon = intraday_prob(local_hour=14.0, **common)
    assert p_morning is not None and p_afternoon is not None
    assert p_afternoon < p_morning
