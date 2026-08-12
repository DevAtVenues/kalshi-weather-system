"""Once-per-window run guard — cron + launchd must not double-fire, but a missed
run must never be blocked on doubt (fail-open)."""
import time

from kalshi_weather.scheduling import acquire_slot, mark_and_release


def test_first_run_acquires(tmp_path):
    slot = acquire_slot(tmp_path, "job", min_gap_s=1800)
    assert slot is not None
    mark_and_release(tmp_path, "job", slot)


def test_second_run_within_gap_skips(tmp_path):
    s1 = acquire_slot(tmp_path, "job", min_gap_s=1800)
    mark_and_release(tmp_path, "job", s1)
    # immediate re-fire (cron + launchd same window) → skipped
    assert acquire_slot(tmp_path, "job", min_gap_s=1800) is None


def test_run_after_gap_allowed(tmp_path):
    s1 = acquire_slot(tmp_path, "job", min_gap_s=1800)
    mark_and_release(tmp_path, "job", s1)
    # marker older than the gap → next scheduled window runs
    (tmp_path / "job.last_run").write_text(str(time.time() - 3600))
    s2 = acquire_slot(tmp_path, "job", min_gap_s=1800)
    assert s2 is not None
    mark_and_release(tmp_path, "job", s2)


def test_force_overrides_gap(tmp_path):
    s1 = acquire_slot(tmp_path, "job", min_gap_s=1800)
    mark_and_release(tmp_path, "job", s1)
    s2 = acquire_slot(tmp_path, "job", min_gap_s=1800, force=True)
    assert s2 is not None
    mark_and_release(tmp_path, "job", s2)


def test_concurrent_instance_skips(tmp_path):
    held = acquire_slot(tmp_path, "job", min_gap_s=1800)     # instance A holds the lock
    assert held is not None
    assert acquire_slot(tmp_path, "job", min_gap_s=1800) is None   # instance B backs off
    mark_and_release(tmp_path, "job", held)


def test_unreadable_marker_fails_open(tmp_path):
    (tmp_path / "job.last_run").write_text("not-a-number")
    slot = acquire_slot(tmp_path, "job", min_gap_s=1800)
    assert slot is not None      # doubt → run, never silently skip
    mark_and_release(tmp_path, "job", slot)
