"""Preflight: scheduled jobs must refuse to run hollow — and must not block a
healthy run."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather import preflight as pf


def test_healthy_environment_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_ROOT", tmp_path)
    monkeypatch.setenv("SOME_TOKEN", "x")
    (tmp_path / "artifact.json").write_text("{}")
    monkeypatch.setattr(pf, "_REQUIREMENTS", {"job": [
        ("env", "SOME_TOKEN", "t"),
        ("file", "artifact.json", "t"),
        ("dir", "out", "t"),
    ]})
    pf.preflight("job")                       # must not raise/exit
    assert (tmp_path / "out").is_dir()        # dirs get created, not failed


def test_missing_env_exits_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_ROOT", tmp_path)
    monkeypatch.delenv("NOPE_TOKEN", raising=False)
    monkeypatch.setattr(pf, "_REQUIREMENTS", {"job": [("env", "NOPE_TOKEN", "t")]})
    with pytest.raises(SystemExit) as e:
        pf.preflight("job")
    assert e.value.code == 78


def test_missing_file_exits_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_ROOT", tmp_path)
    monkeypatch.setattr(pf, "_REQUIREMENTS", {"job": [("file", "gone.json", "t")]})
    with pytest.raises(SystemExit) as e:
        pf.preflight("job")
    assert e.value.code == 78


def test_unknown_job_is_noop():
    pf.preflight("job-with-no-requirements")


def test_real_requirements_reference_real_things():
    """The requirement TABLE itself can rot (paths renamed). Every file/dir target
    for every job must exist in the actual repo right now."""
    for job, reqs in pf._REQUIREMENTS.items():
        for kind, target, _ in reqs:
            if kind in ("file", "dir"):
                assert (pf._ROOT / target).exists(), f"{job}: {target} does not exist"
