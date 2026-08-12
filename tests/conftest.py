"""Global test guards.

No test may ever send a real push: preflight's failure path and the health/skill
monitors all publish via NTFY_TOPIC, and the .env autoload (kalshi_weather
__init__) sets it in every process — including pytest. Learned live 2026-07-12:
the preflight fixture failures ('gone.json', 'NOPE_TOKEN') pushed to the phone
on every commit-gate run. Blanking the topic makes _topic() return None, which
turns every notify into a no-op, without touching any production code path.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _no_real_pushes(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "")
