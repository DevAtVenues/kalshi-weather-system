"""Load the project .env on first import of the package.

launchd agents and cron jobs do NOT inherit shell env, and the crontab lines never
sourced .env — so NTFY_TOPIC/KALSHI_API_KEY were absent in scheduled contexts and
every push silently no-oped (notify_transition even returns True with no topic, so
run_live's log said "pushed"). One load here covers every entry point; existing
environment variables win (load_dotenv does not override).
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
