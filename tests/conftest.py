"""conftest.py — auto-loads .env for all pytest tests."""

import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root before any test runs
load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.fixture(autouse=True)
def rate_limit_guard(request):
    """Add a small delay between integration tests to avoid Groq rate limits."""
    yield
    # Only sleep after integration tests (test_scenarios.py)
    if "test_scenarios" in request.fspath.basename:
        time.sleep(3)
