import types
import pandas as pd
from core.data_sources import tiingo

class DummyResp:
    def __init__(self, status_code: int, text: str = "", raise_http: bool = False):
        self.status_code = status_code
        self.text = text or ""
        self._raise = raise_http
    def raise_for_status(self):
        if self._raise and self.status_code >= 400:
            # Simulate requests.HTTPError without needing a real Response object
            from requests import HTTPError
            raise HTTPError(f"HTTP {self.status_code}")

# Monkeypatch pattern inside test to simulate 429 then skip

def test_tiingo_rate_limit_skip(monkeypatch):
    # Reset guard
    tiingo.reset_tiingo_rate_limit()

    calls = {"count": 0}

    def fake_get(url, params=None, timeout=0):
        calls["count"] += 1
        # First call: 429 triggers rate-limit mark
        if calls["count"] == 1:
            return DummyResp(429, "rate limit exceeded", raise_http=True)
        # Subsequent calls would normally fetch, but we expect skip so should not reach here
        return DummyResp(200, "date,close,high,low,open,volume,adjClose\n", raise_http=False)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setenv("TIINGO_API_KEY", "FAKEKEY")
    monkeypatch.setenv("TIINGO_SKIP_ON_RATE_LIMIT", "1")

    # First fetch -> triggers rate limit
    df1 = tiingo.fetch_daily("SPY")
    assert df1.empty, "First call should return empty after 429"
    assert tiingo.tiingo_rate_limited() is True, "Rate limit flag not set"

    # Second fetch -> should be skipped, NOT increment calls
    start_count = calls["count"]
    df2 = tiingo.fetch_daily("AAPL")
    end_count = calls["count"]
    assert df2.empty, "Second call should return empty due to skip"
    assert end_count == start_count, "Second network call executed despite rate-limit skip"
