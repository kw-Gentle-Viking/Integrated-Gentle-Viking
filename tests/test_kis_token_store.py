import asyncio
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

import app.routes_kis as rk
from app.kis_token_store import is_rate_limited, load_token, save_token


def test_token_roundtrip_and_expiry(tmp_path):
    p = tmp_path / "t.json"
    now = datetime(2026, 10, 5, 10, 0)
    save_token(p, "TOK", now + timedelta(hours=1))
    assert load_token(p, now) == ("TOK", now + timedelta(hours=1))
    assert load_token(p, now + timedelta(hours=2)) is None  # 만료


def test_missing_or_corrupt_file_is_treated_as_no_token(tmp_path):
    now = datetime(2026, 10, 5, 10, 0)
    assert load_token(tmp_path / "none.json", now) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_token(bad, now) is None


def test_rate_limit_detection():
    assert is_rate_limited('{"error_code":"EGW00133"}')
    assert not is_rate_limited('{"error_code":"EGW00002"}')


@pytest.fixture()
def clean_state(monkeypatch):
    monkeypatch.setitem(rk._real_token_cache, "token", None)
    monkeypatch.setitem(rk._real_token_cache, "expires_at", None)
    monkeypatch.setitem(rk._last_volume_rank, "data", None)
    monkeypatch.setitem(rk._last_volume_rank, "at", None)
    monkeypatch.setattr(rk, "demo_mode_enabled", lambda: False)


def test_restart_uses_stored_token_without_calling_kis(tmp_path, monkeypatch, clean_state):
    p = tmp_path / "t.json"
    save_token(p, "STORED", datetime.now() + timedelta(hours=5))
    monkeypatch.setenv("KIS_REAL_TOKEN_FILE", str(p))

    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("KIS must not be called when a valid token is stored")

    monkeypatch.setattr(rk.httpx, "AsyncClient", _NoNetwork)
    assert asyncio.run(rk.get_real_access_token()) == "STORED"


def test_volume_rank_returns_last_good_result_when_token_is_rate_limited(monkeypatch, clean_state):
    good = {"rt_cd": "0", "output": [{"mksc_shrn_iscd": "005930"}]}
    rk._last_volume_rank.update(data=good, at=datetime.now())

    async def limited():
        raise HTTPException(status_code=503, detail="잠시 후 다시 시도하세요.")

    monkeypatch.setattr(rk, "_fetch_volume_rank", limited)
    assert asyncio.run(rk.get_volume_rank()) == good


def test_volume_rank_raises_friendly_error_when_nothing_cached(monkeypatch, clean_state):
    async def limited():
        raise HTTPException(status_code=503, detail="잠시 후 다시 시도하세요.")

    monkeypatch.setattr(rk, "_fetch_volume_rank", limited)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(rk.get_volume_rank())
    assert exc.value.status_code == 503
    assert "EGW" not in str(exc.value.detail)
