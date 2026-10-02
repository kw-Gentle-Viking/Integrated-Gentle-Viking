"""백엔드 /trade/once가 command_queue에 넣은 커맨드를 아무도 가져가지 않아 ONCE 요청이 조용히
아무 일도 안 하고 끝났다 -- 운영 원본 poll_commands.py는 production api_server.py를 직접
import해서 이 프로젝트 서빙과는 무관했다(2026-10-03 실제 end-to-end 테스트로 발견)."""
import serving.poll_commands as pc


def test_relay_forwards_command_shape_to_local_ai_server(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"status": "accepted"}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(pc.requests, "post", fake_post)
    monkeypatch.setattr(pc, "AI_SERVING_URL", "http://localhost:8001")
    monkeypatch.setattr(pc, "AI_SERVER_API_KEY", "test-key")
    monkeypatch.setattr(pc, "BACKEND_WEBHOOK_URL", "http://backend.example")

    pc.relay({"command": "ONCE", "user_id": 1, "tickers": ["005930"], "callback_url": None})

    assert captured["url"] == "http://localhost:8001/command"
    assert captured["json"] == {
        "command": "ONCE", "user_id": "1", "tickers": ["005930"],
        "callback_url": "http://backend.example/ai/callback",
    }
    assert captured["headers"] == {"X-API-Key": "test-key"}


def test_relay_preserves_explicit_callback_url(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {}

    monkeypatch.setattr(pc.requests, "post", lambda url, json, headers, timeout: (captured.update(json=json), FakeResp())[1])
    monkeypatch.setattr(pc, "AI_SERVING_URL", "http://localhost:8001")
    monkeypatch.setattr(pc, "AI_SERVER_API_KEY", "test-key")

    pc.relay({"command": "ONCE", "user_id": 2, "tickers": ["000660"], "callback_url": "http://x/custom"})
    assert captured["json"]["callback_url"] == "http://x/custom"


def test_poll_once_skips_when_backend_url_unset(monkeypatch):
    monkeypatch.setattr(pc, "BACKEND_WEBHOOK_URL", "")
    calls = []
    monkeypatch.setattr(pc.requests, "get", lambda *a, **k: calls.append(1))
    pc.poll_once()
    assert calls == []


def test_poll_once_relays_each_pending_command(monkeypatch):
    monkeypatch.setattr(pc, "BACKEND_WEBHOOK_URL", "http://backend.example")

    class FakeGetResp:
        def raise_for_status(self): pass
        def json(self): return {"commands": [{"command": "ONCE", "user_id": 1, "tickers": ["005930"]}]}

    monkeypatch.setattr(pc.requests, "get", lambda *a, **k: FakeGetResp())
    relayed = []
    monkeypatch.setattr(pc, "relay", lambda cmd: relayed.append(cmd))
    pc.poll_once()
    assert len(relayed) == 1 and relayed[0]["command"] == "ONCE"


def test_poll_once_one_bad_command_does_not_block_others(monkeypatch):
    monkeypatch.setattr(pc, "BACKEND_WEBHOOK_URL", "http://backend.example")

    class FakeGetResp:
        def raise_for_status(self): pass
        def json(self):
            return {"commands": [{"command": "ONCE", "user_id": 1}, {"command": "ONCE", "user_id": 2}]}

    monkeypatch.setattr(pc.requests, "get", lambda *a, **k: FakeGetResp())
    relayed = []
    def flaky_relay(cmd):
        if cmd["user_id"] == 1:
            raise RuntimeError("boom")
        relayed.append(cmd)
    monkeypatch.setattr(pc, "relay", flaky_relay)
    pc.poll_once()
    assert len(relayed) == 1 and relayed[0]["user_id"] == 2
