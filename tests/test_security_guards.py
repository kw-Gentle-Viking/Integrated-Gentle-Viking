"""GCP 등 외부 공개 배포에서 문제가 되는 인증 공백 수정 테스트."""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


def _client_with(dep):
    app = FastAPI()

    @app.get("/x")
    def x(_=Depends(dep)):
        return {"ok": True}

    return TestClient(app)


# ---- AI 서버/서비스 키 인증: X-API-Key 또는 Bearer (원본 poll_commands.py 는 Bearer 로 보낸다) ----
def test_service_key_accepts_x_api_key_and_bearer_and_rejects_others(monkeypatch):
    from app.security_guards import service_key_auth
    monkeypatch.setenv("AI_SERVER_API_KEY", "k-ai")
    monkeypatch.setenv("AI_COMMAND_API_KEY", "k-cmd")
    c = _client_with(service_key_auth)
    assert c.get("/x", headers={"X-API-Key": "k-ai"}).status_code == 200
    assert c.get("/x", headers={"Authorization": "Bearer k-cmd"}).status_code == 200
    assert c.get("/x", headers={"Authorization": "Bearer k-ai"}).status_code == 200
    assert c.get("/x").status_code == 401
    assert c.get("/x", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/x", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/x", headers={"Authorization": "k-ai"}).status_code == 401      # Bearer 형식이 아님


def test_service_key_falls_back_to_dev_key_only_when_env_unset(monkeypatch):
    from app.security_guards import service_key_auth
    monkeypatch.delenv("AI_SERVER_API_KEY", raising=False)
    monkeypatch.delenv("AI_COMMAND_API_KEY", raising=False)
    assert _client_with(service_key_auth).get("/x", headers={"X-API-Key": "dev-ai-key"}).status_code == 200


# ---- 디버그 엔드포인트는 명시적으로 켜야만 열린다 ----
def test_debug_endpoints_hidden_unless_enabled(monkeypatch):
    from app.security_guards import require_debug_endpoints
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)
    assert _client_with(require_debug_endpoints).get("/x").status_code == 404
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "true")
    assert _client_with(require_debug_endpoints).get("/x").status_code == 200


# ---- 운영 모드에서는 dev 기본 시크릿으로 기동하지 못한다 ----
def test_production_startup_rejects_default_or_missing_secrets():
    from app.security_guards import validate_production_secrets
    with pytest.raises(RuntimeError) as e:
        validate_production_secrets({"APP_ENV": "production", "JWT_SECRET": "dev-secret"})
    msg = str(e.value)
    assert "JWT_SECRET" in msg and "REFRESH_TOKEN_PEPPER" in msg and "AI_SERVER_API_KEY" in msg


def test_production_startup_accepts_real_secrets_and_dev_mode_is_unchecked():
    from app.security_guards import validate_production_secrets
    validate_production_secrets({"APP_ENV": "production", "JWT_SECRET": "a" * 32, "REFRESH_TOKEN_PEPPER": "b" * 32,
                                 "AI_SERVER_API_KEY": "c" * 32})
    validate_production_secrets({})                       # APP_ENV 없음 = 개발 모드: 검사 안 함
    validate_production_secrets({"APP_ENV": "development", "JWT_SECRET": "dev-secret"})


def test_production_startup_rejects_demo_mode_because_it_bypasses_login():
    # get_current_user 는 LOCAL_DEMO_MODE 이면 토큰 없는 요청도 데모 유저로 통과시킨다 -> 운영에서 켜져 있으면 로그인 무력화
    from app.security_guards import validate_production_secrets
    ok = {"APP_ENV": "production", "JWT_SECRET": "a" * 32, "REFRESH_TOKEN_PEPPER": "b" * 32, "AI_SERVER_API_KEY": "c" * 32}
    with pytest.raises(RuntimeError, match="LOCAL_DEMO_MODE"):
        validate_production_secrets({**ok, "LOCAL_DEMO_MODE": "1"})
    validate_production_secrets({**ok, "LOCAL_DEMO_MODE": "false"})


# ---- 라우트 적용 (실제 라우터로 검증) ----
@pytest.fixture()
def users_client(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db import Base, get_db
    import app.models  # noqa: F401
    from app.dependencies import get_current_user
    from app.routes_users import router
    from app.models import User

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    for uid, email in ((1, "a@x.com"), (2, "b@x.com")):
        s.add(User(id=uid, email=email, provider="local", email_verified=True))
    s.commit()

    app = FastAPI()
    app.include_router(router, prefix="/users")
    app.dependency_overrides[get_db] = lambda: s
    holder = {"user": s.get(User, 1)}
    app.dependency_overrides[get_current_user] = lambda: holder["user"]
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)
    return TestClient(app), holder, s


def test_read_user_by_id_is_self_only(users_client):
    client, holder, s = users_client
    assert client.get("/users/1").status_code == 200
    assert client.get("/users/2").status_code == 403       # 남의 개인정보는 조회 불가


def test_list_users_is_hidden_unless_debug_enabled(users_client, monkeypatch):
    client, _, _ = users_client
    assert client.get("/users").status_code == 404         # 회원 목록(이메일/전화/생년월일) 전체 노출 차단
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "true")
    assert client.get("/users").status_code == 200


def test_ai_command_routes_require_service_key(monkeypatch):
    from app.routes_ai_command import router
    monkeypatch.setenv("AI_SERVER_API_KEY", "k-ai")
    monkeypatch.setenv("AI_COMMAND_API_KEY", "k-cmd")
    app = FastAPI()
    app.include_router(router, prefix="/ai")
    c = TestClient(app)
    for method, path in (("get", "/ai/commands/pending"), ("get", "/ai/commands/history")):
        assert getattr(c, method)(path).status_code == 401
        assert getattr(c, method)(path, headers={"Authorization": "Bearer k-cmd"}).status_code == 200   # 원본 poll_commands.py 방식
    assert c.post("/ai/commands/push", json={"command": "STOP", "user_id": 1, "tickers": []}).status_code == 401


def test_price_upsert_requires_service_key():
    from app.routes_prices import router
    app = FastAPI()
    app.include_router(router, prefix="/prices")
    resp = TestClient(app).post("/prices", json={"symbol": "X"})
    assert resp.status_code == 401        # 스키마 검증(422)보다 먼저 인증에서 막혀야 함
