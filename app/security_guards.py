"""외부(GCP 등)에 공개할 때 필요한 인증/기동 가드."""
import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException

_DEV_DEFAULTS = {
    "JWT_SECRET": "dev-secret",
    "REFRESH_TOKEN_PEPPER": "dev-pepper",
    "AI_SERVER_API_KEY": "dev-ai-key",
}
_TRUTHY = {"1", "true", "yes", "y"}
_PRODUCTION_ENVS = {"production", "prod"}


def _eq(provided: Optional[str], expected: Optional[str]) -> bool:
    return bool(provided) and bool(expected) and hmac.compare_digest(str(provided), str(expected))


def service_key_auth(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> None:
    """AI 서버/내부 서비스가 호출하는 API 용 키 인증.

    - X-API-Key: AI_SERVER_API_KEY   (웹훅과 같은 방식)
    - Authorization: Bearer <AI_COMMAND_API_KEY 또는 AI_SERVER_API_KEY>
      (원본 poll_commands.py 가 GCP_BACKEND_API_KEY 를 Bearer 로 보낸다 -> 그 값을 AI_COMMAND_API_KEY 로 설정)
    환경변수가 없으면 웹훅과 동일하게 개발용 기본 키(dev-ai-key)로 동작한다(운영은 기동 시 기본값을 거부).
    """
    ai_key = os.getenv("AI_SERVER_API_KEY", _DEV_DEFAULTS["AI_SERVER_API_KEY"])
    cmd_key = os.getenv("AI_COMMAND_API_KEY", "")

    if _eq(x_api_key, ai_key):
        return
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
        if _eq(bearer, ai_key) or _eq(bearer, cmd_key):
            return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def require_debug_endpoints() -> None:
    """개발/디버깅용 엔드포인트: ENABLE_DEBUG_ENDPOINTS=true 일 때만 존재하는 것처럼 동작(그 외 404)."""
    if os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").lower() not in _TRUTHY:
        raise HTTPException(status_code=404, detail="Not Found")


def validate_production_secrets(env=None) -> None:
    """APP_ENV=production 일 때 개발용 기본 시크릿/데모 모드로는 기동하지 못하게 한다."""
    env = os.environ if env is None else env
    if str(env.get("APP_ENV", "")).lower() not in _PRODUCTION_ENVS:
        return
    problems = []
    for name, dev_value in _DEV_DEFAULTS.items():
        value = env.get(name, "")
        if not value or value == dev_value:
            problems.append(f"{name} (미설정 또는 개발용 기본값)")
    if str(env.get("LOCAL_DEMO_MODE", "false")).lower() in _TRUTHY:
        problems.append("LOCAL_DEMO_MODE (켜져 있으면 토큰 없이도 데모 유저로 로그인 처리됨)")
    if problems:
        raise RuntimeError("운영(APP_ENV=production) 기동 거부 -- 다음을 설정/해제하세요: " + ", ".join(problems))
