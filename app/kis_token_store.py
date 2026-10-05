"""KIS 실전 접근토큰을 파일에 저장한다. 토큰 발급은 1분에 1회로 제한돼서(EGW00133), 프로세스를 재시작할
때마다 새로 받으면 바로 제한에 걸린다. 토큰은 24시간 유효하므로 만료 전이면 파일 값을 그대로 쓴다."""
import json
import os
from datetime import datetime
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".cache" / "kis_real_token.json"

RATE_LIMIT_MSG_CD = "EGW00133"


def token_file_path() -> Path:
    return Path(os.getenv("KIS_REAL_TOKEN_FILE") or _DEFAULT_PATH)


def load_token(path: Path, now: datetime) -> tuple[str, datetime] | None:
    """만료 전 토큰이면 (token, expires_at), 없거나 깨졌거나 만료됐으면 None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expires_at = datetime.fromisoformat(data["expires_at"])
        token = data["token"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not token or expires_at <= now:
        return None
    return token, expires_at


def save_token(path: Path, token: str, expires_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"token": token, "expires_at": expires_at.isoformat()}), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def is_rate_limited(response_text: str) -> bool:
    return RATE_LIMIT_MSG_CD in (response_text or "")
