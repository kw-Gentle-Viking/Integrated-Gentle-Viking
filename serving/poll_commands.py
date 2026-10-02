"""
serving/poll_commands.py
=========================
백엔드의 커맨드 큐(app/routes_ai_command.py, /ai/commands/pending)를 주기적으로 폴링해서 이
프로젝트의 serving/api_server.py(/command, 기본 포트 8001)로 그대로 중계한다.

/home/user/poll_commands.py(운영 원본)는 `from api_server import run_once_inference, ...`로
운영 쪽 /home/user/api_server.py를 직접 import해서 같은 프로세스 안에서 함수를 호출하는 방식이라,
이 프로젝트의 서빙 코드/모델과는 무관하다 -- 그래서 백엔드가 /trade/once로 커맨드를 큐에 넣어도
아무도 가져가지 않아 ONCE 요청이 조용히 아무 일도 안 하고 끝났다(2026-10-03 실제 end-to-end
테스트로 발견). 이 스크립트는 운영 원본을 복사하지 않고, 커맨드를 이 프로젝트의 AI 서버 HTTP
엔드포인트로 그대로 전달하는 단순 릴레이로 새로 작성했다.

환경변수:
    BACKEND_WEBHOOK_URL   : 백엔드 서버 베이스 URL (예: http://localhost:8000)
    AI_SERVER_API_KEY     : 백엔드 <-> 이 프로젝트 AI 서버 공유 키 (조회는 X-API-Key, 로컬 /command
                             전달도 동일 키 사용)
    AI_SERVING_URL        : 이 프로젝트 AI 서버 베이스 URL (기본 http://localhost:8001)
    POLL_INTERVAL_SECONDS : 폴링 주기 (기본 10초)

실행:
    PYTHONPATH=. python -m serving.poll_commands            # 상시 루프
    PYTHONPATH=. python -m serving.poll_commands --once      # 1회만 (크론용)
"""
import logging
import os
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BACKEND_WEBHOOK_URL = os.environ.get("BACKEND_WEBHOOK_URL", "")
AI_SERVER_API_KEY = os.environ.get("AI_SERVER_API_KEY", "dev-ai-key")
AI_SERVING_URL = os.environ.get("AI_SERVING_URL", "http://localhost:8001")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))


def fetch_pending() -> list:
    """백엔드 커맨드 큐에서 미처리 항목 조회 (조회 즉시 backend 쪽에서 delivered 처리됨)."""
    resp = requests.get(
        f"{BACKEND_WEBHOOK_URL}/ai/commands/pending",
        headers={"X-API-Key": AI_SERVER_API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("commands", [])


def relay(cmd: dict) -> None:
    """커맨드를 이 프로젝트 AI 서버의 /command로 그대로 전달 (fire-and-forget: ONCE는 서버가
    백그라운드 태스크로 처리하고 callback_url로 직접 결과를 보낸다)."""
    payload = {
        "command": cmd.get("command", ""),
        "user_id": str(cmd.get("user_id", "")),
        "tickers": cmd.get("tickers", []),
        "callback_url": cmd.get("callback_url") or f"{BACKEND_WEBHOOK_URL}/ai/callback",
    }
    resp = requests.post(
        f"{AI_SERVING_URL}/command",
        json=payload,
        headers={"X-API-Key": AI_SERVER_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    logger.info("relayed %s -> %s (%s)", payload["command"], AI_SERVING_URL, resp.json())


def poll_once() -> None:
    if not BACKEND_WEBHOOK_URL:
        logger.error("BACKEND_WEBHOOK_URL 미설정 -> 건너뜀")
        return
    try:
        commands = fetch_pending()
    except Exception as e:  # noqa: BLE001
        logger.error("폴링 실패: %s", e)
        return
    if not commands:
        return
    logger.info("%d개 커맨드 수신", len(commands))
    for cmd in commands:
        try:
            relay(cmd)
        except Exception as e:  # noqa: BLE001
            logger.error("커맨드 중계 오류: %s / cmd=%s", e, cmd)


def main() -> int:
    if "--once" in sys.argv:
        poll_once()
        return 0
    logger.info("폴링 루프 시작 (주기: %d초, AI 서버: %s)", POLL_INTERVAL, AI_SERVING_URL)
    while True:
        poll_once()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
