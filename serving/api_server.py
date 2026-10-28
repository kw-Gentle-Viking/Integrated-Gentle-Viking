"""
serving/api_server.py
======================
Task 18 Step 5. FastAPI app replicating /home/user/api_server.py's EXACT /command
(START/STOP/ONCE) and /health contract, backed by THIS project's own stock_db_v2 +
champion_config.json + serving/best_model_state_dict.pt + serving/feature_builder.py +
serving/inference.py.

Differences from the production file (deliberate, per this task's scope):
    - project-local ticker registry: serving/active_tickers.json (NOT /home/user/active_tickers.json
      -- must never collide with production's file/process)
    - default port 8001, not 8000 (avoids collision if both were ever run simultaneously),
      configurable via SERVING_PORT
    - ONCE builds encoder_df itself via serving.feature_builder for each requested ticker (one
      real feature_pool + intraday_5min read per ticker) instead of the old inference_features/
      realtime_features batch-query path (those tables/columns belonged to the old, abandoned
      55-column design -- see this task's brief)
    - interpretability extraction (extract_interpretability in the original) was NOT built here
      -- explicitly scoped as a nice-to-have Step 5 could skip if time-constrained; skipped, so
      ONCE results do not include an "interpretability" key. Everything else in results
      (ticker/pred_label/pred_str/prob_buy/prob_hold/prob_sell) matches the original's field
      names exactly.

환경변수:
    AI_SERVER_API_KEY   : 인증 키 (X-API-Key 헤더 검증용)
    SERVING_PORT         : uvicorn 포트 (기본 8001)
    STOCK_DB_V2_DSN       : 이 프로젝트 DB (feature_pool 59일 히스토리)
    PROD_STOCK_DB_DSN (선택) : 운영 stock_db read-only DSN. 없으면 DB_USER/DB_PASSWORD로 구성.
"""

import json
import logging
import os
import threading
from datetime import datetime

import requests
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel

from serving.feature_builder import build_encoder_df_for_ticker
from serving.inference import run_inference
from serving.limits import MAX_ACTIVE_TICKERS
from serving.model import get_feature_columns, get_model, get_model_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_server.log")),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
AI_SERVER_API_KEY = os.environ.get("AI_SERVER_API_KEY", "dev-ai-key")  # 백엔드 기본값(dev-ai-key)과 통일 (2026-10-02 통합 감사)
TICKERS_FILE = os.environ.get("SERVING_TICKERS_FILE", os.path.join(SERVING_DIR, "active_tickers.json"))
LABEL_MAP = {0: "매수", 1: "관망", 2: "매도"}

_tickers_lock = threading.Lock()


def get_prod_dsn() -> str:
    dsn = os.environ.get("PROD_STOCK_DB_DSN")
    if dsn:
        return dsn
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    return f"postgresql://{user}:{password}@localhost/stock_db"


# ============================================================
# 종목 관리 (project-local registry -- serving/active_tickers.json)
# ============================================================
def load_tickers() -> dict:
    if not os.path.exists(TICKERS_FILE):
        return {"updated_at": "", "users": {}, "all_tickers": []}
    with open(TICKERS_FILE) as f:
        return json.load(f)


def save_tickers(data: dict) -> None:
    data["updated_at"] = datetime.now().isoformat()
    # 임시 파일에 쓴 뒤 원자적으로 교체한다. 추론 프로세스가 중간 상태(빈 파일)를 읽지 않도록 한다.
    tmp = TICKERS_FILE + ".tmp"
    with _tickers_lock:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, TICKERS_FILE)


# ============================================================
# ONCE 추론
# ============================================================
def run_once_inference(user_id: str, tickers: list[str], callback_url: str, job_id: str) -> None:
    now = datetime.now()
    v2_dsn = os.environ.get("STOCK_DB_V2_DSN")
    prod_dsn = get_prod_dsn()
    model = get_model()
    cols = get_feature_columns()

    results = []
    for ticker in tickers:
        try:
            encoder_df = build_encoder_df_for_ticker(
                ticker, v2_dsn, prod_dsn, cols["historical"], cols["future"], cols["static"], now=now,
            )
            pred = run_inference(ticker, encoder_df, model, cols["historical"], cols["future"], cols["static"])
            results.append({
                "ticker": ticker,
                "trade_datetime": now.isoformat(),
                "pred_label": pred["pred_label"],
                "pred_str": pred["pred_str"],
                "prob_buy": round(pred["prob_buy"], 4),
                "prob_hold": round(pred["prob_hold"], 4),
                "prob_sell": round(pred["prob_sell"], 4),
                "model_version": get_model_version(),
            })
        except Exception as e:
            logger.error("[%s] %s 추론 실패: %s", job_id, ticker, e)

    payload = {
        "job_id": job_id, "user_id": user_id,
        "inference_at": now.isoformat(), "results": results,
    }

    if callback_url:
        try:
            headers = {"Content-Type": "application/json"}
            if AI_SERVER_API_KEY:
                headers["X-API-Key"] = AI_SERVER_API_KEY
            # 백엔드가 콜백 안에서 Gemini 보고서까지 동기 생성한다(ticker당 최대 3회 재시도 +
            # 지수 백오프 포함) -- 15초는 너무 짧아서 로그에 "전송 실패" 오탐이 났었다(2026-10-28
            # 1차 발견, timeout=90으로 올림). 그런데도 쿼터가 빡빡할 때 재시도가 길어지면 90초도
            # 실제로 넘겨서(2026-10-28 재현, 94초+) 같은 오탐이 또 났다 -- 여유 있게 180초로 올림.
            resp = requests.post(callback_url, json=payload, headers=headers, timeout=180)
            resp.raise_for_status()
            logger.info("[%s] callback 전송 완료 -> %s", job_id, resp.status_code)
        except Exception as e:
            logger.error("[%s] callback 전송 실패: %s", job_id, e)


# ============================================================
# FastAPI 앱
# ============================================================
app = FastAPI(title="AI Inference Server (redesign, serving/)")


class CommandRequest(BaseModel):
    command: str
    user_id: str
    tickers: list[str] = []
    callback_url: str = ""
    job_id: str = ""


@app.post("/command")
async def handle_command(
    req: CommandRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=""),
):
    if x_api_key != AI_SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if req.command == "START":
        data = load_tickers()
        data["users"][req.user_id] = req.tickers
        all_t = sorted(set(t for ts in data["users"].values() for t in ts))
        if len(all_t) > MAX_ACTIVE_TICKERS:
            raise HTTPException(status_code=400,
                                detail=f"활성 종목은 최대 {MAX_ACTIVE_TICKERS}개입니다 (현재 요청 후 {len(all_t)}개)")
        data["all_tickers"] = all_t
        save_tickers(data)
        logger.info("START: %s -> %s / 전체: %s", req.user_id, req.tickers, all_t)
        return {
            "status": "ok", "command": "START", "user_id": req.user_id,
            "registered": req.tickers, "all_active_tickers": all_t,
        }

    elif req.command == "STOP":
        data = load_tickers()
        removed = data["users"].pop(req.user_id, [])
        all_t = sorted(set(t for ts in data["users"].values() for t in ts))
        data["all_tickers"] = all_t
        save_tickers(data)
        logger.info("STOP: %s 제거 / 전체: %s", req.user_id, all_t)
        return {
            "status": "ok", "command": "STOP", "user_id": req.user_id,
            "removed": removed, "all_active_tickers": all_t,
        }

    elif req.command == "ONCE":
        # 백엔드가 큐에 넣은 job_id가 있으면 그대로 써야 /ai/once/{job_id} 폴링(프론트 "새로고침")이
        # 결과를 찾을 수 있다 -- 없을 때만(직접 /command 호출 등) 자체 생성한다.
        job_id = req.job_id or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{req.user_id}"
        background_tasks.add_task(run_once_inference, req.user_id, req.tickers, req.callback_url, job_id)
        logger.info("ONCE 접수: %s / tickers=%s", job_id, req.tickers)
        return {"status": "accepted", "job_id": job_id, "estimated_seconds": 5}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {req.command}")


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
