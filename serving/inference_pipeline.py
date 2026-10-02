"""
serving/inference_pipeline.py
==============================
Task 18 Step 5. Mirrors /home/user/inference_pipeline.py's cadence (5-min crontab, market-hours
guard 09:00-15:30 KST weekdays-only, weekend guard) but is a direct, single-process Python
implementation on top of this project's own model, rather than shelling out to separate scripts
(the original's subprocess-chaining mechanics are NOT reused, per this task's scope -- only the
cadence/guard pattern is).

For each ticker in the active registry (serving/active_tickers.json -- THIS project's own file,
never /home/user/active_tickers.json):
    1. fetch today's intraday 5-min bars from production stock_db.intraday_5min (read-only --
       that table is written by production's own already-running collector_realtime.py cron job)
    2. serving.feature_builder.build_encoder_df_for_ticker -> serving.inference.run_inference
    3. collect one result dict per ticker
    4. POST the batch to BACKEND_WEBHOOK_URL (same env var name as production's
       push_realtime_results.py, for drop-in compatibility) if set, else log and skip

Payload shape mirrors /home/user/api_server.py's run_once_inference results (same field names,
since the backend team's parser expects those exact keys -- this project's "don't change payload
shape" scoping decision): job_id/user_id/inference_at/results[ticker/pred_label/pred_str/
prob_buy/prob_hold/prob_sell].

crontab line that WOULD run this (NOT installed by this task -- see
task-18-step5-report.md for why, and get explicit user approval before adding it):
    */5 * * * 1-5 /home/user/miniconda3/envs/dl_env/bin/python -m serving.inference_pipeline >> /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/serving/pipeline.log 2>&1
"""

import json
import logging
import os
from datetime import datetime

import requests

from serving.feature_builder import build_encoder_df_for_ticker, fetch_today_intraday_rows
from serving.inference import run_inference
from serving.model import get_feature_columns, get_model, get_model_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_FILE = os.environ.get("SERVING_TICKERS_FILE", os.path.join(SERVING_DIR, "active_tickers.json"))
BACKEND_WEBHOOK_URL = os.environ.get("BACKEND_WEBHOOK_URL", "")
AI_SERVER_API_KEY = os.environ.get("AI_SERVER_API_KEY", "")


def load_active_tickers(tickers_file: str = TICKERS_FILE) -> list[str]:
    if not os.path.exists(tickers_file):
        return []
    try:
        with open(tickers_file) as f:
            return json.load(f).get("all_tickers", [])
    except Exception:
        return []


def is_within_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def run_for_tickers(tickers: list[str], v2_dsn: str, prod_dsn: str, now: datetime | None = None) -> list[dict]:
    """The DB-backed batch path: fetch + build + infer for each ticker. Any single ticker's
    failure is logged and skipped rather than aborting the whole batch (matches production's
    per-user/per-job isolation intent)."""
    now = now or datetime.now()
    model = get_model()
    cols = get_feature_columns()
    results = []
    for ticker in tickers:
        try:
            encoder_df = build_encoder_df_for_ticker(
                ticker, v2_dsn, prod_dsn,
                cols["historical"], cols["future"], cols["static"], now=now,
            )
            pred = run_inference(ticker, encoder_df, model, cols["historical"], cols["future"], cols["static"])
            results.append({"ticker": ticker, "trade_datetime": now.isoformat(),
                            "model_version": get_model_version(), **pred})
        except Exception as e:
            logger.error("%s 추론 실패: %s", ticker, e)
    return results


def push_results(results: list[dict], webhook_url: str = BACKEND_WEBHOOK_URL) -> bool:
    """webhook_url은 백엔드 베이스 URL이다(production push_realtime_results.py와 동일 관례) --
    /ai/realtime 아래로 쳐야 한다. 예전엔 베이스 URL에 그대로 POST해서 백엔드에선 404였고,
    X-API-Key도 안 보내서(설정돼 있었어도) 웹훅의 verify_api_key에 401이었을 것이다."""
    if not webhook_url:
        logger.info("BACKEND_WEBHOOK_URL 미설정 -> push 스킵")
        return False
    payload = {
        "job_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "user_id": "pipeline",
        "inference_at": datetime.now().isoformat(),
        "results": results,
    }
    headers = {}
    api_key = os.environ.get("AI_SERVER_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.post(f"{webhook_url.rstrip('/')}/ai/realtime", json=payload, timeout=15, headers=headers)
    resp.raise_for_status()
    logger.info("push 완료 -> %s (%d건)", resp.status_code, len(results))
    return True


def main() -> None:
    now = datetime.now()
    if now.weekday() >= 5:
        logger.info("주말(%s) -> 건너뜀", now.date())
        return
    if not is_within_market_hours(now):
        logger.info("장외 시간 (%s) -> 건너뜀", now.strftime("%H:%M"))
        return

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    prod_dsn = os.environ.get("PROD_STOCK_DB_DSN") or f"postgresql://{user}:{password}@localhost/stock_db"

    tickers = load_active_tickers()
    if not tickers:
        logger.info("활성 종목 없음 -> 건너뜀")
        return

    logger.info("===== 파이프라인 시작 (%s, %d개 종목) =====", now.strftime("%H:%M"), len(tickers))
    results = run_for_tickers(tickers, dsn, prod_dsn, now=now)
    push_results(results)
    logger.info("===== 완료 (%d/%d건 성공) =====", len(results), len(tickers))


if __name__ == "__main__":
    main()
