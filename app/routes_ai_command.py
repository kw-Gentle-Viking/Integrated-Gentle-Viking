# app/routes_ai_command.py
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from collections import defaultdict
from datetime import datetime
from app.schemas import CommandRequest

from app.dependencies import get_current_user
from app.security_guards import service_key_auth
from app.models import User

router = APIRouter()

# 커맨드 큐 (메모리)
command_queue: list[dict] = []




class PredictionRequest(BaseModel):
    tickers: list[str]


@router.post("/predictions/request")
def request_prediction(
    payload: PredictionRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """프론트가 AI 서버에 ONCE 추론을 요청하도록 커맨드 큐에 적재"""
    tickers = [ticker for ticker in payload.tickers if ticker]
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers required")

    job_id = f"once-{current_user.id}-{int(datetime.now().timestamp())}"
    base_url = str(request.base_url).rstrip("/")
    cmd = {
        "command": "ONCE",
        "job_id": job_id,
        "user_id": current_user.id,
        "tickers": tickers,
        "callback_url": f"{base_url}/ai/callback",
        "created_at": datetime.now().isoformat(),
        "status": "pending",
    }
    command_queue.append(cmd)
    return {
        "status": "queued",
        "job_id": job_id,
        "tickers": tickers,
        "callback_url": cmd["callback_url"],
    }


@router.post("/commands/push", dependencies=[Depends(service_key_auth)])
def push_command(payload: CommandRequest):
    """백엔드 내부에서 커맨드 적재 (trade/start 호출 시)"""
    cmd = {
        "command": payload.command,
        "user_id": payload.user_id,
        "tickers": payload.tickers,
        "callback_url": payload.callback_url,
        "warmup": payload.warmup,
        "created_at": datetime.now().isoformat(),
        "status": "pending",
    }
    command_queue.append(cmd)
    return {"status": "queued", "command": payload.command}


@router.get("/commands/pending", dependencies=[Depends(service_key_auth)])
def get_pending_commands():
    """AI 서버가 polling으로 가져감"""
    pending = [c for c in command_queue if c["status"] == "pending"]

    # 가져간 커맨드는 처리 완료로 표시
    for c in pending:
        c["status"] = "delivered"

    return {"commands": pending}


@router.get("/commands/history", dependencies=[Depends(service_key_auth)])
def get_command_history():
    """커맨드 이력 조회 (디버깅용)"""
    return {"commands": command_queue[-50:]}