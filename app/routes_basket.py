# app/routes_basket.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.ai_universe import is_ai_covered_ticker
from app.limits import MAX_BASKET_TICKERS
from app.db import get_db
from app.dependencies import get_current_user
from app.models import User, Basket

router = APIRouter()


class BasketAdd(BaseModel):
    ticker: str
    ticker_name: str


@router.post("")
def add_to_basket(
    payload: BasketAdd,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 중복 체크
    exists = db.query(Basket).filter(
        Basket.user_id == current_user.id,
        Basket.ticker == payload.ticker,
    ).first()

    if exists:
        raise HTTPException(status_code=409, detail="이미 바구니에 있는 종목입니다")

    count = db.query(Basket).filter(Basket.user_id == current_user.id).count()
    if count >= MAX_BASKET_TICKERS:
        raise HTTPException(status_code=400, detail=f"자동매매 종목은 최대 {MAX_BASKET_TICKERS}개까지 담을 수 있습니다")

    if not is_ai_covered_ticker(payload.ticker):
        raise HTTPException(
            status_code=400,
            detail=f"{payload.ticker_name}({payload.ticker})은(는) AI 모델이 분석하지 않는 종목이라 "
                    "자동매매 바구니에 담을 수 없습니다",
        )

    item = Basket(
        user_id=current_user.id,
        ticker=payload.ticker,
        ticker_name=payload.ticker_name,
    )
    db.add(item)
    db.commit()
    return {"message": f"{payload.ticker_name} 추가 완료"}


@router.delete("/{ticker}")
def remove_from_basket(
    ticker: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = db.query(Basket).filter(
        Basket.user_id == current_user.id,
        Basket.ticker == ticker,
    ).first()

    if not item:
        raise HTTPException(status_code=404, detail="바구니에 없는 종목입니다")

    db.delete(item)
    db.commit()
    return {"message": "삭제 완료"}


@router.get("")
def get_basket(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = db.query(Basket).filter(
        Basket.user_id == current_user.id
    ).all()

    return [
        {
            "ticker": item.ticker,
            "ticker_name": item.ticker_name,
            "created_at": item.created_at.isoformat(),
        }
        for item in items
    ]