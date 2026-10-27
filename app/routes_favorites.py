# app/routes_favorites.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db import get_db
from app.dependencies import get_current_user
from app.models import User, Favorite

router = APIRouter()

MAX_FAVORITES = 100


class FavoriteAdd(BaseModel):
    ticker: str
    ticker_name: str


@router.post("")
def add_favorite(
    payload: FavoriteAdd,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    exists = db.query(Favorite).filter(
        Favorite.user_id == current_user.id,
        Favorite.ticker == payload.ticker,
    ).first()
    if exists:
        raise HTTPException(status_code=409, detail="이미 관심 종목에 있습니다")

    count = db.query(Favorite).filter(Favorite.user_id == current_user.id).count()
    if count >= MAX_FAVORITES:
        raise HTTPException(status_code=400, detail=f"관심 종목은 최대 {MAX_FAVORITES}개까지 담을 수 있습니다")

    item = Favorite(
        user_id=current_user.id,
        ticker=payload.ticker,
        ticker_name=payload.ticker_name,
    )
    db.add(item)
    db.commit()
    return {"message": f"{payload.ticker_name} 관심 종목 추가 완료"}


@router.delete("/{ticker}")
def remove_favorite(
    ticker: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = db.query(Favorite).filter(
        Favorite.user_id == current_user.id,
        Favorite.ticker == ticker,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="관심 종목에 없는 종목입니다")

    db.delete(item)
    db.commit()
    return {"message": "삭제 완료"}


@router.get("")
def get_favorites(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = db.query(Favorite).filter(
        Favorite.user_id == current_user.id
    ).all()

    return [
        {
            "ticker": item.ticker,
            "ticker_name": item.ticker_name,
            "created_at": item.created_at.isoformat(),
        }
        for item in items
    ]
