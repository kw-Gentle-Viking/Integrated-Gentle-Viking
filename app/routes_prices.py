from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from app.db import get_db
from app.security_guards import service_key_auth
from app.schemas import MarketPriceCreate, MarketPriceRead
from app.crud_prices import upsert_price, get_prices

router = APIRouter()

@router.post("", response_model=MarketPriceRead, dependencies=[Depends(service_key_auth)])
def create_or_update(payload: MarketPriceCreate, db: Session = Depends(get_db)):
    return upsert_price(db, payload)

@router.get("", response_model=list[MarketPriceRead])
def read_prices(symbol: str, start: datetime, end: datetime, limit: int = 1000, db: Session = Depends(get_db)):
    return get_prices(db, symbol, start, end, limit)