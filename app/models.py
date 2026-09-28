from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String,
    Integer,
    DateTime,
    func,
    ForeignKey,
    UniqueConstraint,
    Boolean,
    Index,
    Date,
    Text,
    JSON,
)

from datetime import datetime, date
from app.db import Base

from typing import Optional

from sqlalchemy import Float, BigInteger, PrimaryKeyConstraint, Index


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)

    password_hash: Mapped[str] = mapped_column(String(255), nullable=True)

    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="local")
    provider_sub: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    nickname: Mapped[str] = mapped_column(String(64), unique=True, nullable=True)

    picture: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )  # 프로필 사진? 구글에 자동 업로드하면 ?
    phone: Mapped[str] = mapped_column(String(20), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)  # 생년월일

    # 투자성향 설문
    investment_goal: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    investment_period: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    risk_tolerance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    investment_experience: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    volatility_preference: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    reports: Mapped[list["RecommendationReport"]] = relationship(back_populates="user")
    
    baskets : Mapped[list["Basket"]] = relationship(back_populates="user") 
    
    trade_logs: Mapped[list["TradeLog"]] = relationship(back_populates="user")


class Persona(Base):
    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    gemini_persona: Mapped[str] = mapped_column(String)
    criteria: Mapped[str] = mapped_column(Text)
    weights: Mapped[dict] = mapped_column(JSON)
    use_value_scout: Mapped[bool] = mapped_column(Boolean)

    reports: Mapped[list["RecommendationReport"]] = relationship(
        back_populates="persona"
    )


class RecommendationReport(Base):
    __tablename__ = "recommendation_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    persona_id: Mapped[int] = mapped_column(ForeignKey("personas.id"))
    report: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    user: Mapped["User"] = relationship(back_populates="reports")
    persona: Mapped["Persona"] = relationship(back_populates="reports")


class SearchKeyword(Base):
    __tablename__ = "search_keywords"

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String)  # 호재_모맨텀, 악재_리스크 등
    keyword: Mapped[str] = mapped_column(String)


class ValueWatchlist(Base):
    __tablename__ = "value_watchlist"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_name: Mapped[str] = mapped_column(String)


class StopWord(Base):
    __tablename__ = "stop_words"

    id: Mapped[int] = mapped_column(primary_key=True)
    word: Mapped[str] = mapped_column(String, unique=True)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # token 고유 id (랜덤)
    jti: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # token_hash : refresh token 원문이 아니라 hash
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # 생성 시각 만료 시각 폐기 시각
    created_at: Mapped["datetime"] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped["datetime"] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    revoked_at: Mapped[Optional["datetime"]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Rotate 되면서 대체된 다음 token jti
    replaced_by_jti: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # 감사 보안
    # user_agent: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


Index("idx_refresh_tokens_user_id", RefreshToken.user_id)


class MarketPrice(Base):
    __tablename__ = "market_price"
    __table_args__ = (
        PrimaryKeyConstraint("symbol", "ts", name="pk_market_price"),
        Index("idx_market_price_symbol_ts", "symbol", "ts"),
    )

    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    ts: Mapped["datetime"] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="KIS")


# Basket 자동매매 들어갈 종목들 임의 설정도 가능
class Basket(Base):
    __tablename__ = "baskets"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    ticker: Mapped[str] = mapped_column(String(10))
    ticker_name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    user: Mapped["User"] = relationship(back_populates="baskets")


class ManualTradeLock(Base):
    __tablename__ = "manual_trade_locks"
    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_manual_trade_locks_user_ticker"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    ticker: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AutoTradeDecision(Base):
    __tablename__ = "auto_trade_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    ticker: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(20))  # HOLD | SKIP | ORDER_SUBMITTED
    reason: Mapped[str] = mapped_column(String(80))
    detail: Mapped[str] = mapped_column(Text, default="")
    ai_signal: Mapped[str] = mapped_column(String(10), default="")
    ai_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    strategy_id: Mapped[str] = mapped_column(String(50), default="")
    price: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class TradeLog(Base):
    __tablename__ = "trade_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    ticker: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(10))       # BUY | SELL
    qty: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)         # qty * price
    ai_signal: Mapped[str] = mapped_column(String(10))   # AI 시그널
    ai_confidence: Mapped[float] = mapped_column(Float)
    strategy_id: Mapped[str] = mapped_column(String(50)) # rsi_reversal 등
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    status : Mapped[str] = mapped_column(String(10), default="FILLED") # FILLED | FAILED
    
    user: Mapped["User"] = relationship(back_populates="trade_logs")


class LiveCandle(Base):
    __tablename__ = "live_candles"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10))
    timeframe: Mapped[str] = mapped_column(String(5))  # 5m
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    trade_datetime: Mapped[datetime] = mapped_column(DateTime)

class AIPredictionHistory(Base):
    """AI 서버가 push 한 예측의 이력. (realtime_predictions 는 종목별 최신 1건만 메모리에 덮어써서 백테스트/감사에 못 썼다)"""
    __tablename__ = "ai_prediction_history"
    __table_args__ = (Index("ix_ai_pred_ticker_dt", "ticker", "trade_datetime"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10))
    trade_datetime: Mapped[datetime] = mapped_column(DateTime)   # 예측 시각 (KST naive)
    signal: Mapped[str] = mapped_column(String(10))              # BUY | HOLD | SELL
    prob_buy: Mapped[float] = mapped_column(Float, default=0.0)
    prob_hold: Mapped[float] = mapped_column(Float, default=0.0)
    prob_sell: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str] = mapped_column(String(50), default="unknown")
    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
