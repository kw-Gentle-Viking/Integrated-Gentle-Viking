from pydantic import BaseModel, EmailStr
from datetime import datetime, date
from pydantic import constr
from typing import Optional, Literal

from enum import IntEnum


class InvestmentGoal(IntEnum):
    CAPITAL_PRESERVATION = 1  # 원금 보존
    STABLE_RETURN = 2  # 안정적 수익
    GROWTH = 3  # 수익 추구
    HIGH_RETURN = 4  # 고수익 추구


class InvestmentPeriod(IntEnum):
    UNDER_3M = 1  # 3개월 미만
    THREE_TO_12M = 2  # 3개월에서 12개월
    ONE_TO_3Y = 3  # 1년에서 3년
    OVER_3Y = 4  # 3년이상


class RiskTolerance(IntEnum):
    VERY_LOW = 1  # -5%만 돼도 불안
    LOW = 2  # -10%까지 가능
    MEDIUM = 3  # -20%까지 가능
    HIGH = 4  # -30% 이상도 가능


class InvestmentExperience(IntEnum):
    NONE = 1  # 경험 없음
    SAVINGS = 2  # 예·적금 위주
    STOCKS = 3  # 주식·ETF
    DERIVATIVES = 4  # 파생/코인 포함


class VolatilityPreference(IntEnum):
    LOW = 1  # 낮은 변동성
    MEDIUM = 2  # 중간
    HIGH = 3  # 높은 변동성


class UserCreate(BaseModel):
    name: str
    nickname: str
    email: EmailStr
    phone: str
    password: constr(min_length=8, max_length=64)  # 평문은 입력만 받고 저장은 hash로
    birth_date: date

    investment_goal: InvestmentGoal
    investment_period: InvestmentPeriod
    risk_tolerance: RiskTolerance
    investment_experience: InvestmentExperience
    volatility_preference: VolatilityPreference


class UserProfileUpdate(BaseModel):
    nickname: Optional[str] = None
    phone: Optional[str] = None
    investment_goal: Optional[InvestmentGoal] = None
    investment_period: Optional[InvestmentPeriod] = None
    risk_tolerance: Optional[RiskTolerance] = None
    investment_experience: Optional[InvestmentExperience] = None
    volatility_preference: Optional[VolatilityPreference] = None


class UserRead(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str]
    nickname: Optional[str] = None
    phone: Optional[str] = None
    birth_date: Optional[date]
    picture: Optional[str]
    provider: str
    email_verified: bool

    # 투자성향
    investment_goal: Optional[int]
    investment_period: Optional[int]
    risk_tolerance: Optional[int]
    investment_experience: Optional[int]
    volatility_preference: Optional[int]
    risk_score: Optional[int]

    created_at: datetime

    class Config:
        from_attributes = True


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MarketPriceCreate(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str = "KIS"


class MarketPriceRead(MarketPriceCreate):
    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    report: str
    timestamp: Optional[str] = None

# Portfolio AI confidence + 페르소나 기반 구현
class AllocationConfig(BaseModel):
    max_weight: float = 0.30          # 한 종목 최대 비중 (기본 30%)
    cash_reserve: float = 0.10        # 현금 보유 비중 (기본 10%)
    min_confidence: float = 0.40      # 최소 확신도 (기본 40%, 2026-10-06: 3분류 모델의 최대 확률이 0.35~0.42라 60%는 사실상 거래 0건)
    use_persona_boost: bool = True    # 페르소나 보정 사용 여부

class TickerStrategy(BaseModel):
    ticker: str
    strategy_id: str = "rsi_reversal"
    params: Optional[dict] = None

class TradeStartRequest(BaseModel):
    tickers: list[str]
    allocation: Optional[AllocationConfig] = None  # 없으면 기본값 사용


#  추후 사용가능? 
# class StrategyConfig(BaseModel):
#     strategy_id: str = "rsi_reversal"  # rsi_reversal | ma_cross
#     params: Optional[dict] = None       # 전략별 파라미터

    # 예시:
    # rsi_reversal: {"period": 14, "oversold": 30, "overbought": 70}
    # ma_cross:     {"fast_period": 5, "slow_period": 20}

# trade
class TradeRequest(BaseModel):
    total_capital: int = 10_000_000
    strategy: Optional[list[TickerStrategy]] = None
    allocation: Optional[AllocationConfig] = None


# command
class CommandRequest(BaseModel):
    command: str          # START | STOP | ONCE
    user_id: int
    tickers: list[str] = []
    callback_url: Optional[str] = None
    warmup: list[dict] = []


# ai_webhook 
class PredictionResult(BaseModel):
    ticker: str
    trade_datetime: str
    pred_label: int
    pred_str: str
    prob_buy: float
    prob_hold: float
    prob_sell: float
    # AI 서빙이 버전을 보내지 않아도 수신되도록 기본값을 둔다 (필수였을 때는 /ai/realtime 이 422 로 거절됨)
    model_version: str = "unknown"
    interpretability: Optional[dict] = None


class RealtimePayload(BaseModel):
    inference_at: str
    results: list[PredictionResult]
    # AI 서버(serving/inference_pipeline.py)가 매번 보내는데 스키마에 없어서 조용히 버려지고 있었다
    # -- push를 job 단위로 추적할 수 없었다(2026-10-02 통합 감사). 당장 로직에서 쓰진 않지만, 받은
    # 값을 버리지 않도록 받아만 둔다.
    job_id: str = ""
    user_id: str = ""


class OnceCallbackPayload(BaseModel):
    job_id: str
    user_id: str
    inference_at: str
    results: list[PredictionResult]

class AgreementAnalysisRequest(BaseModel):
    ticker: str
    name: Optional[str] = None
    recommendation_signal: str
    recommendation_summary: Optional[str] = None
    recommendation_reasons: Optional[str] = None
    tft_signal: str
    confidence: float
    prob_buy: float
    prob_hold: float
    prob_sell: float


class AgreementAnalysisResponse(BaseModel):
    status: str
    recommendation_signal: str
    tft_signal: str
    alignment_label: str
    alignment_level: str
    summary: str
    interpretation: str
    action_note: str


class WarmupPayload(BaseModel):
    ticker: str
    candles: list[dict]


# market
class OHLCVRecord(BaseModel):
    ticker: str
    trade_date: str | None = None # 일봉
    trade_datetime: str | None = None #분봉
    open: float
    high: float
    low: float
    close: float
    volume: int


class OHLCVPayload(BaseModel):
    timeframe: Literal["1d", "5m", "1m"]  # 1d | 5m | 1m
    records: list[OHLCVRecord]


