# app/routes_trade.py
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.ai_universe import is_ai_covered_ticker
from app.db import get_db,SessionLocal
from app.dependencies import get_current_user
from app.models import User,TradeLog,Basket,LiveCandle,ManualTradeLock,AutoTradeDecision
from app.ai_client import AIClient
from app.services_allocation import allocate_portfolio
from app.routes_ai_command import command_queue
from datetime import datetime

from pydantic import BaseModel
from app.schemas import AllocationConfig,TickerStrategy
from app.strategy_factory import create_strategy
from backtest.engine.risk import Portfolio
from typing import Optional

from app.strategy_factory import create_strategy, get_warmup_count, get_timeframe, resolve_strategy_id
from app.trade_helpers import no_auto_basket_message, trade_log_to_dict

import pandas as pd
import os
from app.kis_websocket import KISWebSocket
from app.kis_config import kis_is_mock, kis_real_trading_enabled
from app.demo import demo_autotrade_loop_enabled, demo_mode_enabled

from app.shared_state import ai_signal_event, warmup_events, warmup_received, warmup_requirements


router = APIRouter()

# 유저별 자동매매 상태 저장 (메모리)
active_tasks: dict[int, asyncio.Task] = {}
active_demo_trades: set[int] = set()
ai_client = AIClient()



import mojito

def get_broker():
    """KIS 브로커 인스턴스 생성"""
    try:
        return mojito.KoreaInvestment(
            api_key=os.getenv("KIS_APP_KEY"),
            api_secret=os.getenv("KIS_APP_SECRET"),
            acc_no=os.getenv("KIS_ACC_NO"),
            mock=kis_is_mock(),
        )
    except Exception as e:
        print(f" KIS 연결 실패: {e}")
        return None

broker = get_broker()

def get_balance() -> int:
    """KIS API 예수금 조회"""
    if not broker:
        return 10_000_000  # KIS 미연결 시 로컬 시연 기본값

    try:
        resp = broker.fetch_balance()
        if resp and "output2" in resp and len(resp["output2"]) > 0:
            data = resp["output2"][0]
            return int(data.get("nrciv_blce", data.get("dnca_tot_amt", 0)))
        return 10_000_000
    except Exception as e:
        print(f"잔고 조회 실패: {e}")
        return 10_000_000


def get_auto_trade_basket_items(db: Session, user_id: int) -> tuple[list[Basket], list[str], list[str]]:
    items = db.query(Basket).filter(Basket.user_id == user_id).all()
    manual_tickers = {
        lock.ticker
        for lock in db.query(ManualTradeLock).filter(ManualTradeLock.user_id == user_id).all()
    }
    # AI가 커버하지 않는 종목(코스닥 등)은 ai_client.predict()가 confidence-0 HOLD나(설정에 따라) 무작위
    # 신호로 치환해버리므로, 자동매매 대상에서 먼저 걸러낸다(2026-10-02 통합 감사에서 발견).
    excluded_unsupported = [item.ticker for item in items if not is_ai_covered_ticker(item.ticker)]
    remaining = [item for item in items if item.ticker not in set(excluded_unsupported)]
    auto_items = [item for item in remaining if item.ticker not in manual_tickers]
    excluded_manual = [item.ticker for item in remaining if item.ticker in manual_tickers]
    return auto_items, excluded_manual, excluded_unsupported


def sync_request_basket(db: Session, user_id: int, payload) -> None:
    if payload.basket is None and payload.tickers is None:
        return

    requested: dict[str, str] = {}
    if payload.basket is not None:
        for item in payload.basket:
            ticker = item.ticker.strip()
            if ticker:
                requested[ticker] = item.ticker_name.strip() or ticker
    else:
        for ticker in payload.tickers or []:
            code = ticker.strip()
            if code:
                requested[code] = code

    existing = {
        item.ticker: item
        for item in db.query(Basket).filter(Basket.user_id == user_id).all()
    }

    for ticker, item in existing.items():
        if ticker not in requested:
            db.delete(item)

    for ticker, ticker_name in requested.items():
        item = existing.get(ticker)
        if item:
            item.ticker_name = ticker_name
        else:
            db.add(Basket(user_id=user_id, ticker=ticker, ticker_name=ticker_name))

    db.commit()


def add_auto_decision(
    db: Session,
    user_id: int,
    ticker: str,
    action: str,
    reason: str,
    detail: str = "",
    ai_signal: str = "",
    ai_confidence: float = 0.0,
    strategy_id: str = "",
    price: float = 0.0,
) -> None:
    db.add(
        AutoTradeDecision(
            user_id=user_id,
            ticker=ticker,
            action=action,
            reason=reason,
            detail=detail,
            ai_signal=ai_signal,
            ai_confidence=float(ai_confidence or 0.0),
            strategy_id=strategy_id,
            price=float(price or 0.0),
        )
    )


async def get_market_close(ticker: str) -> int:
    from app.routes_kis import get_current_price

    data = await get_current_price(ticker)
    output = data.get("output") or {}
    close = int(output.get("stck_prpr") or 0)
    if close <= 0:
        raise HTTPException(status_code=502, detail=f"{ticker} 현재가 조회 실패")
    return close

async def trading_loop(user_id: int, tickers: list[str], persona_id: int,
    total_capital: int,
    config: AllocationConfig,
    ticker_strategies: list[TickerStrategy],):
    """유저별 자동매매 루프 (5분 주기)"""
    # strategies = {
    #     t: create_strategy(t, strategy_config.strategy_id, strategy_config.params)
    #     for t in tickers
    # }
    strategies = {}

    # 종목별 다른 전략 생성
    for ts in ticker_strategies:
        strategies[ts.ticker] = create_strategy(
            ts.ticker, ts.strategy_id, ts.params
        )

    # 바구니에 있는데 전략 미지정 종목은 기본 전략
    for t in tickers: 
        if t not in strategies:
            strategies[t] = create_strategy(t,"ultra_safe",None)
    portfolio = Portfolio()
    portfolio.cash = total_capital
    portfolio.equity = total_capital

    ws = KISWebSocket(
        app_key=os.getenv("KIS_APP_KEY"),
        app_secret=os.getenv("KIS_APP_SECRET"),
    )

    ws_task = asyncio.create_task(ws.connect(tickers))

    try:

        await asyncio.sleep(3) # 구독 완료 대기 


        # ── 1. 워밍업 대기 ──────────────────────────────────────
        required_warmup = warmup_requirements.get(user_id) or {
            ticker: get_warmup_count(
                next((ts.strategy_id for ts in ticker_strategies if ts.ticker == ticker), "ultra_safe")
            )
            for ticker in tickers
        }
        print(f"[User {user_id}] 워밍업 데이터 대기 중... required={required_warmup}")

        if user_id not in warmup_events:
            warmup_events[user_id] = asyncio.Event()
            warmup_requirements[user_id] = required_warmup
            warmup_received[user_id] = {}

            warmup_db = SessionLocal()
            try:
                warmup_db.query(LiveCandle).filter(LiveCandle.ticker.in_(tickers)).delete(synchronize_session=False)
                warmup_db.commit()
            finally:
                warmup_db.close()

        try:
            await asyncio.wait_for(warmup_events[user_id].wait(), timeout=90)
            print(f"[User {user_id}] 모든 워밍업 데이터 수신 완료: {warmup_received.get(user_id, {})}")
        except asyncio.TimeoutError:
            missing = [
                ticker
                for ticker, count in required_warmup.items()
                if warmup_received.get(user_id, {}).get(ticker, 0) < count
            ]
            print(f"[User {user_id}] 워밍업 타임아웃 | received={warmup_received.get(user_id, {})} | missing={missing}")

        # 1. 초기 데이터 로드 (과거 봉)
        # for ticker in tickers:
        #     # TODO: market-db에서 최근 N개 봉 로드
        #     # rows = db.query(PriceMin05).filter(...).order_by(asc).limit(50)
        #     # for row in rows:
        #     #     strategies[ticker].generate_orders(row, portfolio)  # 워밍업
        #     print(f"{ticker}: 과거 데이터 워밍업 완료")

        db = SessionLocal()
        try:
            for ticker in tickers:
                candles = db.query(LiveCandle)\
                    .filter(LiveCandle.ticker == ticker)\
                    .order_by(LiveCandle.trade_datetime.desc())\
                    .limit(get_warmup_count(
                        next((ts.strategy_id for ts in ticker_strategies if ts.ticker == ticker), "ultra_safe")
                    )).all()
                candles = list(reversed(candles))

                for candle in candles:
                    row = pd.Series({
                        "close": candle.close,
                        "high": candle.high,
                        "low": candle.low,
                        "volume": candle.volume,
                    }, name=pd.Timestamp(candle.trade_datetime))
                    strategies[ticker].generate_orders(row, portfolio)

                print(f" {ticker}: {len(candles)}개 봉 워밍업 완료")
        finally:
            db.close()


        # 2. 실시간 루프
        while True:
            try:
                await asyncio.wait_for(ai_signal_event.wait(), timeout=600)
                ai_signal_event.clear()  # 다음 push 대기 위해 리셋
            except asyncio.TimeoutError:
                print(f"  AI push 10분 초과 → 스킵")
                continue
            db = SessionLocal()
            try :
                print(f"[User {user_id}] 자동매매 실행...")

                predictions = [] 
                for ticker in tickers:
                    pred = ai_client.predict(ticker)
                    predictions.append(pred)
                
                allocation = allocate_portfolio(
                predictions=predictions,
                persona_id=persona_id,
                total_capital=total_capital,
                max_weight=config.max_weight,
                cash_reserve=config.cash_reserve,
                min_confidence=config.min_confidence,
                use_persona_boost=config.use_persona_boost,
                )
                allocation_map = {a["ticker"]: a for a in allocation}


                for ticker in tickers:

                    market = ws.get_price(ticker)
                    if not market:
                        print(f" {ticker} : 시세없음 -> 스킵")
                        add_auto_decision(
                            db, user_id, ticker, "SKIP", "NO_MARKET_PRICE",
                            "실시간 시세를 받지 못해 자동매매 판단을 건너뜀",
                            strategy_id=strategies[ticker].__class__.__name__,
                        )
                        continue

                    close = market["price"]
                    high = market["high"]
                    low = market["low"]
                    volume = market["volume"]

                    row = pd.Series(
                        {"close": close, "high": high, "low": low, "volume": volume},
                        name=pd.Timestamp.now(),
                    )

                    # AI 추론
                    pred_map = {p["ticker"] : p for p in predictions}
                    signal = pred_map[ticker]["signal"]
                    confidence = pred_map[ticker]["confidence"]
                    print(f"  {ticker}: AI {signal} | confidence={confidence:.3f}")

                    if confidence < config.min_confidence:
                        detail = f"AI 확신도 {confidence:.3f}가 최소 기준 {config.min_confidence:.3f}보다 낮음"
                        print(f"  {ticker}: 확신도 부족 ({confidence:.3f} < {config.min_confidence:.3f}) -> SKIP")
                        add_auto_decision(
                            db, user_id, ticker, "SKIP", "LOW_CONFIDENCE", detail,
                            ai_signal=signal, ai_confidence=confidence,
                            strategy_id=strategies[ticker].__class__.__name__, price=close,
                        )
                        continue
                    # if signal == "HOLD":
                    #     print(f"  {ticker}: AI 관망 부족 -> HOLD")
                    #     continue

                    # 전략 필터 (봉 데이터 자동 누적됨)
                    orders = strategies[ticker].generate_orders(row, portfolio)
                    if not orders:
                        print(f"  {ticker}: 전략 조건 미충족 -> HOLD")
                        add_auto_decision(
                            db, user_id, ticker, "HOLD", "STRATEGY_CONDITION_NOT_MET",
                            "AI 확신도는 기준을 통과했지만 전략이 BUY/SELL 주문 신호를 만들지 않음",
                            ai_signal=signal, ai_confidence=confidence,
                            strategy_id=strategies[ticker].__class__.__name__, price=close,
                        )
                        continue

                    # AI + 전략 일치 시 실행
                    for order in orders:
                        order_signal = "BUY" if order.side.value == "BUY" else "SELL"
                        
                       # 매수: allocation 비중 기반 수량
                        if order_signal == "BUY" and ticker in allocation_map:
                            a = allocation_map[ticker]

                            if signal == order_signal:
                                qty = a["amount"] // close
                            elif signal == "HOLD":
                                qty = (a["amount"] // close) // 2 
                            else:
                                detail = f"AI 신호 {signal}와 전략 신호 {order_signal}가 일치하지 않음"
                                print(f"  {ticker}: AI({signal})와 전략({order_signal}) 불일치 -> HOLD")
                                add_auto_decision(
                                    db, user_id, ticker, "HOLD", "AI_STRATEGY_MISMATCH", detail,
                                    ai_signal=signal, ai_confidence=confidence,
                                    strategy_id=strategies[ticker].__class__.__name__, price=close,
                                )
                                continue

                        # 매도: 보유 수량 기반
                        elif order_signal == "SELL":
                            pos = portfolio.positions.get(ticker)
                            if not pos or pos.qty <= 0:
                                print(f"  {ticker}: 전략 SELL 신호지만 보유 수량 없음 -> HOLD")
                                add_auto_decision(
                                    db, user_id, ticker, "HOLD", "NO_POSITION_TO_SELL",
                                    "전략은 SELL 신호를 만들었지만 포트폴리오에 보유 수량이 없음",
                                    ai_signal=signal, ai_confidence=confidence,
                                    strategy_id=strategies[ticker].__class__.__name__, price=close,
                                )
                                continue

                            if signal == order_signal:
                                qty = pos.qty
                            elif signal == "HOLD":
                                qty = pos.qty // 2
                            else:
                                detail = f"AI 신호 {signal}와 전략 신호 {order_signal}가 일치하지 않음"
                                print(f"  {ticker}: AI({signal})와 전략({order_signal}) 불일치 -> HOLD")
                                add_auto_decision(
                                    db, user_id, ticker, "HOLD", "AI_STRATEGY_MISMATCH", detail,
                                    ai_signal=signal, ai_confidence=confidence,
                                    strategy_id=strategies[ticker].__class__.__name__, price=close,
                                )
                                continue

                        else:
                            print(f"  {ticker}: 전략 {order_signal} 신호지만 매수 배분 없음 -> HOLD")
                            add_auto_decision(
                                db, user_id, ticker, "HOLD", "NO_BUY_ALLOCATION",
                                "전략이 BUY 신호를 만들었지만 AI 매수 배분 대상에 포함되지 않음",
                                ai_signal=signal, ai_confidence=confidence,
                                strategy_id=strategies[ticker].__class__.__name__, price=close,
                            )
                            continue

                        if qty <= 0:
                            print(f"  {ticker}: 주문 가능 수량 0 -> HOLD")
                            add_auto_decision(
                                db, user_id, ticker, "HOLD", "ZERO_ORDER_QUANTITY",
                                "배정 금액과 현재가 기준 주문 가능 수량이 0주",
                                ai_signal=signal, ai_confidence=confidence,
                                strategy_id=strategies[ticker].__class__.__name__, price=close,
                            )
                            continue
                            
                        # 주문 실행 + 재시도 
                        MAX_RETRY = 3
                        RETRY_DELAY = 10
                        order_status = "FAILED"

                        for attempt in range(MAX_RETRY):
                            try : 
                                if not broker:
                                    raise Exception("KIS broker is not configured")
                                if not kis_is_mock() and not kis_real_trading_enabled():
                                    raise Exception("Set KIS_REAL_TRADING_ENABLED=true to allow real-account orders")

                                resp = broker.create_order(
                                    symbol=ticker,
                                    side=order_signal,
                                    qty=qty,
                                    order_type="market",
                                )
                                if resp.get("rt_cd") != "0":
                                    raise Exception(resp.get("msg1"))

                                order_status = "FILLED"
                                print(f"  {ticker}: {order_signal} 체결 | qty={qty} | {qty*close:,}원")
                                break
                            except Exception as e : 
                                print(f"   {ticker}: 주문 실패 ({attempt+1}/{MAX_RETRY}) - {e}")
                                if attempt < MAX_RETRY - 1:
                                    await asyncio.sleep(RETRY_DELAY)
                                else:
                                    print(f"  {ticker}: 최종 실패")

                        db.add(TradeLog(
                                user_id=user_id,
                                ticker=ticker,
                                side=order_signal,
                                qty=int(qty),
                                price=close,
                                amount=int(qty * close),
                                ai_signal=signal,
                                ai_confidence=confidence,
                                strategy_id=strategies[ticker].__class__.__name__,
                                status = order_status, 
                        ))
                        add_auto_decision(
                            db, user_id, ticker, "ORDER_SUBMITTED", order_status,
                            f"{order_signal} 주문 시도 qty={int(qty)} amount={int(qty * close)}",
                            ai_signal=signal, ai_confidence=confidence,
                            strategy_id=strategies[ticker].__class__.__name__, price=close,
                        )
                        print(f"{ticker}: {order_signal} | qty = {qty} | {qty*close: ,}원")
                            # TODO: KIS API 주문
                        
                db.commit()
            finally :
                db.close()
            #await asyncio.sleep(300)  # 5분 대기
    except asyncio.CancelledError:
        await ws.disconnect()
        ws_task.cancel()
        warmup_events.pop(user_id, None)
        warmup_requirements.pop(user_id, None)
        warmup_received.pop(user_id, None)
        print(f" [User {user_id}] 자동매매 중단됨")


async def run_once(
    user_id: int,
    tickers: list[str],
    persona_id: int,
    total_capital: int,
    config: AllocationConfig,
    ticker_strategies: list[TickerStrategy],
):
    """1회 실행"""
    strategies = {}
    for ts in ticker_strategies:
        strategies[ts.ticker] = create_strategy(ts.ticker, ts.strategy_id, ts.params)
    for t in tickers:
        if t not in strategies:
            strategies[t] = create_strategy(t, "ultra_safe", None)

    portfolio = Portfolio()
    portfolio.cash = total_capital
    portfolio.equity = total_capital

    # AI 추론
    predictions = []
    for ticker in tickers:
        pred = ai_client.predict(ticker)
        predictions.append(pred)

    # 포트폴리오 분배
    allocation = allocate_portfolio(
        predictions=predictions,
        persona_id=persona_id,
        total_capital=total_capital,
        max_weight=config.max_weight,
        cash_reserve=config.cash_reserve,
        min_confidence=config.min_confidence,
        use_persona_boost=config.use_persona_boost,
    )
    allocation_map = {a["ticker"]: a for a in allocation}

    pred_map = {p["ticker"]: p for p in predictions}

    db = SessionLocal()
    results = []
    try:
        for ticker in tickers:
            close = await get_market_close(ticker)

            row = pd.Series(
                {"close": close, "high": close, "low": close, "volume": 0},
                name=pd.Timestamp.now(),
            )

            signal = pred_map[ticker]["signal"]
            confidence = pred_map[ticker]["confidence"]

            action = "SKIP"

            if confidence < config.min_confidence:
                action = "SKIP"
            elif signal == "HOLD":
                orders = strategies[ticker].generate_orders(row, portfolio)
                for order in orders:
                    order_signal = "BUY" if order.side.value == "BUY" else "SELL"

                    if order_signal == "BUY" and ticker in allocation_map:
                        qty = (allocation_map[ticker]["amount"] // close) // 2
                    elif order_signal == "SELL":
                        pos = portfolio.positions.get(ticker)
                        qty = pos.qty // 2 if pos and pos.qty > 0 else 0
                    else:
                        continue

                    if qty > 0:
                        action = f"HOLD_{order_signal}"
                        db.add(TradeLog(
                            user_id=user_id,
                            ticker=ticker,
                            side=order_signal,
                            qty=qty,
                            price=close,
                            amount=int(qty * close),
                            ai_signal=signal,
                            ai_confidence=confidence,
                            strategy_id=strategies[ticker].__class__.__name__,
                            status="FILLED",
                        ))
            else:
                orders = strategies[ticker].generate_orders(row, portfolio)
                for order in orders:
                    order_signal = "BUY" if order.side.value == "BUY" else "SELL"

                    if signal == order_signal:
                        if order_signal == "BUY" and ticker in allocation_map:
                            qty = allocation_map[ticker]["amount"] // close
                        elif order_signal == "SELL":
                            pos = portfolio.positions.get(ticker)
                            qty = pos.qty if pos and pos.qty > 0 else 0
                        else:
                            continue

                        if qty > 0:
                            action = signal
                            db.add(TradeLog(
                                user_id=user_id,
                                ticker=ticker,
                                side=order_signal,
                                qty=qty,
                                price=close,
                                amount=int(qty * close),
                                ai_signal=signal,
                                ai_confidence=confidence,
                                strategy_id=strategies[ticker].__class__.__name__,
                                status="FILLED",
                            ))
                    else:
                        action = "HOLD"

            results.append({
                "ticker": ticker,
                "signal": signal,
                "confidence": confidence,
                "action": action,
                "price": close,
            })

        db.commit()
    finally:
        db.close()

    return results


# routes_trade.py에 추가
class TradeBasketItem(BaseModel):
    ticker: str
    ticker_name: str = ""


class TradeRequest(BaseModel):
    total_capital: Optional[int] = None  # None이면 KIS에서 자동 조회
    basket: Optional[list[TradeBasketItem]] = None
    tickers: Optional[list[str]] = None
    ticker_strategies: list[TickerStrategy] = []
    allocation: AllocationConfig = AllocationConfig()


@router.post("/start")
async def start_trading(
    payload: TradeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    
    user_id = current_user.id

    # 총 자본금: 유저가 설정했으면 그거, 아니면 KIS에서 조회
    if payload.total_capital:
        total_capital = payload.total_capital
    else:
        # TODO: KIS API로 예수금 조회
        total_capital = get_balance()

    if user_id in active_tasks and not active_tasks[user_id].done():
        return {"status": "ALREADY_RUNNING", "message": "이미 자동매매 실행 중"}

    sync_request_basket(db, user_id, payload)
    items, excluded_manual_tickers, excluded_unsupported_tickers = get_auto_trade_basket_items(db, user_id)
    if not items and (excluded_manual_tickers or excluded_unsupported_tickers):
        raise HTTPException(
            status_code=400,
            detail=no_auto_basket_message(excluded_manual_tickers, excluded_unsupported_tickers),
        )
    if not items:
        raise HTTPException(status_code=400, detail="바구니가 비어있습니다")

    tickers = [item.ticker for item in items]
    if excluded_manual_tickers:
        print(f"[User {user_id}] 직접매매 종목 자동매매 제외: {excluded_manual_tickers}")
    if excluded_unsupported_tickers:
        print(f"[User {user_id}] AI 미지원 종목 자동매매 제외: {excluded_unsupported_tickers}")
    print(f"[User {user_id}] 자동매매 대상 종목: {tickers}")

    warmup_requests = []
    resolved_strategies = {}
    for ticker in tickers:
        strategy_id = "ultra_safe"
        for ts in payload.ticker_strategies:
            if ts.ticker == ticker:
                strategy_id = ts.strategy_id
                break
        resolved_strategies[ticker] = resolve_strategy_id(strategy_id)
        if resolved_strategies[ticker]["fallback"]:
            print(f"[User {user_id}] 전략 '{strategy_id}' 미지원 -> conservative 로 대체 ({ticker})")
        warmup_requests.append({
            "ticker": ticker,
            "timeframe": get_timeframe(strategy_id),
            "count": get_warmup_count(strategy_id),
            "from_datetime": datetime.now().isoformat(),
        })

    warmup_requirements[user_id] = {item["ticker"]: item["count"] for item in warmup_requests}
    warmup_received[user_id] = {}
    warmup_events[user_id] = asyncio.Event()

    warmup_db = SessionLocal()
    try:
        warmup_db.query(LiveCandle).filter(LiveCandle.ticker.in_(tickers)).delete(synchronize_session=False)
        warmup_db.commit()
    finally:
        warmup_db.close()

    command_queue.append({
        "command": "START",
        "user_id": current_user.id,
        "tickers": tickers,
        "warmup" : warmup_requests,
        "created_at": datetime.now().isoformat(),
        "status": "pending",
    })



    if demo_mode_enabled() and not demo_autotrade_loop_enabled():
        active_demo_trades.add(user_id)
        return {
            "status": "RUNNING",
            "tickers": tickers,
            "excluded_manual_tickers": excluded_manual_tickers,
            "excluded_unsupported_tickers": excluded_unsupported_tickers,
            "strategies": resolved_strategies,
            "message": "데모 자동매매 시작: AI 서버에 START 커맨드를 전달했습니다.",
        }

    task = asyncio.create_task(
        trading_loop(
            user_id=user_id,
            tickers=tickers,
            persona_id=current_user.risk_score or 3,
            total_capital=total_capital,
            config=payload.allocation,
            ticker_strategies=payload.ticker_strategies,
        )
    )
    active_tasks[user_id] = task

    return {
        "status": "RUNNING",
        "tickers": tickers,
        "excluded_manual_tickers": excluded_manual_tickers,
        "excluded_unsupported_tickers": excluded_unsupported_tickers,
        "strategies": resolved_strategies,
        "message": "자동매매 시작 (5분 주기)",
    }
    

@router.post("/stop")
async def stop_trading(current_user: User = Depends(get_current_user)):
    user_id = current_user.id

    if user_id in active_tasks and not active_tasks[user_id].done():
        command_queue.append({
            "command": "STOP",
            "user_id": current_user.id,
            "tickers": [],
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        })
        active_tasks[user_id].cancel()
        del active_tasks[user_id]
        active_demo_trades.discard(user_id)
        return {"status": "STOPPED", "message": "자동매매 중단"}

    if user_id in active_demo_trades:
        active_demo_trades.discard(user_id)
        command_queue.append({
            "command": "STOP",
            "user_id": current_user.id,
            "tickers": [],
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        })
        return {"status": "STOPPED", "message": "데모 자동매매 중단: AI 서버에 STOP 커맨드를 전달했습니다."}

    command_queue.append({
        "command": "STOP",
        "user_id": current_user.id,
        "tickers": [],
        "created_at": datetime.now().isoformat(),
        "status": "pending",
    })

    return {"status": "NOT_RUNNING", "message": "실행 중인 자동매매 없음"}

@router.post("/once")
async def execute_once(
    payload: TradeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sync_request_basket(db, current_user.id, payload)
    items, excluded_manual_tickers, excluded_unsupported_tickers = get_auto_trade_basket_items(db, current_user.id)
    if not items and (excluded_manual_tickers or excluded_unsupported_tickers):
        raise HTTPException(
            status_code=400,
            detail=no_auto_basket_message(excluded_manual_tickers, excluded_unsupported_tickers),
        )
    if not items:
        raise HTTPException(status_code=400, detail="바구니가 비어있습니다")

    tickers = [item.ticker for item in items]
    job_id = f"trade-once-{current_user.id}-{int(datetime.now().timestamp())}"
    base_url = str(request.base_url).rstrip("/")

    command_queue.append({
        "command": "ONCE",
        "job_id": job_id,
        "user_id": current_user.id,
        "tickers": tickers,
        "excluded_manual_tickers": excluded_manual_tickers,
        "callback_url": f"{base_url}/ai/callback",
        "created_at": datetime.now().isoformat(),
        "status": "pending",
        "trade_request": payload.dict(),
    })

    return {
        "status": "QUEUED",
        "job_id": job_id,
        "tickers": tickers,
        "excluded_manual_tickers": excluded_manual_tickers,
        "excluded_unsupported_tickers": excluded_unsupported_tickers,
        "callback_url": f"{base_url}/ai/callback",
        "message": "AI 서버에 1회 분석/실행 커맨드를 전달했습니다. AI 서버가 결과를 콜백하면 리포트와 자동매매 판단에 반영됩니다.",
    }


@router.get("/status")
async def get_status(current_user: User = Depends(get_current_user)):
    user_id = current_user.id
    is_running = (
        user_id in active_demo_trades
        or (user_id in active_tasks and not active_tasks[user_id].done())
    )
    return {"status": "RUNNING" if is_running else "STOPPED"}

@router.get("/decisions")
def get_auto_trade_decisions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    decisions = db.query(AutoTradeDecision)\
        .filter(AutoTradeDecision.user_id == current_user.id)\
        .order_by(AutoTradeDecision.created_at.desc())\
        .limit(100).all()

    return [
        {
            "ticker": d.ticker,
            "action": d.action,
            "reason": d.reason,
            "detail": d.detail,
            "ai_signal": d.ai_signal,
            "ai_confidence": d.ai_confidence,
            "strategy_id": d.strategy_id,
            "price": d.price,
            "created_at": d.created_at.isoformat(),
        }
        for d in decisions
    ]


@router.get("/history")
def get_trade_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logs = db.query(TradeLog)\
        .filter(TradeLog.user_id == current_user.id)\
        .order_by(TradeLog.created_at.desc())\
        .limit(100).all()

    return [trade_log_to_dict(log) for log in logs]

