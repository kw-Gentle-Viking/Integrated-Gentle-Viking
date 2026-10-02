import math
from typing import Optional
import numpy as np
import psycopg2


def compute_next_day_return(close_prices: list[float]) -> list[Optional[float]]:
    returns: list[Optional[float]] = []
    for i in range(len(close_prices)):
        if i + 1 >= len(close_prices) or close_prices[i + 1] is None or close_prices[i] is None:
            returns.append(None)
        else:
            returns.append((close_prices[i + 1] - close_prices[i]) / close_prices[i])
    return returns


def derive_threshold(returns: list[float], target_hold_ratio: float = 0.5) -> float:
    """|return| 분포에서, 절댓값 기준으로 target_hold_ratio 분위수를 threshold로 삼는다.
    즉 관망 비율이 정확히 target_hold_ratio가 되는 |r| 값을 반환."""
    abs_returns = np.abs(np.array([r for r in returns if r is not None]))
    return float(np.quantile(abs_returns, target_hold_ratio))


def assign_label(returns: list[Optional[float]], threshold: float) -> list[Optional[int]]:
    labels: list[Optional[int]] = []
    for r in returns:
        if r is None:
            labels.append(None)
        elif r >= threshold:
            labels.append(0)  # 매수
        elif r <= -threshold:
            labels.append(2)  # 매도
        else:
            labels.append(1)  # 관망
    return labels


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def derive_volnorm_params(returns: list, vols: list, target_hold_ratio: float = 0.5,
                          floor_quantile: float = 0.01) -> dict:
    """변동성 정규화 라벨 파라미터 산출 (호출자는 train 구간 데이터만 넘길 것).

    floor = vols의 floor_quantile 분위수 (변동성 ~0인 날의 z 폭발 방지),
    k     = |ret / max(vol, floor)| 의 target_hold_ratio 분위수.
    ret 또는 vol이 None/NaN인 쌍은 제외한다."""
    pairs = [(float(r), float(v)) for r, v in zip(returns, vols)
             if not _is_missing(r) and not _is_missing(v)]
    if not pairs:
        raise ValueError("no complete (return, vol) pairs to derive volnorm params from")
    r = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    floor = float(np.quantile(v, floor_quantile))
    k = float(np.quantile(np.abs(r) / np.maximum(v, floor), target_hold_ratio))
    return {"k": k, "floor": floor}


def assign_label_volnorm(returns: list, vols: list, k: float, floor: float) -> list[Optional[int]]:
    """z = ret / max(vol, floor); z>=k 매수(0), z<=-k 매도(2), 그 외 관망(1).
    ret 또는 vol이 None/NaN이면 None."""
    labels: list[Optional[int]] = []
    for r, v in zip(returns, vols):
        if _is_missing(r) or _is_missing(v):
            labels.append(None)
            continue
        z = r / max(v, floor)
        if z >= k:
            labels.append(0)
        elif z <= -k:
            labels.append(2)
        else:
            labels.append(1)
    return labels


def build_label_rows(ticker: str, trade_dates: list[str], close_prices: list[float],
                      threshold: float) -> list[dict]:
    returns = compute_next_day_return(close_prices)
    labels = assign_label(returns, threshold)
    return [
        {"ticker": ticker, "trade_date": d, "next_day_return": r, "label": l}
        for d, r, l in zip(trade_dates, returns, labels)
    ]


def upsert_labels(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO labels (ticker, trade_date, next_day_return, label)
                   VALUES (%(ticker)s, %(trade_date)s, %(next_day_return)s, %(label)s)
                   ON CONFLICT (ticker, trade_date) DO UPDATE SET
                     next_day_return = EXCLUDED.next_day_return, label = EXCLUDED.label""",
                r,
            )
    conn.commit()
    conn.close()
