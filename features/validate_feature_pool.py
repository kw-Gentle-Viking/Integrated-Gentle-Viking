import pandas as pd


def check_market_cap_consistency(df: pd.DataFrame, tolerance: float = 0.1) -> list[str]:
    """두 독립 소스(시총 스냅샷 계산 vs KIS 밸류에이션 API)로 구한 시총이 tolerance 이상 어긋나면
    단위 변환 버그일 가능성이 높다 — 티커 목록 반환."""
    flagged = []
    for _, row in df.iterrows():
        a, b = row["universe_market_cap"], row["valuation_market_cap"]
        if a == 0 or b == 0:
            flagged.append(row["ticker"])
            continue
        rel_diff = abs(a - b) / max(a, b)
        if rel_diff > tolerance:
            flagged.append(row["ticker"])
    return flagged


def check_value_ranges(df: pd.DataFrame) -> list[str]:
    """가격은 양수, 거래량은 0 이상이어야 함 — 위반 컬럼명 반환."""
    issues = []
    if "close_price" in df.columns and (df["close_price"] <= 0).any():
        issues.append("close_price")
    if "volume" in df.columns and (df["volume"] < 0).any():
        issues.append("volume")
    return issues


def check_null_rates(df: pd.DataFrame, max_null_ratio: float = 0.3) -> dict[str, float]:
    """컬럼별 결측 비율이 max_null_ratio를 넘으면 {컬럼명: 실제비율} 반환."""
    issues = {}
    for col in df.columns:
        ratio = df[col].isna().mean()
        if ratio > max_null_ratio:
            issues[col] = ratio
    return issues


def check_trading_day_gaps(df: pd.DataFrame, expected_min_days: int) -> list[str]:
    """종목별 행 수가 expected_min_days에 못 미치면 수집 누락 의심 — 티커 목록 반환."""
    counts = df.groupby("ticker").size()
    return counts[counts < expected_min_days].index.tolist()
