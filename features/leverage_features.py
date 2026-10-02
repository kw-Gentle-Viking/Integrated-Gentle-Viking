def compute_rebalancing_flow(prev_aum: float, underlying_return: float, multiple: float) -> float:
    """설계 §6.3: 리밸런싱 금액 = 전일 AUM × 기초자산 당일 수익률 × (배율 - 1)."""
    return prev_aum * underlying_return * (multiple - 1)


def aggregate_leverage_signals(product_rows: list[dict], underlying_return: float,
                                 underlying_market_cap: float) -> dict:
    if not product_rows:
        return {"lev_total_volume": 0.0, "lev_total_aum": 0.0,
                "lev_aum_to_mktcap": 0.0, "est_rebalancing_flow": 0.0}
    total_volume = sum(r["close_price"] * r["volume"] for r in product_rows)
    total_aum = sum(r["aum"] for r in product_rows)
    total_flow = sum(
        compute_rebalancing_flow(r["aum"], underlying_return, r["multiple"])
        for r in product_rows
    )
    return {
        "lev_total_volume": total_volume, "lev_total_aum": total_aum,
        "lev_aum_to_mktcap": total_aum / underlying_market_cap if underlying_market_cap else 0.0,
        "est_rebalancing_flow": total_flow,
    }
