"""FEATURE_DESC는 AI 모델의 실제 33개 피처(training/champion_config.json, AI_Gentle_Viking_RE 레포
README.md에 고정)와 1:1로 맞아야 한다. 예전엔 bb_position/vol_ratio/macd_ratio/per/pbr/prop_* 등
지금 모델에 없는 8개가 매핑돼 있었고, disparity_5d/20d/60d·sector_*·lev_* 등 실제 피처 대부분은
매핑이 없어 원본 컬럼명이 그대로 노출됐다(2026-10-02 통합 감사).

이 테스트는 그 컬럼 목록을 여기에 고정해둔다 -- AI 쪽 champion_config.json이 바뀌면(피처 추가/제거)
이 목록도 같이 리뷰해서 갱신해야 한다."""
from app.services_report import FEATURE_DESC

AI_CHAMPION_COLUMNS = [
    "log_ret", "disparity_5d", "disparity_20d", "disparity_60d", "rsi_14", "volatility_20d",
    "sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_ma_ratio_20d", "sector_volatility",
    "sector_volume_ratio", "is_dividend", "is_bonus_issue", "is_rights_offering", "is_split",
    "day_of_week", "lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow", "is_vi_triggered",
    "vi_count_recent5d", "kospi_ret", "kosdaq_ret", "snp500_ret", "nasdaq_ret", "phlx_semi_ret",
    "vix_chg", "usd_krw_chg", "us_10y_yield_chg", "rate_spread_us_kr", "wti_ret", "gold_ret",
]


def test_feature_desc_covers_exactly_the_ai_models_real_columns():
    assert set(FEATURE_DESC) == set(AI_CHAMPION_COLUMNS)
    assert len(AI_CHAMPION_COLUMNS) == 33


def test_feature_desc_interpreters_run_without_error_on_plausible_values():
    sample_values = {
        "log_ret": 0.01, "disparity_5d": 1.0, "disparity_20d": 1.0, "disparity_60d": 1.0,
        "rsi_14": 50.0, "volatility_20d": 0.015, "sector_ret_1d": 0.0, "sector_ret_5d": 0.0,
        "sector_ret_20d": 0.0, "sector_ma_ratio_20d": 1.0, "sector_volatility": 0.015,
        "sector_volume_ratio": 1.0, "is_dividend": 0, "is_bonus_issue": 0, "is_rights_offering": 0,
        "is_split": 0, "day_of_week": 2, "lev_total_aum": 1_000_000_000.0, "lev_aum_to_mktcap": 0.01,
        "est_rebalancing_flow": 0.0, "is_vi_triggered": 0, "vi_count_recent5d": 0, "kospi_ret": 0.0,
        "kosdaq_ret": 0.0, "snp500_ret": 0.0, "nasdaq_ret": 0.0, "phlx_semi_ret": 0.0, "vix_chg": 0.0,
        "usd_krw_chg": 0.0, "us_10y_yield_chg": 0.0, "rate_spread_us_kr": 0.0, "wti_ret": 0.0,
        "gold_ret": 0.0,
    }
    assert set(sample_values) == set(AI_CHAMPION_COLUMNS)
    for name, value in sample_values.items():
        desc_name, interpreter = FEATURE_DESC[name]
        assert isinstance(desc_name, str) and desc_name
        assert isinstance(interpreter(value), str)
