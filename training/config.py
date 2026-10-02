from omegaconf import OmegaConf, DictConfig

# 설계 §4, §5: 기존 55개 기반 + 레버리지 6개 등 feature_pool 전체 컬럼 중
# ablation(Task 15)에서 선택된 subset이 여기로 주입된다.
# 2026-09-08 라이브 백필 중 확인: KIS 무료 API는 PER/PBR/시총(daily_valuation)과
# 수급(investor_flow_daily) 둘 다 과거 시점 데이터를 지원하지 않음(전자는 API 자체가 "현재"만
# 반환, 후자는 최근 30영업일 롤링만 가능 — 히스토리 백필 불가). 사용자 결정: PER/PBR은 이번
# 사이클에서 포기, 수급은 보류(토스증권 API — 발급된 키 있음, 별도 클라이언트 코드 필요, 추후
# 검토). daily_valuation/investor_flow_daily 테이블·백필 코드(Task 5)는 남겨두되(추후 토스 연동
# 대비) feature_pool에는 조인하지 않고, 아래 목록에서도 제외한다.
#
# 2026-09-09 Task 13 재조정: 이 목록의 원래 초안(plan 문서 최초 작성 시점)은 아직 계산되지 않은
# 파생 피처(rel_close/rel_high/rel_low, disparity_5/20/60(무접미), vol_ratio, bb_position,
# macd_ratio/macd_signal_ratio/macd_hist_ratio, log_ret_1d, kospi_ret/kosdaq_ret/snp500_ret/
# nasdaq_ret/phlx_semi_ret, vix_chg/usd_krw_chg/us_10y_yield_chg/rate_spread_us_kr, wti_ret/
# gold_ret, is_merger, is_earnings, listing_days)를 가정하고 있었으나, 실제 라이브
# feature_pool(61개 컬럼, Task 10 fix round 2 이후 스키마)에는 이 이름들이 하나도 존재하지
# 않는다 — Task 11의 `features/run_fit_clip_scale.py` 상단 주석에서 이미 같은 간극을
# "build_features.py / Task 13이 처리할 gap"으로 명시적으로 플래그해 두었음. 이번 Task 13의
# 실제 작업 범위(config.py/train.py/test_config.py)에는 그 파생 피처 계산(원본 레벨 →
# 수익률/변화량 변환) 로직이 포함되어 있지 않았으므로, 당시엔 라이브 스키마에 실존하는
# 컬럼만 남기고 나머지는 드롭했다(23개). 매크로/지수 계열(코스피·코스닥·S&P500·나스닥·
# 필라델피아 반도체·VIX·환율·금리·유가·금)은 feature_pool에 원시 레벨(snp500_close, vix,
# usd_krw, us_10y_yield, fed_rate, kr_base_rate, wti_crude_oil, gold_price, index_0001,
# index_1001, nasdaq_close, phlx_semi_close)로만 존재하고 수익률/변화량 파생본이 전혀 없어
# 통째로 빠졌었다.
#
# 2026-09-09 Task 13 fix round 1 (Task 10.5 follow-up): Task 10.5(`features/add_macro_features.py`,
# 커밋 5eb4a26)가 위에서 빠졌던 매크로 파생 11개(kospi_ret, kosdaq_ret, snp500_ret, nasdaq_ret,
# phlx_semi_ret, vix_chg, usd_krw_chg, us_10y_yield_chg, rate_spread_us_kr, wti_ret, gold_ret)를
# feature_pool에 실제로 계산/백필(72개 컬럼, 376,682행)했으므로, 이제 라이브 스키마에 존재하는
# 컬럼으로서 목록에 추가한다(23 → 34개). 산식/단위는 `docs/data_units.md`의
# "feature_pool 매크로 파생 컬럼 11개 (Task 10.5, 2026-09-09)" 절 참고. 나머지 드롭된 파생
# 피처(rel_close 등, 위 문단)는 여전히 미계산 상태이므로 계속 빠져 있다 — 후속 태스크(Task 15
# ablation 또는 build_features.py 신설)에서 재검토할 것.
HISTORICAL_COLS_DEFAULT = [
    "log_ret", "disparity_5d", "disparity_20d", "disparity_60d",
    "rsi_14", "volatility_20d",
    "sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_ma_ratio_20d",
    "sector_volatility", "sector_volume_ratio",
    "is_dividend", "is_bonus_issue", "is_rights_offering", "is_split",
    "day_of_week",
    "lev_total_volume", "lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow",
    "is_vi_triggered", "vi_count_recent5d",
    "kospi_ret", "kosdaq_ret", "snp500_ret", "nasdaq_ret", "phlx_semi_ret",
    "vix_chg", "usd_krw_chg", "us_10y_yield_chg", "rate_spread_us_kr",
    "wti_ret", "gold_ret",
]
KNOWN_FUTURE_COLS = ["time_progress", "is_bok", "is_fomc", "is_witching_kr", "is_witching_us"]
STATIC_COLS = ["sector_id", "market_id"]


def build_tft_config(feature_columns: dict, num_classes: int = 3,
                      state_size: int = 32, attention_heads: int = 4,
                      lstm_layers: int = 1, dropout: float = 0.1) -> DictConfig:
    return OmegaConf.create({
        "task_type": "classification",
        "target_window_start": None,
        "data_props": {
            "num_historical_numeric": len(feature_columns["historical"]),
            "num_historical_categorical": 0, "historical_categorical_cardinalities": [],
            "num_static_numeric": 0, "num_static_categorical": len(feature_columns["static_cardinalities"]),
            "static_categorical_cardinalities": feature_columns["static_cardinalities"],
            "num_future_numeric": len(feature_columns["future"]),
            "num_future_categorical": 0, "future_categorical_cardinalities": [],
        },
        "model": {"state_size": state_size, "attention_heads": attention_heads,
                   "dropout": dropout, "lstm_layers": lstm_layers, "num_classes": num_classes},
    })
