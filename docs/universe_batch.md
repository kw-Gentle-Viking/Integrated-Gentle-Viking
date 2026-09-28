# 200종목 배치 추론 (serving/universe_batch.py)

`ticker_universe` 200종목을 feature_pool의 **마지막 완성 거래일(as-of)** 기준으로 한 번에 추론하고, 같은
as-of 날짜 안에서 횡단면 순위 백분위를 계산해 JSON으로 저장한다. 모델 무관(`serving.model.get_model()`,
`TFT_MODEL_PATH`/`CHAMPION_CONFIG_PATH` env로 교체).

## 실행

```bash
cd /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 STOCK_DB_V2_DSN=... \
  /home/user/miniconda3/envs/dl_env/bin/python -m serving.universe_batch [--asof YYYY-MM-DD] [--out path]
```

- `--asof` 생략: 유니버스의 50% 이상이 행을 가진 가장 최근 feature_pool 날짜(일부만 적재된 최신일 방지).
- `--out` 생략: `training/artifacts/universe_scores_<asof>.json`.
- DB는 SELECT만(`stock_db_v2`). 운영 DB/KIS 호출 없음. CPU 전용.

실측(CPU 6코어, OMP 2스레드): as-of 2026-09-08 200종목 198 산출 / 2 제외 / 0 실패, 모델 추론 포함 4.1초
(프로세스 전체 ~5.7초). as-of 2024-09-19(추석 연휴 직후)도 4초.

## 계산

종목별로 as-of 이전 59행 + as-of 행(완성 일봉을 5분봉 1개로 감싼 "오늘" 행, 15:30 시각)을
`build_encoder_df` -> `run_inference`에 그대로 태운다(라이브 서빙과 같은 경로, time_progress=1.0 포함).
`score = p_buy - p_sell`. `rank_pct`는 score 오름차순 평균 순위 / (유효 종목 수 - 1): 최저 0.0, 최고 1.0,
높을수록 매수 쪽. 동점은 평균 순위 공유(exact 비교), NaN/inf/None 점수는 제외(rank_pct=null), 유효 점수가
2개 미만이면 전원 null. 순위 계산은 순수 함수 `compute_rank_pct`.

## 출력 JSON

`asof, generated_at, model_path, model_version, champion_config_version, git_commit, score_definition,
n_universe/n_scored/n_excluded/n_failed, elapsed_sec` + 아래 3개 목록.

- `scores`: `{ticker, p_buy, p_hold, p_sell, score, rank_pct}` (매수 쪽이 먼저)
- `excluded`(데이터 부족, 명시적 제외): `no_row_on_asof`(as-of 행 없음/상장폐지·거래정지), `insufficient_history`
  (행 <60), `history_gap`(60행 창 안에 시장 거래일이 빠짐 -- 시장 거래일은 유니버스 50% 이상이 행을 가진 날짜)
- `failed`(예외): `error`(예외 종류+메시지, 로그에도 남김), `non_finite_output`

한 종목의 실패/제외는 나머지에 영향이 없다.

## 크론 (등록하지 않음, 권고만)

일 1회, 평일, 그날 feature_pool 적재가 끝난 뒤(저녁, 예: 19:00 KST). 예시 한 줄(설치 금지, 승인 후에만):

```
0 19 * * 1-5 cd /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /home/user/miniconda3/envs/dl_env/bin/python -m serving.universe_batch >> serving/universe_batch.log 2>&1
```

휴장일에 돌면 마지막 완성 거래일이 as-of로 잡혀 같은 결과 파일이 다시 써진다(멱등). 새 거래일 데이터가 아직
적재 전이면 전 거래일이 as-of가 되므로, 적재 완료 이후 시각에 돌려야 한다.

## 한계

- as-of 행은 완성 일봉이므로 라이브 장중 추론과 달리 FFILL 열(섹터/매크로 등)도 실제 당일 값이 아니라
  전일 이월값이 들어간다(서빙과 동일 정책, `docs/serving_parity.md`).
- 점수 동점은 부동소수 exact 비교라 실질적으로 거의 발생하지 않는다.
