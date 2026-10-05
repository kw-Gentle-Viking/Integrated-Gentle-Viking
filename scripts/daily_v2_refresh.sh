#!/bin/bash
# AI용 stock_db_v2 일별 갱신: 운영 stock_db 복사 -> leverage 갱신 -> feature_pool 재빌드.
# 평일 16:45 (운영 수집 16:00 / build_batch_features 16:30 이후). 2026-10-06 전에는 아무것도 하지 않는다.
set -u
REPO=/home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
PY=/home/user/miniconda3/envs/dl_env/bin/python
START_DATE=2026-10-06

today=$(date +%F)
if [[ "$today" < "$START_DATE" ]]; then
  echo "[$(date '+%F %T')] $today < $START_DATE -> skip"
  exit 0
fi

cd "$REPO" || exit 1
set -a; . ./.env; set +a
echo "[$(date '+%F %T')] start daily v2 refresh"

$PY -m data_collection.sync_v2_from_prod || { echo "sync failed"; exit 1; }
$PY -m data_collection.run_leverage_backfill || echo "leverage backfill failed (continuing)"
from=$(date -d '-14 days' +%F)  # 저장 구간 시작. 룩백(120영업일)은 build_features가 영업일로 따로 읽는다
$PY -m features.build_features --start "$from" --end "$today" || { echo "build_features failed"; exit 1; }
# 매크로 파생 11개(kospi_ret 등)는 build_features가 쓰지 않아서 새 행이 NULL로 남는다 -> 매번 채운다
$PY -m features.add_macro_features || { echo "add_macro_features failed"; exit 1; }
# 레버리지 NAV·상장주식수·추정 AUM 일별 스냅샷 (과거 AUM은 KIS에 없어서 매일 쌓는다). 실패해도 갱신은 계속.
$PY -m data_collection.snapshot_leverage_aum || echo "WARN: leverage AUM snapshot failed"

echo "[$(date '+%F %T')] done"
