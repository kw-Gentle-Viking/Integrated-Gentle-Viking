#!/bin/bash
# 자동 점검 (cron). 결과는 /home/user/ops_logs/status.log 에 한 줄씩 남긴다.
#   preflight : 장 전 점검 (preflight_check.sh 전체 결과 보관)
#   watchdog  : 백엔드(8000)·추론 API(8001) 가 죽었으면 다시 띄운다
#   open      : 장 시작 후 추론·신호가 들어왔는지
#   close     : 장 마감 후 하루 요약
REPO=/home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
BACK=/home/user/team_repos/Back-Gentle-Viking
LOGDIR=/home/user/ops_logs
STATUS=$LOGDIR/status.log
PY=/home/user/miniconda3/envs/dl_env/bin/python
BPY=/home/user/venvs/backend/bin/python
mode=$1
log(){ echo "$(date '+%F %T') [$1] $2" >> $STATUS; }

set -a; . $REPO/.env; set +a
dow=$(date +%u)
if [ "$mode" != "watchdog" ]; then
  [ "$dow" -ge 6 ] && exit 0
  open_day=$($PY -W ignore -c "from datetime import date; from serving.market_calendar import is_market_open_day as f; print(f(date.today()))" 2>/dev/null)
  [ "$open_day" = "False" ] && { log INFO "$mode: 휴장일 -> 건너뜀"; exit 0; }
fi

case "$mode" in
preflight)
  out=$LOGDIR/preflight_$(date +%F).log
  (cd $REPO && bash scripts/preflight_check.sh) > $out 2>&1
  fails=$(grep -c "^FAIL" $out)
  if [ "$fails" = "0" ]; then log OK "preflight 통과 ($out)"; else log FAIL "preflight 실패 $fails건: $(grep '^FAIL' $out | tr '\n' ' ')"; fi
  ;;
watchdog)
  if ! curl -fs -m 5 localhost:8000/health >/dev/null; then
    (cd $BACK && set -a && . ./.env && set +a && setsid nohup $BPY -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> $LOGDIR/backend.log 2>&1 &)
    sleep 8; curl -fs -m 5 localhost:8000/health >/dev/null && log WARN "백엔드 재기동 성공" || log FAIL "백엔드 재기동 실패"
  fi
  if ! curl -fs -m 5 localhost:8001/health >/dev/null; then
    (cd $REPO && set -a && . ./.env && set +a && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. setsid nohup $PY -m uvicorn serving.api_server:app --host 0.0.0.0 --port 8001 >> $LOGDIR/api_server.log 2>&1 &)
    sleep 8; curl -fs -m 5 localhost:8001/health >/dev/null && log WARN "추론 API 재기동 성공" || log FAIL "추론 API 재기동 실패"
  fi
  ;;
open)
  cnt=$($PY -W ignore -c "
import os, psycopg2
c = psycopg2.connect(os.environ['DATABASE_URL']).cursor()
c.execute(\"select count(*) from ai_prediction_history where trade_datetime::date = current_date\")
print(c.fetchone()[0])" 2>/dev/null)
  if [ -n "$cnt" ] && [ "$cnt" -gt 0 ]; then log OK "장 시작 후 추론 기록 ${cnt}건 (오늘)"; else log FAIL "장 시작 후 추론 기록 없음 (5분 추론 확인 필요)"; fi
  errs=$(grep -c "ERROR\|Traceback" $REPO/serving/pipeline.log 2>/dev/null | tail -1)
  log INFO "pipeline.log 오류 줄 누적: ${errs:-0}"
  ;;
close)
  $PY -W ignore -c "
import os, psycopg2
c = psycopg2.connect(os.environ['DATABASE_URL']).cursor()
c.execute(\"select count(*) from ai_prediction_history where trade_datetime::date = current_date\"); p = c.fetchone()[0]
c.execute(\"select status, count(*) from trade_logs where created_at::date = current_date group by 1\"); t = c.fetchall()
c.execute(\"select action, count(*) from auto_trade_decisions where created_at::date = current_date group by 1\"); d = c.fetchall()
print(f'predictions={p} trade_logs={t} decisions={d}')" 2>/dev/null | while read line; do log INFO "마감 요약: $line"; done
  ;;
esac
