#!/bin/bash
# 10/6 장 시작 전 점검 (읽기 전용). 각 항목 OK / WARN / FAIL 출력.
REPO=/home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
set -a; . "$REPO/.env"; set +a
ok(){ echo "OK    $1"; }; warn(){ echo "WARN  $1"; }; fail(){ echo "FAIL  $1"; }

curl -s -m 5 localhost:8000/health | grep -q ok && ok "backend 8000" || fail "backend 8000 down"
curl -s -m 5 localhost:8001/health | grep -q ok && ok "ai serving 8001" || fail "ai serving 8001 down"
curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:3000 | grep -q 200 && ok "frontend 3000" || warn "frontend 3000 down"

cron=$(crontab -l 2>/dev/null)
for pat in "serving.inference_pipeline" "serving.poll_commands" "uvicorn serving.api_server" "daily_v2_refresh.sh"; do
  echo "$cron" | grep -q "$pat" && ok "cron: $pat" || fail "cron missing: $pat"
done

a=$(python3 -c "import json;print(','.join(sorted(json.load(open('$REPO/serving/active_tickers.json')).get('all_tickers',[]))))")
p=$(python3 -c "import json;print(','.join(sorted(json.load(open('/home/user/active_tickers.json')).get('all_tickers',[]))))")
[ "$a" = "$p" ] && ok "ticker lists match: $a" || fail "ticker mismatch: serving=[$a] collector=[$p]"

fp=$(psql "$STOCK_DB_V2_DSN" -tAc "SELECT MAX(trade_date) FROM feature_pool")
[ "$fp" = "$(date -d 'last friday' +%F)" ] || [ "$fp" = "2026-10-02" ] && ok "feature_pool max: $fp" || warn "feature_pool max: $fp"
ticks=$(psql "$STOCK_DB_V2_DSN" -tAc "SELECT COUNT(DISTINCT ticker) FROM feature_pool WHERE trade_date=(SELECT MAX(trade_date) FROM feature_pool)")
[ "$ticks" -ge 200 ] && ok "feature_pool tickers on max date: $ticks" || warn "feature_pool tickers: $ticks"

pm=$(PGPASSWORD=0180 psql -h localhost -U stock_user -d stock_db -tAc "SELECT MAX(trade_date) FROM price_daily")
ok "prod price_daily max: $pm"

if [ -n "${KIS_ACC_NO:-}" ]; then
  code=$(curl -s -m 10 localhost:8000/account/assets -o /dev/null -w "%{http_code}")
  [ "$code" = "200" ] && ok "KIS balance via backend" || fail "KIS balance via backend HTTP $code (계좌 불일치 미해결 시 예상된 실패)"
fi

grep -q "^GEMINI_API_KEY=AQ" "$REPO/../../../team_repos/Back-Gentle-Viking/.env" 2>/dev/null && ok "gemini key set" || warn "gemini key not set"
