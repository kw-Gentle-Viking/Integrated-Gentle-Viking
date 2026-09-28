import os

# app.db / backtest.db 는 import 시점에 create_engine 을 부른다. 테스트에서는 실제 Postgres 없이
# sqlite 로 대체한다 (psycopg2 미설치 환경에서도 import 가능).
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("MARKET_DB_URL", "sqlite:///:memory:")
