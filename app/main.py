import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.health import router as health_router
from app.routes_kis import router as kis_router
from app.db import Base, engine
import app.models

from app.routes_auth import router as auth_router
from app.routes_users import router as users_router
from app.routes_prices import router as prices_router
from backtest.routes import router as backtest_router

from app.routes_rec import router as rec_router

from app.routes_basket import router as basket_router
from app.routes_trade import router as trade_router

from app.routes_ai_webhook import router as ai_webhook_router
from app.routes_ai_command import router as ai_command_router
from app.routes_market import router as market_router
from app.routes_kis import router as kis_router
from app.demo import demo_mode_enabled, ensure_demo_user
from app.db import SessionLocal

from contextlib import asynccontextmanager


from app.security_guards import validate_production_secrets

# APP_ENV=production 이면 개발용 기본 시크릿/데모 모드로는 기동하지 않는다 (GCP 배포 안전장치)
validate_production_secrets()

# Base.metadata.drop_all(bind=engine)

Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app):
    from app.seed_rec import seed
    seed()
    yield


app = FastAPI(
    title="Control Plane",
    version="0.1.0",
)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health")
app.include_router(users_router, prefix="/users", tags=["users"])
app.include_router(prices_router, prefix="/prices", tags=["prices"])
app.include_router(auth_router,prefix="/auth", tags=["auth"])

app.include_router(backtest_router, prefix="/backtest", tags=["backtest"])

app.include_router(rec_router, prefix="/recommendation", tags=["recommendation"])

app.include_router(basket_router, prefix="/basket", tags=["basket"])
app.include_router(trade_router, prefix="/trade", tags=["trade"])

app.include_router(ai_webhook_router, prefix="/ai", tags=["ai-webhook"])
app.include_router(ai_command_router, prefix="/ai", tags=["ai-command"])
app.include_router(market_router, prefix="/market", tags=["market"])
app.include_router(kis_router, tags=["kis"])


@app.get("/")
def root():
    return {"status": "ok", "service": "control-plane"}


@app.on_event("startup")
def setup_local_demo_data():
    if not demo_mode_enabled():
        return
    db = SessionLocal()
    try:
        ensure_demo_user(db)
    finally:
        db.close()
