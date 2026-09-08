import time
from datetime import datetime, time as dtime
from typing import Optional
import requests

# 기존 운영 수집기가 KIS API를 사용 중인 시간대(장중 실시간, collector_kis, collector_batch) —
# 이 구간엔 신규 백필 스크립트를 절대 실행하지 않는다 (토큰 충돌 방지, 설계 §9.1 참고)
PRODUCTION_WINDOWS = [
    (dtime(8, 55), dtime(15, 30)),
    (dtime(15, 50), dtime(16, 0)),
    (dtime(16, 0), dtime(16, 10)),
]


def assert_outside_production_window(now: Optional[datetime] = None) -> None:
    now = now or datetime.now()
    if now.weekday() >= 5:  # 토(5)/일(6)
        return
    t = now.time()
    for start, end in PRODUCTION_WINDOWS:
        if start <= t < end:
            raise RuntimeError(
                f"{now.time()} falls inside a production collection window "
                f"({start}-{end}) — reschedule this backfill outside market hours."
            )


class KisClient:
    def __init__(self, app_key: str, app_secret: str,
                 base_url: str = "https://openapi.koreainvestment.com:9443"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self._token: Optional[str] = None
        self._token_issued_at: float = 0.0

    def get_token(self) -> str:
        if self._token and (time.time() - self._token_issued_at) < 23 * 3600:
            return self._token
        resp = requests.post(
            f"{self.base_url}/oauth2/tokenP",
            json={"grant_type": "client_credentials",
                  "appkey": self.app_key, "appsecret": self.app_secret},
            timeout=10,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        self._token_issued_at = time.time()
        return self._token

    def request(self, path: str, tr_id: str, params: dict) -> dict:
        assert_outside_production_window()
        headers = {
            "authorization": f"Bearer {self.get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }
        resp = requests.get(f"{self.base_url}{path}", headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
