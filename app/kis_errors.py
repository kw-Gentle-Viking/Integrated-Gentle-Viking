"""KIS 주문 실패를 재시도할지 판단한다. 일시 장애(네트워크, 초당/분당 요청 제한)만 재시도하고,
영업일 아님·잔고 부족·주문 조건 오류처럼 다시 해도 같은 결과가 나오는 건 즉시 중단한다."""
import requests

RETRYABLE_MSG_CODES = {
    "EGW00133",  # 접근토큰 발급 1분당 1회 제한
    "EGW00201",  # 초당 거래건수 초과
}


class KISOrderError(Exception):
    def __init__(self, message: str, msg_cd: str | None = None):
        super().__init__(message)
        self.msg_cd = msg_cd
        self.retryable = bool(msg_cd and msg_cd in RETRYABLE_MSG_CODES)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, KISOrderError):
        return exc.retryable
    return isinstance(exc, (requests.exceptions.RequestException, TimeoutError, ConnectionError))
