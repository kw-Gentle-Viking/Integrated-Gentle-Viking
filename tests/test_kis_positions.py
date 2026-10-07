"""trading_loop가 매 사이클 Portfolio()를 새로 만들고 체결 후에도 positions를 갱신하지 않아서, 전략의
'이미 보유 중인지' 판단(current_qty)이 영원히 거짓이었다 -- 같은 종목 반복 매수가 무한정 가능했고
매도(청산) 조건은 거의 발화하지 않았다(2026-10-02 통합 감사). app.kis_positions는 KIS 실제 잔고를
그대로 읽어 이 상태를 채운다."""
import pytest

from app.kis_positions import apply_fill, ensure_fresh_token, load_live_positions, parse_balance_positions
from backtest.engine.risk import Portfolio


class _TokenStub:
    """mojito 토큰 메서드만 흉내낸다. load_live_positions가 매 사이클 호출하므로 다른 가짜 broker들도
    이걸 섞어 쓴다."""
    def __init__(self, valid: bool):
        self.valid = valid
        self.loaded = False
        self.issued = False

    def check_access_token(self):
        return self.valid

    def load_access_token(self):
        self.loaded = True

    def issue_access_token(self):
        self.issued = True


def test_parse_balance_positions_builds_position_per_held_ticker():
    balance = {"output1": [
        {"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "70000.0000"},
        {"pdno": "000660", "hldg_qty": "5", "pchs_avg_pric": "150000"},
    ]}
    positions = parse_balance_positions(balance)
    assert set(positions) == {"005930", "000660"}
    assert positions["005930"].qty == 10 and positions["005930"].entry_price == 70000


def test_parse_balance_positions_drops_zero_quantity_rows():
    # 전량 매도한 과거 종목이 output1에 수량 0으로 계속 잡혀 나오는 경우가 있다.
    balance = {"output1": [{"pdno": "005930", "hldg_qty": "0", "pchs_avg_pric": "70000"}]}
    assert parse_balance_positions(balance) == {}


def test_parse_balance_positions_handles_missing_or_empty_output1():
    assert parse_balance_positions({}) == {}
    assert parse_balance_positions(None) == {}
    assert parse_balance_positions({"output1": []}) == {}


def test_load_live_positions_raises_when_broker_not_configured():
    # 조용히 '포지션 없음'으로 넘어가면 이 모듈이 고치려는 버그(포지션 추적 없음)가 재현된다 --
    # 호출부(trading_loop)가 이 예외를 받아 사이클을 스킵해야 한다.
    with pytest.raises(RuntimeError):
        load_live_positions(None)


def test_load_live_positions_propagates_broker_fetch_failures():
    class FailingBroker(_TokenStub):
        def fetch_balance(self):
            raise ConnectionError("KIS 연결 실패")

    with pytest.raises(ConnectionError):
        load_live_positions(FailingBroker(valid=True))


def test_load_live_positions_parses_real_broker_response_shape():
    class FakeBroker(_TokenStub):
        def fetch_balance(self):
            return {"output1": [{"pdno": "005930", "hldg_qty": "3", "pchs_avg_pric": "71000"}],
                    "output2": [{"dnca_tot_amt": "1000000"}]}

    positions = load_live_positions(FakeBroker(valid=True))
    assert positions["005930"].qty == 3


# ---- ensure_fresh_token: 장시간 실행되는 프로세스에서 만료된 토큰이 그대로 남는 사고 (2026-10-07) ----
def test_ensure_fresh_token_reloads_when_cached_token_still_valid():
    broker = _TokenStub(valid=True)
    ensure_fresh_token(broker)
    assert broker.loaded is True and broker.issued is False


def test_ensure_fresh_token_reissues_when_cached_token_expired():
    broker = _TokenStub(valid=False)
    ensure_fresh_token(broker)
    assert broker.issued is True and broker.loaded is False


def test_load_live_positions_refreshes_token_before_fetching_balance():
    class FakeBroker(_TokenStub):
        def fetch_balance(self):
            assert self.loaded or self.issued, "fetch_balance 전에 토큰을 갱신해야 한다"
            return {"output1": []}

    load_live_positions(FakeBroker(valid=False))


# ---- apply_fill: 같은 사이클 내 즉시 반영 (다음 KIS 조회를 기다리지 않음) ----
def test_apply_fill_buy_opens_a_new_position():
    portfolio = Portfolio(cash=1_000_000, equity=1_000_000)
    apply_fill(portfolio, "005930", "BUY", qty=5, price=70_000)
    pos = portfolio.positions["005930"]
    assert pos.qty == 5 and pos.entry_price == 70_000


def test_apply_fill_buy_averages_entry_price_on_a_second_buy():
    portfolio = Portfolio(cash=1_000_000, equity=1_000_000)
    apply_fill(portfolio, "005930", "BUY", qty=10, price=70_000)   # 평단 70,000
    apply_fill(portfolio, "005930", "BUY", qty=10, price=80_000)   # 추가매수
    pos = portfolio.positions["005930"]
    assert pos.qty == 20 and pos.entry_price == 75_000            # (10*70000 + 10*80000) / 20


def test_apply_fill_sell_reduces_qty_and_clears_entry_price_at_zero():
    portfolio = Portfolio(cash=0, equity=1_000_000)
    apply_fill(portfolio, "005930", "BUY", qty=10, price=70_000)
    apply_fill(portfolio, "005930", "SELL", qty=10, price=75_000)  # 전량 매도
    pos = portfolio.positions["005930"]
    assert pos.qty == 0 and pos.entry_price == 0                  # current_qty <= 0 판단이 다시 정확해짐


def test_apply_fill_sell_never_goes_negative():
    portfolio = Portfolio(cash=0, equity=1_000_000)
    apply_fill(portfolio, "005930", "BUY", qty=5, price=70_000)
    apply_fill(portfolio, "005930", "SELL", qty=999, price=75_000)  # 보유량보다 큰 매도 요청(방어적)
    assert portfolio.positions["005930"].qty == 0
