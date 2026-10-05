import pandas as pd

from scripts.return_report import fifo_pnl, forward_returns, signal_hits


def test_forward_return_uses_next_trading_day_close():
    px = pd.DataFrame({"ticker": ["A", "A", "A"], "trade_date": ["2026-10-06", "2026-10-07", "2026-10-08"],
                       "close": [100.0, 110.0, 99.0]})
    f = forward_returns(px)
    assert list(f["next_ret"].round(4)) == [0.1, -0.1]


def test_signal_hit_uses_last_signal_of_day_and_skips_hold():
    fwd = pd.DataFrame({"ticker": ["A", "A"], "trade_date": ["2026-10-06", "2026-10-07"], "next_ret": [0.02, -0.01]})
    sig = pd.DataFrame({"ticker": ["A", "A", "A"],
                        "trade_datetime": ["2026-10-06 10:00", "2026-10-06 15:20", "2026-10-07 15:20"],
                        "signal": ["SELL", "BUY", "HOLD"]})
    hits = signal_hits(sig, fwd)
    assert len(hits) == 1 and bool(hits.iloc[0]["hit"]) is True  # 10/6 마지막 신호 BUY, 다음날 +2%


def test_fifo_pnl_matches_lots_in_order():
    t = pd.DataFrame({"ticker": ["A", "A", "A"], "side": ["BUY", "BUY", "SELL"], "qty": [10, 10, 15],
                      "price": [100, 200, 150], "created_at": ["1", "2", "3"]})
    p = fifo_pnl(t)
    assert round(p["pnl"].sum(), 2) == round(10 * 50 + 5 * (150 - 200), 2)
