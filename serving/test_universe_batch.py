import json
import math
from datetime import date, timedelta

import pytest

from serving import universe_batch as ub
from serving.model import get_feature_columns

COLS = get_feature_columns()
ASOF = date(2026, 9, 8)


# ---------------------------------------------------------------- compute_rank_pct (pure)
def test_rank_pct_orders_high_score_as_high_percentile():
    r = ub.compute_rank_pct({"A": -0.5, "B": 0.0, "C": 0.9})
    assert r == {"A": 0.0, "B": 0.5, "C": 1.0}


def test_rank_pct_ties_share_the_average_rank():
    r = ub.compute_rank_pct({"A": 0.1, "B": 0.5, "C": 0.5, "D": 0.9})
    assert r["A"] == 0.0 and r["D"] == 1.0
    assert r["B"] == r["C"] == pytest.approx(0.5)  # average of ranks 1 and 2 of 0..3 -> 1.5/3


def test_rank_pct_all_tied_is_all_half():
    r = ub.compute_rank_pct({"A": 0.2, "B": 0.2, "C": 0.2})
    assert set(r.values()) == {0.5}


def test_rank_pct_excludes_nan_and_inf_and_ranks_the_rest():
    r = ub.compute_rank_pct({"A": 0.1, "B": float("nan"), "C": 0.7, "D": float("inf"), "E": None})
    assert r["B"] is None and r["D"] is None and r["E"] is None
    assert r["A"] == 0.0 and r["C"] == 1.0


def test_rank_pct_fewer_than_two_valid_scores_gives_none():
    assert ub.compute_rank_pct({}) == {}
    assert ub.compute_rank_pct({"A": 0.3}) == {"A": None}
    assert ub.compute_rank_pct({"A": 0.3, "B": float("nan")}) == {"A": None, "B": None}


def test_rank_pct_values_within_0_1():
    scores = {f"T{i}": math.sin(i) for i in range(50)}
    r = ub.compute_rank_pct(scores)
    assert all(0.0 <= v <= 1.0 for v in r.values())
    assert min(r.values()) == 0.0 and max(r.values()) == 1.0


# ---------------------------------------------------------------- score_universe (no DB)
def _dates(n, end=ASOF):
    """n consecutive weekdays ending at `end` (an as-of weekday)."""
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _rows(n, end=ASOF, close=100.0):
    rows = []
    for d in _dates(n, end):
        r = {c: 0.01 for c in COLS["historical"]}
        r.update({c: 0 for c in ("is_bok", "is_fomc", "is_witching_kr", "is_witching_us")})
        r.update({"trade_date": d, "close_price": close, "open_price": close, "high_price": close * 1.01,
                  "low_price": close * 0.99, "volume": 1000.0, "sector_id": 3, "market_id": 1,
                  "day_of_week": float(d.weekday())})
        rows.append(r)
    return rows


def _fake_predict(table):
    def predict(ticker, encoder_df):
        if ticker not in table:
            raise RuntimeError("boom")
        b, h, s = table[ticker]
        return {"prob_buy": b, "prob_hold": h, "prob_sell": s}
    return predict


def _market_dates(rows_by_ticker):
    return {r["trade_date"] for rows in rows_by_ticker.values() for r in rows}


def test_score_universe_scores_ranks_and_records_exclusions_and_failures():
    rows_by_ticker = {
        "AAA": _rows(70), "BBB": _rows(70), "CCC": _rows(70),
        "NEW": _rows(30),                               # insufficient history
        "STALE": _rows(70, end=ASOF - timedelta(days=7)),  # no row on as-of
        "BOOM": _rows(70),                              # predict raises
    }
    predict = _fake_predict({"AAA": (0.6, 0.3, 0.1), "BBB": (0.2, 0.3, 0.5), "CCC": (0.34, 0.33, 0.33)})
    res = ub.score_universe(["AAA", "BBB", "CCC", "NEW", "STALE", "BOOM", "GONE"], rows_by_ticker,
                            _market_dates({k: v for k, v in rows_by_ticker.items() if k != "STALE"}),
                            ASOF, predict, COLS)
    by = {r["ticker"]: r for r in res["scores"]}
    assert set(by) == {"AAA", "BBB", "CCC"}
    assert by["AAA"]["score"] == pytest.approx(0.5) and by["BBB"]["score"] == pytest.approx(-0.3)
    assert by["AAA"]["rank_pct"] == 1.0 and by["BBB"]["rank_pct"] == 0.0
    assert 0.0 < by["CCC"]["rank_pct"] < 1.0
    assert [r["ticker"] for r in res["scores"]] == ["AAA", "CCC", "BBB"]  # best (buy) first
    ex = {e["ticker"]: e for e in res["excluded"]}
    assert ex["NEW"]["reason"] == "insufficient_history" and "30" in ex["NEW"]["detail"]
    assert ex["STALE"]["reason"] == "no_row_on_asof"
    assert ex["GONE"]["reason"] == "no_row_on_asof"
    fl = {f["ticker"]: f for f in res["failed"]}
    assert fl["BOOM"]["reason"] == "error" and "boom" in fl["BOOM"]["detail"]


def test_score_universe_flags_history_gap_vs_market_dates():
    rows = _rows(70)
    holey = _rows(70)
    del holey[-20]   # a market trading day missing inside the 60-row window
    res = ub.score_universe(["OK", "HOLEY"], {"OK": rows, "HOLEY": holey}, _market_dates({"OK": rows}),
                            ASOF, _fake_predict({"OK": (0.5, 0.3, 0.2), "HOLEY": (0.5, 0.3, 0.2)}), COLS)
    assert [r["ticker"] for r in res["scores"]] == ["OK"]
    assert res["excluded"][0]["ticker"] == "HOLEY" and res["excluded"][0]["reason"] == "history_gap"


def test_score_universe_non_finite_output_is_a_failure_not_a_score():
    rows = {"A": _rows(70), "B": _rows(70), "C": _rows(70)}
    res = ub.score_universe(["A", "B", "C"], rows, _market_dates(rows), ASOF,
                            _fake_predict({"A": (0.5, 0.3, 0.2), "B": (float("nan"), 0.3, 0.2),
                                           "C": (0.1, 0.3, 0.6)}), COLS)
    assert {r["ticker"] for r in res["scores"]} == {"A", "C"}
    assert res["failed"][0]["ticker"] == "B" and res["failed"][0]["reason"] == "non_finite_output"


def test_score_universe_uses_the_completed_asof_bar_as_todays_row():
    seen = {}

    def predict(ticker, df):
        seen["df"] = df
        return {"prob_buy": 0.5, "prob_hold": 0.3, "prob_sell": 0.2}

    rows = {"A": _rows(70)}
    rows["A"][-1]["close_price"] = 110.0  # as-of close jumped +10% vs 100
    ub.score_universe(["A"], rows, _market_dates(rows), ASOF, predict, COLS)
    df = seen["df"]
    assert len(df) == 60
    assert df["log_ret"].iloc[-1] == pytest.approx(math.log(1.1))
    assert (df["time_progress"] == 1.0).all()


# ---------------------------------------------------------------- default as-of / output path / io
def test_default_out_path_is_training_artifacts_universe_scores_asof():
    p = ub.default_out_path(date(2026, 9, 8))
    assert p.endswith("training/artifacts/universe_scores_2026-09-08.json")


def test_pick_asof_uses_latest_date_with_enough_universe_coverage():
    coverage = {date(2026, 9, 8): 190, date(2026, 9, 9): 3}  # 9/9 is a partial batch
    assert ub.pick_asof(coverage, n_universe=200) == date(2026, 9, 8)
    with pytest.raises(ValueError):
        ub.pick_asof({date(2026, 9, 9): 3}, n_universe=200)


def test_run_batch_writes_json_with_metadata(tmp_path, monkeypatch):
    rows = {"A": _rows(70), "B": _rows(70)}
    monkeypatch.setattr(ub, "load_universe", lambda dsn: ["A", "B", "C"])
    monkeypatch.setattr(ub, "resolve_asof", lambda dsn, universe, asof=None: asof or ASOF)
    monkeypatch.setattr(ub, "load_universe_rows", lambda dsn, universe, asof, cols, n_rows=60, lookback_days=150: (rows, _market_dates(rows)))
    monkeypatch.setattr(ub, "_load_model", lambda: object())
    monkeypatch.setattr(ub, "make_predict_fn", lambda model, cols: _fake_predict({"A": (0.7, 0.2, 0.1), "B": (0.1, 0.2, 0.7)}))
    out = tmp_path / "scores.json"
    res = ub.run_batch("dsn", asof=None, out=str(out))
    data = json.loads(out.read_text())
    assert data["asof"] == "2026-09-08"
    assert [r["ticker"] for r in data["scores"]] == ["A", "B"]
    assert data["scores"][0]["rank_pct"] == 1.0 and data["scores"][1]["rank_pct"] == 0.0
    assert data["n_universe"] == 3 and data["n_scored"] == 2
    assert {e["ticker"] for e in data["excluded"]} == {"C"}
    for k in ("generated_at", "model_version", "model_path", "git_commit", "elapsed_sec"):
        assert k in data
    assert res["n_scored"] == 2


def test_cli_parses_asof_and_out():
    ns = ub.parse_args(["--asof", "2026-09-08", "--out", "/tmp/x.json"])
    assert ns.asof == date(2026, 9, 8) and ns.out == "/tmp/x.json"
    ns = ub.parse_args([])
    assert ns.asof is None and ns.out is None
