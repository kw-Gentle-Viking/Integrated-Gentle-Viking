import numpy as np
import pandas as pd
import torch

from training.config import build_tft_config
from training.run_gbm_meta import prepare_gbm_meta_data
from training.test_run_tfx_experiments import FakeData

# FakeData.frames() only ever produces columns from training.config.HISTORICAL_COLS_DEFAULT (+ label/
# static/future) -- it ignores champion["columns"] entirely, so the champion fixture used with FakeData
# MUST pick real names out of that list (not arbitrary "f1"/"f2"; those don't exist in FakeData's frames).
CHAMPION = {"columns": ["log_ret", "disparity_5d"], "state_size": 4, "attention_heads": 2,
           "lstm_layers": 1, "dropout": 0.0}


def _toy_model_checkpoint(tmp_path):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from tft_torch.tft import TemporalFusionTransformer
    cfg = build_tft_config(
        {"historical": CHAMPION["columns"], "future": KNOWN_FUTURE_COLS, "static_cardinalities": [21, 3]},
        num_classes=3, state_size=CHAMPION["state_size"], attention_heads=CHAMPION["attention_heads"],
        lstm_layers=CHAMPION["lstm_layers"], dropout=CHAMPION["dropout"])
    model = TemporalFusionTransformer(cfg)
    path = tmp_path / "toy.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def _fake_tabular_df():
    """FakeData가 내는 모든 (ticker, date) 조합을 커버하는 합성 F3 표 피처. label이 NaN인 행
    (FakeData가 마지막 행 등에 일부러 넣는 결측)은 실제 feature_pool처럼 제외한다."""
    data = FakeData()
    rows = []
    for split in ("train", "val_2024", "oot_2026"):
        frames = data.frames(split, "label")
        for tk, df in frames.items():
            for _, r in df.iterrows():
                if pd.isna(r["label"]):
                    continue
                row = {c: r[c] for c in CHAMPION["columns"]}
                row.update({"ticker": tk, "date_s": r["trade_date"].strftime("%Y-%m-%d"),
                           "label": int(r["label"]), "next_day_return": 0.001})
                rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "date_s"])


def test_prepare_gbm_meta_data_returns_joined_frames_for_all_windows(tmp_path):
    ckpt = _toy_model_checkpoint(tmp_path)
    out = prepare_gbm_meta_data(
        dsn=None, champion=CHAMPION, checkpoint_path=ckpt, device=torch.device("cpu"),
        data=FakeData(), tabular_df=_fake_tabular_df())
    assert set(out) == {"train", "val_2024", "oot_2026"}
    for name, df in out.items():
        assert len(df) > 0, name
        assert "emb_0" in df.columns and "emb_3" in df.columns  # state_size=4
        assert "label" in df.columns and "next_day_return" in df.columns
        assert df["date_s"].min() >= ("2019-01-01" if name == "train" else
                                      "2024-01-01" if name == "val_2024" else "2026-01-01")
        assert not ((df["date_s"] >= "2025-01-01") & (df["date_s"] <= "2025-12-31")).any()


from training.run_gbm_meta import fit_and_score


def _synthetic_prepared():
    """emb_0이 next_day_return과 상관되도록 만들어 IC가 유의미하게 나오는 합성 데이터. 20종목/일
    (compute_signal_metrics의 기본 min_names_per_day=20을 정확히 채움 -- 5종목/일이던 첫 시도는
    모든 날이 스킵돼 IC가 전부 None이 됐음), train은 2022-06-01 기준 400영업일이라 fit(<=2022)과
    select(2023) 양쪽에 실제로 행이 생김 (2019년부터 시작한 첫 시도는 2023년까지 전혀 못 미쳐
    select 구간이 0행이었음)."""
    rng = np.random.default_rng(0)
    n_tickers = 20

    def _mk(n_days, start):
        dates = np.repeat(pd.bdate_range(start, periods=n_days), n_tickers)
        n = len(dates)
        ret = rng.standard_normal(n) * 0.01
        emb0 = ret * 5 + rng.standard_normal(n) * 0.001  # 신호
        label = np.where(ret > 0.005, 0, np.where(ret < -0.005, 2, 1))
        df = pd.DataFrame({"date_s": [d.strftime("%Y-%m-%d") for d in dates],
                           "ticker": [f"T{i % n_tickers}" for i in range(n)],
                           "emb_0": emb0, "emb_1": rng.standard_normal(n),
                           "f1": rng.standard_normal(n), "label": label,
                           "next_day_return": ret})
        return df
    return {"train": _mk(400, "2022-06-01"), "val_2024": _mk(20, "2024-01-02"),
            "oot_2026": _mk(20, "2026-01-02")}


def test_fit_and_score_selects_via_2023_split_and_scores_val_oot():
    prepared = _synthetic_prepared()
    rec = fit_and_score(prepared)
    assert rec["chosen_n_iter"] in (25, 50, 100, 200, 300, 400)
    for w in ("val_2024", "oot_2026"):
        assert rec[w]["signal"]["mean_daily_rank_ic"] is not None
        assert 0.0 <= rec[w]["metrics"]["macro_f1"] <= 1.0
    assert rec["n_features"] == 3  # emb_0, emb_1, f1 (ticker/date_s/label/next_day_return excluded)


def test_fit_and_score_raises_if_classes_not_0_1_2():
    prepared = _synthetic_prepared()
    prepared["train"] = prepared["train"][prepared["train"]["label"] != 2]  # drop sell entirely
    import pytest
    with pytest.raises(ValueError, match="classes_"):
        fit_and_score(prepared)


import json as _json


def test_main_writes_result_json_and_doc(tmp_path, monkeypatch):
    # main()'s own job is I/O orchestration (call prepare -> fit_and_score -> write json/doc/
    # model_versions), not re-proving embedding extraction (already covered by Task 3's test) or
    # HGB fitting (already covered above) -- so prepare_gbm_meta_data is stubbed to return the
    # same well-formed synthetic frames fit_and_score's own tests use. FakeData's train window
    # (a handful of months, sized only for the TFX window-count assertions) has no rows before
    # 2023 at all, so it can't exercise fit_and_score's fit<=2022/select-2023 split -- that's a
    # fixture mismatch between Task 3's and Task 4's needs, not something to route real (slow,
    # DB/checkpoint-dependent) prepare_gbm_meta_data around.
    import training.run_gbm_meta as gm
    monkeypatch.setattr(gm, "prepare_gbm_meta_data", lambda *a, **k: _synthetic_prepared())
    out_json, doc_path, mv_path = tmp_path / "res.json", tmp_path / "doc.md", tmp_path / "mv.md"
    mv_path.write_text("# model_versions\n")
    rc = gm.main(dsn=None, champion=CHAMPION, checkpoint_path="unused", out_json=str(out_json),
                doc_path=str(doc_path), model_versions_path=str(mv_path))
    assert rc == 0
    result = _json.loads(out_json.read_text())
    assert "val_2024" in result and "oot_2026" in result
    assert "gbm-meta" in mv_path.read_text()
    # idempotent: 다시 돌려도 model_versions에 중복 행이 안 생김
    gm.main(dsn=None, champion=CHAMPION, checkpoint_path="unused", out_json=str(out_json),
            doc_path=str(doc_path), model_versions_path=str(mv_path))
    assert mv_path.read_text().count("gbm-meta") == 1
