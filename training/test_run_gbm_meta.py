import numpy as np
import pandas as pd
import torch

from training.config import build_tft_config
from training.run_gbm_meta import build_tabular_df_for_gbm, prepare_gbm_meta_data
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


def test_build_tabular_df_for_gbm_keeps_only_f3_plus_keys_not_row_no():
    # Code review finding (Critical): build_features() also returns a `row_no` column (a per-ticker
    # cumulative row counter, added only so run_tabular_baseline.py can filter warmup rows) that is
    # NOT part of F3. The old prepare_gbm_meta_data discarded build_features' `fsets` return value
    # and kept every non-excluded column of the raw df, so row_no silently rode along into the GBM's
    # feature matrix -- a near-perfect proxy for calendar time (train rows always have a smaller
    # row_no than any val/OOT row), invalidating the F3 comparison. This locks the fix: only F3's own
    # columns (plus the join/label/target keys) survive.
    from training.run_tabular_baseline import RANK_BASE, LAGS
    champ_cols = ["log_ret", "disparity_5d"]
    n = 70
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    raw = pd.DataFrame({
        "ticker": "A", "trade_date": dates,
        "log_ret": np.random.default_rng(0).standard_normal(n) * 0.01,
        "disparity_5d": np.random.default_rng(1).standard_normal(n) * 0.01,
        "disparity_20d": np.zeros(n), "disparity_60d": np.zeros(n), "rsi_14": np.zeros(n),
        "volatility_20d": np.ones(n) * 0.01,
        "next_day_return": np.zeros(n), "label_vn": np.zeros(n, dtype=int), "label": np.ones(n, dtype=int),
    })
    out = build_tabular_df_for_gbm(raw, champ_cols)
    assert "row_no" not in out.columns
    expected_f3 = set(champ_cols + [f"log_ret_lag{k}" for k in LAGS] +
                      [f"{c}_csr" for c in RANK_BASE + [f"log_ret_lag{k}" for k in LAGS]])
    key_cols = {"ticker", "trade_date", "date_s", "label", "label_vn", "next_day_return"}
    assert set(out.columns) == expected_f3 | key_cols


def test_embed_split_uses_run_tfx_experiments_encoder_len_constant(tmp_path, monkeypatch):
    # Code review finding (Important, Review Focus #5): _embed_split used to hardcode encoder_len=60
    # as its own separate literal instead of the run_tfx_experiments.ENCODER_LEN the val/OOT path
    # (build_window_dataset) reads. They happened to agree only because nobody had changed either
    # one yet. This proves they are now wired to the same source.
    import training.dataset as dsmod
    import training.run_tfx_experiments as tfx
    from training.run_gbm_meta import _embed_split, _load_model

    captured = {}
    orig_init = dsmod.TickerDayDataset.__init__

    def spy_init(self, ticker_dfs, historical_cols, future_cols, static_cols, encoder_len=60, align="legacy"):
        captured["encoder_len"], captured["align"] = encoder_len, align
        return orig_init(self, ticker_dfs, historical_cols, future_cols, static_cols, encoder_len, align)

    monkeypatch.setattr(dsmod.TickerDayDataset, "__init__", spy_init)
    monkeypatch.setattr(tfx, "ENCODER_LEN", 40)

    ckpt = _toy_model_checkpoint(tmp_path)
    model = _load_model(CHAMPION, ckpt, torch.device("cpu"))
    frames = FakeData().frames("train", "label")
    _embed_split("train", model, frames, CHAMPION["columns"], torch.device("cpu"), align="today")
    assert captured["encoder_len"] == 40
    assert captured["align"] == "today"


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


from training.run_gbm_meta import render_doc


def _sample_rec(chosen_n_iter=400):
    sig = {"mean_daily_rank_ic": 0.02, "ic_ir": 0.1}
    per_window = lambda ic_se: {"signal": sig, "ic_se": ic_se, "metrics": {"macro_f1": 0.3}}  # noqa: E731
    return {"chosen_n_iter": chosen_n_iter, "n_features": 81,
           "select_ic_2023_by_iter": {"25": 0.01, "50": 0.02, "100": 0.03, "200": 0.05, "300": 0.06, "400": 0.061},
           "val_2024": per_window(0.0099), "oot_2026": per_window(0.0150)}


def test_render_doc_includes_ic_se_column():
    # Code review finding (Important): ic_se is already computed by bundle() and saved in the JSON,
    # but the rendered doc dropped it -- so a reader could not tell 0.047 (2 SE) from 0.047 (6 SE).
    doc = render_doc(_sample_rec(), "abc1234")
    assert "0.0099" in doc and "0.0150" in doc


def test_render_doc_notes_val_selected_checkpoint_leakage():
    # Code review finding (Important, from Recommendations): the R0 checkpoint used for embedding
    # extraction was itself early-stopped on val-2024 loss, so val_2024 IC here is a mild upward bias
    # relative to OOT 2026 (which the checkpoint never influenced) -- the spec requires this kind of
    # caveat to be stated in the doc, not just known informally. Checking a specific phrase, not a
    # bare "val" substring -- the table header already contains "val_2024" and would pass vacuously.
    doc = render_doc(_sample_rec(), "abc1234")
    assert "체크포인트" in doc and "낙관" in doc


def test_render_doc_warns_when_selected_n_iter_is_at_grid_boundary():
    # Code review finding (Important): chosen_n_iter=400 is the largest value in HGB_ITERS and the
    # 2023-select IC was still rising there (0.06->0.061) -- the doc should flag that the optimum may
    # lie outside the tried grid, exactly as run_tabular_baseline.py's own docs already do for the
    # same situation (INTERPRETATION's "still falling/rising" caveats).
    doc = render_doc(_sample_rec(chosen_n_iter=400), "abc1234")
    assert "그리드" in doc or "경계" in doc
    doc_mid = render_doc(_sample_rec(chosen_n_iter=100), "abc1234")
    assert "그리드" not in doc_mid and "경계" not in doc_mid


import json as _json


def test_main_writes_reproducibility_meta(tmp_path, monkeypatch):
    # Code review finding (Important): the spec requires seed/checkpoint-hash/feature-set/commit
    # metadata "기존 지문 패턴과 동일하게" (same as the other runners' fingerprint/meta pattern), and
    # the project's own standing rule is that every experiment output records its run conditions.
    # The old main() saved only a bare `code_commit`.
    import training.run_gbm_meta as gm
    monkeypatch.setattr(gm, "prepare_gbm_meta_data", lambda *a, **k: _synthetic_prepared())
    ckpt = _toy_model_checkpoint(tmp_path)
    out_json, doc_path, mv_path = tmp_path / "res.json", tmp_path / "doc.md", tmp_path / "mv.md"
    mv_path.write_text("# model_versions\n")
    gm.main(dsn=None, champion=CHAMPION, checkpoint_path=ckpt, out_json=str(out_json),
           doc_path=str(doc_path), model_versions_path=str(mv_path))
    result = _json.loads(out_json.read_text())
    meta = result["meta"]
    assert meta["seed"] == 0
    assert meta["data_version"]  # non-empty
    assert meta["hgb_params"]["random_state"] == 0
    assert len(meta["tft_checkpoint"]["sha256"]) == 64
    assert meta["tft_checkpoint"]["path"] == ckpt
    assert "+dirty" in result["code_commit"] or len(result["code_commit"]) >= 7  # short sha, dirty-checked


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
