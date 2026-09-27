# TFT 임베딩 + GBM 메타 모델 (2안, Phase 1: 오프라인만) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** R0 TFT 체크포인트에서 32차원 임베딩을 뽑아 F3 표 피처(반전 정보 포함)와 합친 뒤 GBM을
학습해, val 2024 IC로 확인하고 OOT 2026을 1회 채점한 결과 문서를 만든다. 서빙 통합은 이 결과를
보고 나서 별도 계획(Phase 2)으로 진행한다 — 아래 "범위 축소" 참고.

**Architecture:** `training/tft_embeddings.py`(hook 기반 임베딩 추출) →
`training/gbm_meta_data.py`(임베딩·표 피처 조인) → `training/run_gbm_meta.py`(데이터 준비 +
sklearn HistGradientBoostingClassifier 학습/선택/채점 + 결과 문서). 모두 기존
`training/run_tfx_experiments.py`(TFT 데이터 준비/윈도)와 `training/run_tabular_baseline.py`
(F3 피처·정렬된 샘플셋·HGB 학습 관행)의 함수를 import해서 재사용하고, 재구현하지 않는다.

**Tech Stack:** Python, PyTorch(추론 전용, forward hook), scikit-learn
`HistGradientBoostingClassifier`, pandas, 기존 `evaluation.evaluate.compute_signal_metrics`.

**Spec:** `docs/superpowers/specs/2026-09-27-tft-embedding-gbm-design.md`

## 범위 축소 (spec 대비)

spec은 서빙 통합(라이브 200종목 횡단면 스냅샷, `FINAL_PREDICTOR` 스위치)까지 포함하지만, 이
플랜은 **오프라인 학습/평가까지만** 다룬다. 이유: 서빙 배관을 만들기 전에 "TFT 임베딩 +
GBM이 실제로 반전/HGB-단독보다 나은가"부터 실측해야 그 투자가 정당화된다 — 이는 서로 독립적으로
테스트 가능한 두 하위 시스템이라 writing-plans의 "Scope Check"에 따라 분리한다. 결과가 나오면
서빙 통합(spec의 컴포넌트 3, 4)은 `2026-09-27-tft-embedding-gbm-serving.md`(가칭)로 별도
계획한다.

## Global Constraints

- 학습: train ≤ 2023-12-31만. 선택: 기존 E0/E0v2 관행과 동일하게 train 내부의 시간 분할
  (fit ≤ 2022-12-31, select 2023, 최종 refit ≤ 2023-12-31)로 HGB `n_iter`를 고른다 — val 2024는
  이 선택에 전혀 쓰지 않는다(코드베이스 기존 관행과 일치, spec의 "val IC로 선택"은 이 모델
  자체를 다른 레시피와 비교/채택할지 판단할 때 쓰는 기준이지 HGB 내부 하이퍼파라미터 튜닝
  기준이 아니다 — 본문에 이유 명시).
- 2025 test 구간(2025-01-01~2025-12-31)은 fit/select/평가 어디에도 쓰지 않는다. OOT 2026은
  최종 config로 1회만 채점한다.
- 임베딩 추출은 R0 체크포인트(`training/artifacts/checkpoints/tfx-aligned.pt`, seed 0,
  `align="today"`) 고정. 다른 체크포인트로 바꾸지 않는다.
- `tft-torch`(벤더 코드, `tft-torch/tft_torch/tft.py`)는 수정하지 않는다 — forward hook만 사용.
- 표 피처는 F3(`training/run_tabular_baseline.py`의 `build_features`가 만드는 열, champ 33 +
  log_ret lag 5개 + 랭크 11개 = 49열) 그대로 재사용한다. 재구현 금지.
- 임베딩·표 피처 조인은 (ticker, date) 내부조인이며, 커버리지가 기준 미달이면(아래 Task 2)
  무조건 예외를 던진다 — 조용한 드롭 금지.
- 이 플랜의 모든 스크립트는 CPU에서 돈다(추론 전용 forward pass, 학습은 sklearn). GPU 유휴
  감시자 체이닝은 필요 없다. `CUDA_VISIBLE_DEVICES=""`로 실행할 것.
- 산출물 경로는 모두 신규(`training/artifacts/gbm_meta_results.json`,
  `docs/gbm_meta_results.md`)이고 기존 `tfx_results.json`/`tabular_baseline_v2.json` 등은
  건드리지 않는다(읽기만).
- 운영 `stock_db`/crontab/`/home/user/*.py` 금지. push는 사용자가 명시적으로 요청할 때만.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

## Review Focus

- **조인 커버리지 저하가 조용히 넘어감**: 임베딩 키와 표 피처 키 형식이 미묘하게 달라지면(예:
  `date_s` 포맷 차이) 조인된 행이 크게 줄어도 프로그램이 계속 돈다 — Task 2의
  `assert_join_coverage`가 모든 조인 지점(train/val/oot)에서 반드시 호출되는지 확인.
- **(ticker, date) 중복 키**: 임베딩 추출이나 표 피처 로딩이 같은 키를 두 번 내면 merge가
  행을 부풀리거나 임의로 하나를 고른다 — 조인 전 양쪽 다 유니크 검사.
- **GBM 클래스 순서 가정**: `HistGradientBoostingClassifier.classes_`가 `[0,1,2]`가 아니면
  `p[:,0]-p[:,2]`가 buy/sell이 아닌 다른 클래스를 가리킨다 — fit 직후 반드시 assert.
- **2025 구간 유입**: train/select/full 날짜 경계를 문자열 비교(`date_s <= "2023-12-31"`)로
  하는데, 오프바이원으로 2025 값이 select(2023) 구간에 섞여 들어가면 채점 오염 — 각 서브셋에
  대해 날짜 범위 assert.
- **임베딩 열 순서/정렬 불일치**: train/val/oot 임베딩을 각각 다른 코드 경로로 뽑는데
  (train은 `prepare_recipe`, val/oot는 `build_window_dataset`), 둘이 쓰는 컬럼 리스트
  (`champion["columns"]`)나 `align`이 어긋나면 임베딩이 서로 다른 의미의 벡터가 된다 —
  모든 경로에서 같은 `cols`/`align="today"`를 쓰는지 테스트로 고정.

---

### Task 1: TFT 임베딩 추출기

**Files:**
- Create: `training/tft_embeddings.py`
- Test: `training/test_tft_embeddings.py`

**Interfaces:**
- Consumes: `training.dataset.TickerDayDataset`(이미 존재), `training.signal_data.sample_meta`(이미
  존재), `training.train._to_device_batch`(이미 존재, static 텐서 3D→2D squeeze 재사용).
- Produces: `extract_embeddings(model, dataset, device, batch_size=256) -> tuple[list[tuple[str,str]], np.ndarray]`
  — `(keys, embeddings)`, `keys`는 `dataset` 순서의 `(ticker, date)` 리스트,
  `embeddings.shape == (len(dataset), model_state_size)`. Task 3/4가 이 함수를 그대로 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_tft_embeddings.py
import numpy as np
import pandas as pd
import torch

from training.config import build_tft_config
from training.dataset import TickerDayDataset
from training.tft_embeddings import extract_embeddings

HIST_COLS = ["f1", "f2"]
FUT_COLS = ["fut1"]
STATIC_COLS = ["s1", "s2"]


def _toy_frames(n_tickers=3, n_days=65, seed=0):
    rng = np.random.default_rng(seed)
    frames = {}
    for i in range(n_tickers):
        dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
        df = pd.DataFrame({
            "trade_date": dates,
            "f1": rng.standard_normal(n_days), "f2": rng.standard_normal(n_days),
            "fut1": np.ones(n_days), "s1": rng.integers(0, 2, n_days),
            "s2": rng.integers(0, 2, n_days),
            "label": rng.integers(0, 3, n_days),
        })
        frames[f"T{i}"] = df
    return frames


def _toy_model():
    cfg = build_tft_config(
        {"historical": HIST_COLS, "future": FUT_COLS, "static_cardinalities": [2, 2]},
        num_classes=3, state_size=4, attention_heads=2, lstm_layers=1, dropout=0.0)
    from tft_torch.tft import TemporalFusionTransformer
    return TemporalFusionTransformer(cfg)


def test_extract_embeddings_shape_and_keys_match_dataset_order():
    frames = _toy_frames()
    ds = TickerDayDataset(frames, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60, align="today")
    model = _toy_model()
    keys, emb = extract_embeddings(model, ds, torch.device("cpu"), batch_size=8)
    assert len(keys) == len(ds) == emb.shape[0]
    assert emb.shape[1] == 4  # state_size
    assert np.isfinite(emb).all()
    from training.signal_data import sample_meta
    expected_keys = [(tk, d) for tk, d, _ in sample_meta(ds)]
    assert keys == expected_keys


def test_extract_embeddings_no_grad_and_deterministic_in_eval_mode():
    frames = _toy_frames()
    ds = TickerDayDataset(frames, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60, align="today")
    model = _toy_model()
    model.eval()
    _, emb1 = extract_embeddings(model, ds, torch.device("cpu"))
    _, emb2 = extract_embeddings(model, ds, torch.device("cpu"))
    np.testing.assert_allclose(emb1, emb2)
    assert all(not p.requires_grad or p.grad is None for p in model.parameters())
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_tft_embeddings.py -q`
Expected: FAIL (`training.tft_embeddings` 모듈 없음)

- [ ] **Step 3: 최소 구현 작성**

```python
# training/tft_embeddings.py
"""forward hook으로 TFT의 output_layer 직전 32차원(=state_size) 은닉 표현을 뽑는다.
tft-torch 벤더 코드(tft-torch/tft_torch/tft.py)는 수정하지 않는다 -- gated_poswise_ff를 만드는
model.pos_wise_ff_gating 모듈의 forward 출력을 hook으로 가로챈다(tft.py:983-987 참고)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.signal_data import sample_meta
from training.train import _to_device_batch


def extract_embeddings(model, dataset, device, batch_size: int = 256) -> tuple[list[tuple[str, str]], np.ndarray]:
    keys = [(tk, d) for tk, d, _ in sample_meta(dataset)]
    model = model.to(device)
    model.eval()
    captured: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.pos_wise_ff_gating.register_forward_hook(_hook)
    chunks = []
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = dict(batch)
                batch.pop("label", None)
                batch = _to_device_batch(batch, device)
                captured.clear()
                model(batch)
                assert len(captured) == 1, "expected exactly one pos_wise_ff_gating call per forward"
                emb = captured[0].squeeze(1)  # [batch, output_seq_len=1, state_size] -> [batch, state_size]
                chunks.append(emb.cpu().numpy())
    finally:
        handle.remove()
    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
    return keys, embeddings
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_tft_embeddings.py -q -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add training/tft_embeddings.py training/test_tft_embeddings.py
git commit -m "feat(gbm-meta): TFT embedding extractor via forward hook (no vendor edits)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: 임베딩-표 피처 조인 + 커버리지 검증

**Files:**
- Create: `training/gbm_meta_data.py`
- Test: `training/test_gbm_meta_data.py`

**Interfaces:**
- Consumes: 없음(순수 함수, pandas/numpy만).
- Produces:
  - `join_embeddings_with_tabular(keys: list[tuple[str,str]], emb: np.ndarray, tab_df: pd.DataFrame, ticker_col="ticker", date_col="date_s") -> pd.DataFrame`
  - `assert_join_coverage(n_joined: int, n_emb: int, n_tab: int, min_ratio: float, label: str) -> None`
  Task 3이 둘 다 그대로 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_gbm_meta_data.py
import numpy as np
import pandas as pd
import pytest

from training.gbm_meta_data import assert_join_coverage, join_embeddings_with_tabular


def test_join_keeps_only_matching_keys_and_adds_emb_columns():
    keys = [("A", "2024-01-02"), ("B", "2024-01-02"), ("A", "2024-01-03")]
    emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    tab = pd.DataFrame({
        "ticker": ["A", "A", "C"], "date_s": ["2024-01-02", "2024-01-03", "2024-01-02"],
        "label": [0, 1, 2],
    })
    out = join_embeddings_with_tabular(keys, emb, tab)
    assert len(out) == 2  # only (A, 01-02) and (A, 01-03) match
    assert set(out.columns) >= {"emb_0", "emb_1", "label", "ticker", "date_s"}
    row = out[(out.ticker == "A") & (out.date_s == "2024-01-02")].iloc[0]
    assert row["emb_0"] == 1.0 and row["emb_1"] == 2.0 and row["label"] == 0


def test_join_raises_on_duplicate_embedding_keys():
    keys = [("A", "2024-01-02"), ("A", "2024-01-02")]
    emb = np.zeros((2, 1))
    tab = pd.DataFrame({"ticker": ["A"], "date_s": ["2024-01-02"], "label": [0]})
    with pytest.raises(ValueError, match="duplicate"):
        join_embeddings_with_tabular(keys, emb, tab)


def test_join_raises_on_duplicate_tabular_keys():
    keys = [("A", "2024-01-02")]
    emb = np.zeros((1, 1))
    tab = pd.DataFrame({"ticker": ["A", "A"], "date_s": ["2024-01-02", "2024-01-02"], "label": [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        join_embeddings_with_tabular(keys, emb, tab)


def test_assert_join_coverage_passes_when_ratio_met():
    assert_join_coverage(n_joined=95, n_emb=100, n_tab=200, min_ratio=0.9, label="train")


def test_assert_join_coverage_raises_when_below_ratio():
    with pytest.raises(ValueError, match="coverage"):
        assert_join_coverage(n_joined=50, n_emb=100, n_tab=200, min_ratio=0.9, label="train")
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_gbm_meta_data.py -q`
Expected: FAIL (모듈 없음)

- [ ] **Step 3: 최소 구현 작성**

```python
# training/gbm_meta_data.py
"""임베딩 배열 + (ticker,date) 키를 표 피처 DataFrame과 내부조인. 커버리지가 기준 미달이면
예외 -- 조용한 드롭 금지(spec §에러 처리)."""
import numpy as np
import pandas as pd


def join_embeddings_with_tabular(keys: list[tuple[str, str]], emb: np.ndarray, tab_df: pd.DataFrame,
                                 ticker_col: str = "ticker", date_col: str = "date_s") -> pd.DataFrame:
    if len(set(keys)) != len(keys):
        raise ValueError(f"join_embeddings_with_tabular: duplicate embedding keys ({len(keys)} keys, "
                         f"{len(set(keys))} unique)")
    tab_keys = list(zip(tab_df[ticker_col], tab_df[date_col]))
    if len(set(tab_keys)) != len(tab_keys):
        raise ValueError(f"join_embeddings_with_tabular: duplicate tabular keys ({len(tab_keys)} rows, "
                         f"{len(set(tab_keys))} unique)")
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df[ticker_col] = [k[0] for k in keys]
    emb_df[date_col] = [k[1] for k in keys]
    return emb_df.merge(tab_df, on=[ticker_col, date_col], how="inner")


def assert_join_coverage(n_joined: int, n_emb: int, n_tab: int, min_ratio: float, label: str) -> None:
    baseline = min(n_emb, n_tab)
    ratio = (n_joined / baseline) if baseline else 1.0
    if ratio < min_ratio:
        raise ValueError(
            f"[{label}] join coverage {ratio:.3f} below min_ratio={min_ratio} "
            f"(joined={n_joined}, n_emb={n_emb}, n_tab={n_tab})")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_gbm_meta_data.py -q -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add training/gbm_meta_data.py training/test_gbm_meta_data.py
git commit -m "feat(gbm-meta): join TFT embeddings with F3 tabular features + coverage guard

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: 데이터 준비 (`prepare_gbm_meta_data`)

**Files:**
- Create: `training/run_gbm_meta.py` (이 태스크에서는 데이터 준비 부분만)
- Test: `training/test_run_gbm_meta.py`

**Interfaces:**
- Consumes: `training.run_tfx_experiments.{RealData, SPLITS, EVAL_WINDOWS, ENCODER_LEN, DEFAULT_ALIGN,
  prepare_recipe, build_window_dataset}`(이미 존재), `training.run_tabular_baseline.{load_frame,
  build_features, aligned_sample_sets}`(이미 존재), `training.tft_embeddings.extract_embeddings`
  (Task 1), `training.gbm_meta_data.{join_embeddings_with_tabular, assert_join_coverage}`(Task 2).
- Produces: `prepare_gbm_meta_data(dsn, champion, checkpoint_path, device, data=None, tabular_df=None) -> dict`
  — 커버리지 최소 비율은 모듈 상수 `MIN_JOIN_RATIO`(train 0.95, val/oot 1.0)로 고정, 인자로
  받지 않는다. `{"train": DataFrame, "val_2024": DataFrame, "oot_2026": DataFrame}`, 각 DataFrame은
  `emb_0..emb_{state_size-1}` + F3 표 피처 열 + `label`, `next_day_return`, `ticker`, `date_s`를
  가진다. `data`/`tabular_df` 인자는 테스트에서 실 DB 없이 주입하기 위한 것
  (`run_tfx_experiments.RealData`와 같은 injectable 패턴).

- [ ] **Step 1: 실패하는 테스트 작성**

`training.test_run_tfx_experiments`의 `FakeData`를 재사용해 TFT 쪽 프레임을 만들고, 표 피처는
합성 DataFrame을 직접 만들어 `tabular_df`로 주입한다(`load_frame`의 DB 호출을 피함).

```python
# training/test_run_gbm_meta.py
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
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_gbm_meta.py -q`
Expected: FAIL (`training.run_gbm_meta` 모듈 없음)

- [ ] **Step 3: 최소 구현 작성**

```python
# training/run_gbm_meta.py (파일 상단부 -- Task 4에서 main()을 이어서 추가)
"""TFT 임베딩(R0 체크포인트, forward hook) + F3 표 피처를 합쳐 GBM 메타 모델을 학습/평가한다.
spec: docs/superpowers/specs/2026-09-27-tft-embedding-gbm-design.md
plan: docs/superpowers/plans/2026-09-27-tft-embedding-gbm-meta.md

    set -a && source .env && set +a
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python training/run_gbm_meta.py
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from training.gbm_meta_data import assert_join_coverage, join_embeddings_with_tabular
from training.tft_embeddings import extract_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "checkpoints", "tfx-aligned.pt")
RESULT_JSON = os.path.join(ARTIFACTS_DIR, "gbm_meta_results.json")
DOC_PATH = "docs/gbm_meta_results.md"
FIT_END, SELECT_START, SELECT_END, TRAIN_END = "2022-12-31", "2023-01-01", "2023-12-31", "2023-12-31"
MIN_JOIN_RATIO = {"train": 0.95, "val_2024": 1.0, "oot_2026": 1.0}


def _load_model(champion: dict, checkpoint_path: str, device):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
    from training.run_stage1_search import STATIC_CARDINALITIES
    from tft_torch.tft import TemporalFusionTransformer
    tft_config = build_tft_config(
        {"historical": champion["columns"], "future": KNOWN_FUTURE_COLS,
         "static_cardinalities": STATIC_CARDINALITIES},
        num_classes=3, state_size=champion["state_size"], attention_heads=champion["attention_heads"],
        lstm_layers=champion["lstm_layers"], dropout=champion["dropout"])
    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _embed_split(name: str, model, frames: dict, cols: list[str], device, align: str = "today"):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from training.dataset import TickerDayDataset
    ds = TickerDayDataset(frames, cols, KNOWN_FUTURE_COLS, STATIC_COLS, 60, align=align)
    keys, emb = extract_embeddings(model, ds, device)
    logger.info("[%s] embedded %d samples, dim=%d", name, len(keys), emb.shape[1] if emb.size else 0)
    return keys, emb


def prepare_gbm_meta_data(dsn: str | None, champion: dict, checkpoint_path: str, device,
                          data=None, tabular_df: pd.DataFrame | None = None) -> dict:
    """R0 임베딩(train/val_2024/oot_2026) + F3 표 피처를 (ticker, date) 조인. `data`/`tabular_df`는
    테스트 주입용(RealData/load_frame을 대체)."""
    from training.run_tfx_experiments import DEFAULT_ALIGN, EVAL_WINDOWS, RealData, build_window_dataset, prepare_recipe

    model = _load_model(champion, checkpoint_path, device)
    data = data or RealData(dsn)
    opts = {"champion": champion, "align": DEFAULT_ALIGN}
    prep = prepare_recipe("aligned", data, opts)  # R0: raw 33 cols, no preprocessing, fixed label
    cols = prep["kept_columns"]

    if tabular_df is None:
        from training.run_tabular_baseline import build_features, load_frame
        raw = load_frame(dsn, champion["columns"])
        tabular_df, _ = build_features(raw, champion["columns"])
        tabular_df["date_s"] = tabular_df["trade_date"].dt.strftime("%Y-%m-%d")

    out = {}
    # --- train ---
    keys, emb = _embed_split("train", model, prep["train_frames"], cols, device)
    joined = join_embeddings_with_tabular(keys, emb, tabular_df)
    joined = joined[joined["date_s"] <= TRAIN_END]
    assert_join_coverage(len(joined), len(keys), tabular_df["date_s"].le(TRAIN_END).sum(),
                         MIN_JOIN_RATIO["train"], "train")
    out["train"] = joined
    # --- val/oot: exact keys from aligned_sample_sets (already asserted == TFT runner's own list) ---
    for w in EVAL_WINDOWS:
        ds, meta = build_window_dataset(w, prep["eval_frames"][w], cols, opts)
        eval_keys, emb = extract_embeddings(model, ds, device)
        joined = join_embeddings_with_tabular(eval_keys, emb, tabular_df)
        assert_join_coverage(len(joined), len(eval_keys), len(tabular_df), MIN_JOIN_RATIO[w], w)
        out[w] = joined
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_gbm_meta.py -q -v`
Expected: PASS. 실패하면 `FakeData`가 만드는 (ticker,date) 조합과 `_fake_tabular_df`가 실제로
일치하는지, `prepare_recipe("aligned", ...)`가 `FakeData.frames`를 어떤 `label_col`로 호출하는지
로그(`data.calls`)로 확인.

- [ ] **Step 5: 커밋**

```bash
git add training/run_gbm_meta.py training/test_run_gbm_meta.py
git commit -m "feat(gbm-meta): prepare_gbm_meta_data joins R0 embeddings with F3 features

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: GBM 학습/선택/채점 + 결과 문서 (`main`)

**Files:**
- Modify: `training/run_gbm_meta.py` (Task 3 파일에 이어서 작성)
- Test: `training/test_run_gbm_meta.py` (Task 3 파일에 이어서 작성)

**Interfaces:**
- Consumes: Task 3의 `prepare_gbm_meta_data`; `training.run_tabular_baseline.{HGB_PARAMS,
  select_hgb_iters, bundle, mean_ic}`(이미 존재, import); `training.run_stage1_search.
  append_model_version_row`(이미 존재, `record_model_version`처럼 재사용).
- Produces: `fit_and_score(prepared: dict) -> dict`(결과 레코드), `main(dsn=None, ...) -> int`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_run_gbm_meta.py 이어서 추가
from training.run_gbm_meta import fit_and_score


def _synthetic_prepared():
    """emb_0이 next_day_return과 상관되도록 만들어 IC가 유의미하게 나오는 합성 데이터."""
    rng = np.random.default_rng(0)
    def _mk(n, start):
        dates = pd.bdate_range(start, periods=n // 5).repeat(5)[:n]
        ret = rng.standard_normal(n) * 0.01
        emb0 = ret * 5 + rng.standard_normal(n) * 0.001  # 신호
        label = np.where(ret > 0.005, 0, np.where(ret < -0.005, 2, 1))
        df = pd.DataFrame({"date_s": [d.strftime("%Y-%m-%d") for d in dates],
                           "ticker": [f"T{i % 20}" for i in range(n)],
                           "emb_0": emb0, "emb_1": rng.standard_normal(n),
                           "f1": rng.standard_normal(n), "label": label,
                           "next_day_return": ret})
        return df
    return {"train": _mk(2000, "2019-01-02"), "val_2024": _mk(400, "2024-01-02"),
            "oot_2026": _mk(400, "2026-01-02")}


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
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_gbm_meta.py -q -k fit_and_score`
Expected: FAIL (`fit_and_score` 없음)

- [ ] **Step 3: 최소 구현 작성**

```python
# training/run_gbm_meta.py 이어서 추가
EXCLUDE_COLS = {"ticker", "date_s", "label", "label_vn", "next_day_return", "trade_date"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _staged_score(model, X):
    for p in model.staged_predict_proba(X):   # classes_는 fit 직후 [0,1,2]로 assert됨
        yield p[:, 0] - p[:, 2]


def fit_and_score(prepared: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from evaluation.evaluate import compute_metrics
    from training.run_tabular_baseline import HGB_PARAMS, bundle, mean_ic, select_hgb_iters
    from training.tabular_features import argmax_nan_safe

    train = prepared["train"]
    cols = _feature_cols(train)
    fit_df = train[train["date_s"] <= FIT_END]
    sel_df = train[(train["date_s"] >= SELECT_START) & (train["date_s"] <= SELECT_END)]
    full_df = train[train["date_s"] <= TRAIN_END]
    assert fit_df["date_s"].max() <= FIT_END and full_df["date_s"].max() <= TRAIN_END
    for df in (fit_df, sel_df, full_df):
        assert not ((df["date_s"] >= "2025-01-01") & (df["date_s"] <= "2025-12-31")).any()

    # Checked BEFORE fitting: select_hgb_iters's staged_predict_proba scoring (`_staged_score`)
    # indexes columns 0 and 2 of predict_proba's output, which silently means something else (or
    # raises a confusing IndexError) if a class is entirely absent from the fit rows -- catch it
    # here with a clear message instead.
    for name, df in (("fit", fit_df), ("full", full_df)):
        present = sorted(df["label"].dropna().astype(int).unique().tolist())
        if present != [0, 1, 2]:
            raise ValueError(f"fit_and_score: expected classes_ == [0,1,2] in {name} rows, got {present}")

    mk = lambda n: HistGradientBoostingClassifier(max_iter=n, **HGB_PARAMS)  # noqa: E731
    best_n, grid = select_hgb_iters(
        mk, fit_df[cols].values, fit_df["label"].astype(int).values,
        sel_df[cols].values, sel_df["date_s"].values, sel_df["next_day_return"].values, _staged_score)
    model = mk(best_n).fit(full_df[cols].values, full_df["label"].astype(int).values)
    assert list(model.classes_) == [0, 1, 2]  # guaranteed by the check above; belt-and-braces

    rec = {"chosen_n_iter": best_n, "select_ic_2023_by_iter": {str(k): v for k, v in grid.items()},
          "n_features": len(cols), "feature_cols": cols}
    for w in ("val_2024", "oot_2026"):
        df = prepared[w]
        probs = model.predict_proba(df[cols].values)
        score = probs[:, 0] - probs[:, 2]
        b, _ics = bundle(df["date_s"].values, df["next_day_return"].values, score, df["ticker"].values)
        preds = model.predict(df[cols].values)
        b["metrics"] = compute_metrics(df["label"].astype(int).tolist(), preds.astype(int).tolist())
        rec[w] = b
    return rec
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_gbm_meta.py -q -v`
Expected: PASS (전부)

- [ ] **Step 5: 결과 문서/model_versions 기록 + `main()` 작성 (실패 테스트부터)**

```python
# training/test_run_gbm_meta.py 이어서 추가
def test_main_writes_result_json_and_doc(tmp_path, monkeypatch):
    import training.run_gbm_meta as gm
    ckpt = _toy_model_checkpoint(tmp_path)
    out_json, doc_path, mv_path = tmp_path / "res.json", tmp_path / "doc.md", tmp_path / "mv.md"
    mv_path.write_text("# model_versions\n")
    rc = gm.main(dsn=None, champion=CHAMPION, checkpoint_path=ckpt, data=FakeData(),
                tabular_df=_fake_tabular_df(), out_json=str(out_json), doc_path=str(doc_path),
                model_versions_path=str(mv_path))
    assert rc == 0
    result = json.loads(out_json.read_text())
    assert "val_2024" in result and "oot_2026" in result
    assert "gbm-meta" in mv_path.read_text()
    # idempotent: 다시 돌려도 model_versions에 중복 행이 안 생김
    gm.main(dsn=None, champion=CHAMPION, checkpoint_path=ckpt, data=FakeData(),
            tabular_df=_fake_tabular_df(), out_json=str(out_json), doc_path=str(doc_path),
            model_versions_path=str(mv_path))
    assert mv_path.read_text().count("gbm-meta") == 1
```

Run(실패 확인): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_gbm_meta.py -q -k test_main`
Expected: FAIL (`main` 없음/시그니처 다름)

```python
# training/run_gbm_meta.py 이어서 추가
def render_doc(rec: dict, code_commit: str) -> str:
    L = ["# TFT 임베딩 + GBM 메타 모델 (Phase 1, 오프라인)", "",
        f"code_commit={code_commit}, chosen_n_iter={rec['chosen_n_iter']}, n_features={rec['n_features']}", "",
        "| window | IC | IC IR | macro F1 |", "|---|---|---|---|"]
    for w in ("val_2024", "oot_2026"):
        s = rec[w]["signal"]
        L.append(f"| {w} | {s['mean_daily_rank_ic']:.4f} | {s.get('ic_ir') or float('nan'):.3f} | "
                f"{rec[w]['metrics']['macro_f1']:.4f} |")
    L += ["", "선택: HGB n_iter는 train 내부 2023 분할(fit<=2022, select 2023)로만 골랐다 -- "
         "val 2024/OOT 2026은 이 선택에 전혀 쓰이지 않았다.", ""]
    return "\n".join(L)


def record_model_version(rec: dict, path: str) -> None:
    from training import run_stage1_search as rss
    from training.run_tfx_experiments import model_versions_row_exists  # reuse, don't reimplement
    if model_versions_row_exists(path, "gbm-meta"):
        return
    old = rss.MODEL_VERSIONS_PATH
    rss.MODEL_VERSIONS_PATH = path
    try:
        rss.append_model_version_row(
            version="gbm-meta", stage="2단계-GBM메타",
            feature_desc=f"R0 TFT 임베딩(32차원) + F3 표 피처, 합계 {rec['n_features']}열 입력, "
                        f"HistGradientBoostingClassifier(n_iter={rec['chosen_n_iter']})",
            hparams={"n_iter": rec["chosen_n_iter"]}, macro_f1_val=rec["val_2024"]["metrics"]["macro_f1"],
            note=f"val IC={rec['val_2024']['signal']['mean_daily_rank_ic']:.4f}, "
                f"OOT IC={rec['oot_2026']['signal']['mean_daily_rank_ic']:.4f}(확인용). "
                f"상세: docs/gbm_meta_results.md")
    finally:
        rss.MODEL_VERSIONS_PATH = old


def main(dsn=None, champion=None, checkpoint_path=CHECKPOINT_PATH, data=None, tabular_df=None,
        out_json=RESULT_JSON, doc_path=DOC_PATH, model_versions_path="docs/model_versions.md") -> int:
    import subprocess
    champion = champion or json.load(open(CHAMPION_CONFIG_PATH))
    device = torch.device("cpu")
    prepared = prepare_gbm_meta_data(dsn, champion, checkpoint_path, device, data=data, tabular_df=tabular_df)
    rec = fit_and_score(prepared)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    rec["code_commit"] = commit
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    with open(doc_path, "w") as f:
        f.write(render_doc(rec, commit))
    record_model_version(rec, model_versions_path)
    logger.info("wrote %s, %s", out_json, doc_path)
    return 0


if __name__ == "__main__":
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise SystemExit("STOCK_DB_V2_DSN environment variable not set")
    raise SystemExit(main(dsn=dsn))
```

Run(통과 확인): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_gbm_meta.py -q -v`
Expected: PASS 전부

- [ ] **Step 6: 커밋**

```bash
git add training/run_gbm_meta.py training/test_run_gbm_meta.py
git commit -m "feat(gbm-meta): fit/select/score GBM meta-model, write results doc + model_versions row

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: 실제 실행 (CPU, 실 DB) + 결과 확인

**Files:**
- 없음(신규/수정 파일 없음 -- Task 1-4의 산출물을 실제 DB로 1회 실행)

**Interfaces:**
- Consumes: Task 4의 `training/run_gbm_meta.py` CLI(`python -m training.run_gbm_meta` 또는
  `python training/run_gbm_meta.py`).
- Produces: `training/artifacts/gbm_meta_results.json`, `docs/gbm_meta_results.md`,
  `docs/model_versions.md`에 `gbm-meta` 행(모두 gitignore 대상 캐시 제외 커밋).

- [ ] **Step 1: 전체 테스트(Task 1-4) 한 번 더 CPU로 통과 확인**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_tft_embeddings.py training/test_gbm_meta_data.py training/test_run_gbm_meta.py -q`
Expected: 전부 PASS.

- [ ] **Step 2: 실제 실행**

```bash
cd /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
set -a && source .env && set +a
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/run_gbm_meta.py
```

train 임베딩 추출이 234,226 샘플에 대한 forward pass라 CPU에서 수 분~수십 분 걸릴 수 있다(학습이
아니라 추론이므로 GPU 없이도 완주 가능 -- tfx 학습의 에폭당 5.5~6.5분과는 다른 작업). 오래 걸리면
`tmux new-session -d -s gbm-meta '<위 명령> 2>&1 | tee training/artifacts/gbm_meta.log'`로 돌리고
`tail -f training/artifacts/gbm_meta.log`로 직접 완료를 확인할 것(사용자에게 진행률을 먼저
보고하지 말 것 -- `feedback_no_proactive_inference_checks`).

- [ ] **Step 3: 결과를 반전/HGB-단독/TFT-단독과 비교**

```bash
python3 - <<'EOF'
import json
gbm = json.load(open("training/artifacts/gbm_meta_results.json"))
tab = json.load(open("training/artifacts/tabular_baseline_v2.json"))
tfx = json.load(open("training/artifacts/tfx_results.json"))
for w in ("val_2024", "oot_2026"):
    print(w, "gbm-meta IC", gbm[w]["signal"]["mean_daily_rank_ic"],
         "| HGB clf label F3", tab["configs"]["hgb_clf_label_F3"][w]["signal"]["mean_daily_rank_ic"],
         "| reversal", tab["references"][w]["reversal"]["signal"]["mean_daily_rank_ic"],
         "| TFT R0", tfx["aligned"][w]["signal"]["mean_daily_rank_ic"])
EOF
```

이 비교표를 사용자에게 보고한다(수치 그대로, 해석은 SE 대비로 — 기존 사이클의 다른 결과
보고와 같은 형식).

- [ ] **Step 4: 커밋**

```bash
git add docs/gbm_meta_results.md docs/model_versions.md
git commit -m "docs(gbm-meta): record R0 embedding + F3 GBM meta-model results

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

`training/artifacts/gbm_meta_results.json`과 `gbm_meta.joblib`(만든다면)은 기존 관행대로
gitignore 대상이면 커밋하지 않는다 — `training/artifacts/.gitignore` 확인 후 따를 것.

---

## 다음 단계 (이 플랜 범위 밖)

결과가 반전/HGB-단독보다 확실히(≥2 SE) 낫다면 서빙 통합(Phase 2, 새 계획서)을 진행한다. 아니라면
사용자에게 결과를 그대로 보고하고 이 트랙을 접을지, 다른 조합(예: E2 승자 체크포인트로 임베딩
교체)을 시도할지 판단을 구한다 — 스스로 다음 실험을 결정해 바로 진행하지 않는다
(`feedback_scoring_only_on_explicit_request`, `feedback_data_grounded_analysis` 정신에 따라
실측 없이 다음 트랙을 단정하지 말 것).
