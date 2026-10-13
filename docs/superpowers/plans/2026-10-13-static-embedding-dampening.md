# 정적 임베딩 과의존 억제 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TFT의 정적 임베딩(종목/섹터 구분)이 동적 피처를 압도해 일부 종목(207940, 068270)의 예측이
입력 변화에 거의 반응하지 않는 문제를, 정적 경로 전용 드롭아웃 + weight decay로 완화한 재학습 후보를
만들고, 종목별 반응성 지표로 검증한다.

**Architecture:** `tft-torch`(별도 저장소, 포크) 모델에 정적 임베딩 전용 드롭아웃을 추가하고,
`training/train.py`의 optimizer를 정적/비정적 파라미터 그룹으로 나눠 정적 쪽에만 weight decay를
건다. `training/run_e4_experiments.py`에 좁은 스윕(3개 조합) 레시피를 추가해 현재 챔피언
레시피(`v3_structure_nowd_cs`)와 같은 조건에서 이 두 하이퍼파라미터만 바꿔 비교한다. 새 평가지표
(종목별 예측 반응성)는 순수 함수로 구현해 테스트하고, 기존 OOT 평가 결과물에 적용한다.

**Tech Stack:** PyTorch(`tft-torch` 포크), `training/run_e4_experiments.py` 기존 E4 실험 러너, pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-static-embedding-dampening-design.md`

## Global Constraints

- TFT 구조를 유지한다(다른 모델 계열로 전환하지 않음). 정적변수(sector_id/market_id)는 제거하지 않고 유지한다.
- 기본값(하이퍼파라미터 미지정)일 때 기존 챔피언 레시피의 동작이 **정확히 동일**해야 한다(회귀 없음).
- 2026-10-16(금) 중 채택 여부를 판단할 수 있도록, 좁은 스윕(3개 조합)으로 제한한다. 전체 Optuna
  재탐색은 하지 않는다.
- 채택 기준(스펙 §목표/성공 기준과 동일): 검증 2024 타이밍 IC가 현재 챔피언(`+0.0723`) 대비 −0.01
  이내, AND 207940·068270의 반응성(예측 score의 일별 표준편차, 실제 교차단면 순위와의 상관)이
  뚜렷이 개선, AND 현재 서빙 중인 5종목 중 새로 반응성이 죽는 종목이 없을 것.

## Review Focus

- **정적 드롭아웃 기본값이 0이 아니면 기존 챔피언 레시피 재현이 깨진다** — `static_dropout` 미지정 시
  정확히 0.0이 되는지, 추론(eval) 모드에서 드롭아웃이 꺼지는지 테스트로 확인한다(Task 1).
- **정적 weight decay가 동적 파라미터에도 새어 들어가면 전역 weight_decay=0 전제가 깨진다** —
  파라미터 그룹 분리가 이름 프리픽스를 정확히 매칭하는지, 정적/비정적 파라미터 수의 합이 전체
  파라미터 수와 같은지(빠지거나 겹치는 파라미터가 없는지) 테스트한다(Task 3).
- **반응성 지표가 종목당 샘플이 너무 적을 때(예: 거래일 1~2일) 통계적으로 무의미한 값을 조용히
  낼 수 있다** — 최소 샘플 수 미만이면 `None`/제외로 처리하고 그 사실을 결과에 남기는지 테스트한다(Task 5).
- **E4_DROP_COLS 같은 기존 실험용 env 오버라이드와 새 하이퍼파라미터가 동시에 쓰였을 때 서로
  간섭하지 않는지** — 새 레시피가 기존 `recipe_opts()` 오버라이드 메커니즘과 같은 경로를 타므로,
  기존 레시피(정적 드롭아웃 미지정)가 새 코드 경로를 거쳐도 하이퍼파라미터가 하나도 안 바뀌는지
  회귀 테스트한다(Task 4).
- **체크포인트 구조가 바뀌면 기존 챔피언 체크포인트를 새 코드로 못 불러올 수 있다** — 정적
  드롭아웃은 파라미터가 없는 모듈(`nn.Dropout`)이라 `state_dict` 키가 안 늘어나지만, 이걸 실제로
  확인하는 테스트를 추가한다(Task 1).

---

## Task 1: tft-torch — 정적 임베딩 전용 드롭아웃

**Files:**
- Modify: `tft-torch/tft_torch/tft.py:596-656`(`__init__`, `static_selection` 정의 부근),
  `tft-torch/tft_torch/tft.py:914-935`(`forward`)
- Test: `tft-torch/tests/test_static_dropout.py`(신규)

**Interfaces:**
- Consumes: `config.model.static_dropout`(float, OmegaConf DictConfig, 없으면 0.0) — Task 2가 생산.
- Produces: `TemporalFusionTransformer.static_dropout`(`nn.Dropout` 인스턴스), `forward()`가
  `selected_static`을 `get_static_encoders()`에 넘기기 전에 이 드롭아웃을 통과시킴.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tft-torch/tests/test_static_dropout.py
import torch
from omegaconf import OmegaConf
from tft_torch.tft import TemporalFusionTransformer


def _tiny_config(static_dropout=None):
    model_cfg = {"state_size": 4, "attention_heads": 1, "dropout": 0.0, "lstm_layers": 1, "num_classes": 3}
    if static_dropout is not None:
        model_cfg["static_dropout"] = static_dropout
    return OmegaConf.create({
        "task_type": "classification",
        "target_window_start": None,
        "data_props": {
            "num_historical_numeric": 2, "num_historical_categorical": 0, "historical_categorical_cardinalities": [],
            "num_static_numeric": 0, "num_static_categorical": 1, "static_categorical_cardinalities": [3],
            "num_future_numeric": 1, "num_future_categorical": 0, "future_categorical_cardinalities": [],
        },
        "model": model_cfg,
    })


def _tiny_batch():
    return {
        "static_feats_categorical": torch.zeros((2, 1), dtype=torch.long),
        "historical_ts_numeric": torch.randn(2, 5, 2),
        "future_ts_numeric": torch.randn(2, 1, 1),
    }


def test_static_dropout_defaults_to_zero_when_unset():
    model = TemporalFusionTransformer(_tiny_config(static_dropout=None))
    assert model.static_dropout.p == 0.0


def test_static_dropout_reads_configured_value():
    model = TemporalFusionTransformer(_tiny_config(static_dropout=0.3))
    assert model.static_dropout.p == 0.3


def test_static_dropout_is_off_in_eval_mode_output_is_deterministic():
    """eval() 모드에서는 드롭아웃이 꺼져야 추론 결과가 매번 동일하다(드롭아웃이 forward 경로에
    실제로 연결돼 있는지 확인하는 용도 -- 연결 안 돼 있으면 이 테스트는 애초에 드롭아웃 값과
    무관하게 항상 통과하므로, 아래 train-mode 테스트와 쌍으로 봐야 의미가 있다)."""
    model = TemporalFusionTransformer(_tiny_config(static_dropout=0.9))
    model.eval()
    batch = _tiny_batch()
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    key = next(iter(out1))
    assert torch.allclose(out1[key], out2[key])


def test_static_dropout_is_connected_in_forward_train_mode_varies_output():
    """train() 모드 + 큰 드롭아웃 확률이면, 같은 입력이라도 결과가 달라져야 한다(연결 안 돼 있으면
    항상 같은 값이 나와 이 테스트가 실패한다 -- 즉 이 테스트가 '진짜 연결됐는지'를 검증한다)."""
    torch.manual_seed(0)
    model = TemporalFusionTransformer(_tiny_config(static_dropout=0.9))
    model.train()
    batch = _tiny_batch()
    out1 = model(batch)
    out2 = model(batch)
    key = next(iter(out1))
    assert not torch.allclose(out1[key], out2[key])


def test_state_dict_keys_unchanged_for_default_config():
    """static_dropout 미지정(기존 챔피언 레시피와 동일 조건)이면 state_dict 키가 기존과 똑같아야
    한다 -- nn.Dropout은 학습 파라미터가 없으므로 state_dict에 새 키가 생기면 안 된다."""
    before = TemporalFusionTransformer(_tiny_config(static_dropout=None))
    after = TemporalFusionTransformer(_tiny_config(static_dropout=0.0))
    assert set(before.state_dict().keys()) == set(after.state_dict().keys())
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest tft-torch/tests/test_static_dropout.py -v`
Expected: FAIL — `TemporalFusionTransformer`에 `static_dropout` 속성이 없음(`AttributeError`).

- [ ] **Step 3: 최소 구현**

`tft-torch/tft_torch/tft.py`의 `__init__`에서 `self.static_selection = VariableSelectionNetwork(...)` 정의
바로 다음 줄(기존 626-628줄 뒤)에 추가:

```python
        self.static_dropout = nn.Dropout(p=config.model.get("static_dropout", 0.0))
```

`forward()`에서(기존 928줄) `selected_static, static_weights = self.static_selection(static_rep)` 바로
다음 줄에 추가:

```python
        selected_static = self.static_dropout(selected_static)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest tft-torch/tests/test_static_dropout.py -v`
Expected: PASS(5개 전부).

- [ ] **Step 5: tft-torch 저장소에 커밋** (별도 git 저장소 — `AI_Gentle_Viking_RE`가 아니라 `tft-torch` 자체에 커밋한다)

```bash
cd /home/user/AI_Gentle_Viking_RE/tft-torch
git add tft_torch/tft.py tests/test_static_dropout.py
git commit -m "feat: static embedding dropout (config.model.static_dropout, default 0.0)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: training/config.py — static_dropout 플러밍

**Files:**
- Modify: `training/config.py:53-69`(`build_tft_config`)
- Test: `training/test_config.py`(기존 파일에 추가)

**Interfaces:**
- Consumes: 없음(순수 함수 시그니처 확장).
- Produces: `build_tft_config(..., static_dropout: float = 0.0)` — Task 4가 호출부에서 사용.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_config.py 에 추가
def test_build_tft_config_default_static_dropout_is_zero():
    from training.config import build_tft_config
    cfg = build_tft_config({"historical": ["a"], "future": [], "static_cardinalities": [3]})
    assert cfg.model.static_dropout == 0.0


def test_build_tft_config_passes_through_static_dropout():
    from training.config import build_tft_config
    cfg = build_tft_config({"historical": ["a"], "future": [], "static_cardinalities": [3]}, static_dropout=0.25)
    assert cfg.model.static_dropout == 0.25
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_config.py -v -k static_dropout`
Expected: FAIL — `AttributeError` 또는 `ConfigAttributeError`(`static_dropout` 키 없음).

- [ ] **Step 3: 최소 구현**

`training/config.py`의 `build_tft_config` 시그니처와 반환 딕셔너리를 수정:

```python
def build_tft_config(feature_columns: dict, num_classes: int = 3,
                      state_size: int = 32, attention_heads: int = 4,
                      lstm_layers: int = 1, dropout: float = 0.1,
                      static_dropout: float = 0.0) -> DictConfig:
    return OmegaConf.create({
        "task_type": "classification",
        "target_window_start": None,
        "data_props": {
            "num_historical_numeric": len(feature_columns["historical"]),
            "num_historical_categorical": 0, "historical_categorical_cardinalities": [],
            "num_static_numeric": 0, "num_static_categorical": len(feature_columns["static_cardinalities"]),
            "static_categorical_cardinalities": feature_columns["static_cardinalities"],
            "num_future_numeric": len(feature_columns["future"]),
            "num_future_categorical": 0, "future_categorical_cardinalities": [],
        },
        "model": {"state_size": state_size, "attention_heads": attention_heads,
                   "dropout": dropout, "lstm_layers": lstm_layers, "num_classes": num_classes,
                   "static_dropout": static_dropout},
    })
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_config.py -v`
Expected: PASS(전체 — 기존 테스트 포함 회귀 없음).

- [ ] **Step 5: 커밋**

```bash
cd /home/user/AI_Gentle_Viking_RE
git add training/config.py training/test_config.py
git commit -m "feat(training): plumb static_dropout through build_tft_config

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: training/train.py — 정적 파라미터 전용 weight decay

**Files:**
- Modify: `training/train.py:144-146`(`run_training`의 optimizer 생성부)
- Test: `training/test_train.py`(기존 파일에 추가)

**Interfaces:**
- Consumes: `config["static_weight_decay"]`(float, optional, 기본 0.0) — Task 4가 생산.
- Produces: `split_static_params(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]`,
  `build_optimizer(model, lr, weight_decay=0.0, static_weight_decay=0.0) -> torch.optim.Adam`.
  둘 다 Task 5/6에서는 쓰지 않음(train.py 내부 전용). `run_training`은 `build_optimizer`를 호출하기만
  한다 — 기존 `resume_epoch_and_best_loss`가 `run_training`에서 뽑혀 나온 것과 같은 이유(실제
  모델/DataLoader 없이 단위 테스트하기 위함)로 분리한다. `test_train.py`에는 지금 `run_training`의
  엔드투엔드 CPU dry-run 테스트가 없으므로(확인됨), 새로 만들지 않고 이 분리로 대신한다.

정적 모듈 이름 프리픽스(`tft-torch`의 `TemporalFusionTransformer.__init__`에 정의된 속성명과 정확히
일치): `static_transform`, `static_selection`, `static_encoder_selection`, `static_encoder_enrichment`,
`static_encoder_sequential_cell_init`, `static_encoder_sequential_state_init`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_train.py 에 추가
import torch.nn as nn


class _FakeStaticModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


class _FakeTFT(nn.Module):
    """실제 TemporalFusionTransformer를 흉내낸 가짜 모델. 정적 모듈 이름만 실물과 맞으면 되므로
    torch/tft-torch 전체를 띄우지 않고도 split_static_params의 이름 매칭 로직을 빠르게 검증한다."""
    def __init__(self):
        super().__init__()
        self.static_transform = _FakeStaticModule()
        self.static_selection = _FakeStaticModule()
        self.static_encoder_selection = _FakeStaticModule()
        self.static_encoder_enrichment = _FakeStaticModule()
        self.static_encoder_sequential_cell_init = _FakeStaticModule()
        self.static_encoder_sequential_state_init = _FakeStaticModule()
        self.historical_ts_transform = _FakeStaticModule()  # 비정적 -- 이름에 static이 없음


def test_split_static_params_separates_by_module_name():
    from training.train import split_static_params
    model = _FakeTFT()
    static_params, other_params = split_static_params(model)
    assert len(static_params) == 6          # static_* 모듈 6개, 각 1개 파라미터
    assert len(other_params) == 1            # historical_ts_transform 1개


def test_split_static_params_covers_every_parameter_exactly_once():
    from training.train import split_static_params
    model = _FakeTFT()
    static_params, other_params = split_static_params(model)
    total = sum(p.numel() for p in static_params) + sum(p.numel() for p in other_params)
    expected = sum(p.numel() for p in model.parameters())
    assert total == expected


def test_build_optimizer_single_group_when_static_weight_decay_unset():
    """static_weight_decay를 안 주면(기존 챔피언 레시피와 동일 조건) 기존과 똑같이 파라미터 그룹
    1개짜리 Adam이 나와야 한다 -- 회귀 없음."""
    from training.train import build_optimizer
    model = _FakeTFT()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.0, static_weight_decay=0.0)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_build_optimizer_two_groups_when_static_weight_decay_set():
    from training.train import build_optimizer
    model = _FakeTFT()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.0, static_weight_decay=0.05)
    assert len(opt.param_groups) == 2
    static_group = next(g for g in opt.param_groups if g["weight_decay"] == 0.05)
    other_group = next(g for g in opt.param_groups if g["weight_decay"] != 0.05)
    assert len(static_group["params"]) == 6   # static_* 모듈 6개, 각 1개 파라미터
    assert len(other_group["params"]) == 1    # historical_ts_transform
    assert other_group["weight_decay"] == 0.0


def test_build_optimizer_lr_applies_to_both_groups():
    from training.train import build_optimizer
    model = _FakeTFT()
    opt = build_optimizer(model, lr=5e-4, weight_decay=0.0, static_weight_decay=0.05)
    assert all(g["lr"] == 5e-4 for g in opt.param_groups)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_train.py -v -k "static or build_optimizer"`
Expected: FAIL — `split_static_params`/`build_optimizer`가 `training.train`에 없음(`ImportError`).

- [ ] **Step 3: 최소 구현**

`training/train.py`에 `run_training` 함수 위에 추가:

```python
STATIC_MODULE_PREFIXES = (
    "static_transform", "static_selection", "static_encoder_selection",
    "static_encoder_enrichment", "static_encoder_sequential_cell_init",
    "static_encoder_sequential_state_init",
)


def split_static_params(model) -> tuple[list, list]:
    """모델 파라미터를 (정적 임베딩 경로 파라미터, 나머지)로 나눈다. 이름이 STATIC_MODULE_PREFIXES
    중 하나로 시작하는 최상위 서브모듈에 속하면 정적으로 분류한다."""
    static_params, other_params = [], []
    for name, param in model.named_parameters():
        if name.startswith(STATIC_MODULE_PREFIXES):
            static_params.append(param)
        else:
            other_params.append(param)
    return static_params, other_params


def build_optimizer(model, lr: float, weight_decay: float = 0.0,
                    static_weight_decay: float = 0.0) -> "torch.optim.Adam":
    """static_weight_decay가 0이면 기존과 똑같이 파라미터 그룹 1개짜리 Adam(완전히 기존 동작과
    동일). 0이 아니면 정적 임베딩 경로 파라미터만 따로 묶어 그 그룹에만 static_weight_decay를
    건다. resume_epoch_and_best_loss와 같은 이유로 run_training에서 뽑아내 모델/옵티마이저만으로
    단위 테스트 가능하게 한다."""
    if static_weight_decay:
        static_params, other_params = split_static_params(model)
        other_kwargs = {"weight_decay": weight_decay} if weight_decay else {}
        return torch.optim.Adam([
            {"params": static_params, "weight_decay": static_weight_decay},
            {"params": other_params, **other_kwargs},
        ], lr=lr)
    opt_kwargs = {"weight_decay": weight_decay} if weight_decay else {}
    return torch.optim.Adam(model.parameters(), lr=lr, **opt_kwargs)
```

`run_training` 안의 기존 optimizer 생성 코드(144-146줄, `opt_kwargs = ...` / `optimizer = torch.optim.Adam(...)` 두 줄)를 교체:

```python
    optimizer = build_optimizer(model, lr=config["lr"], weight_decay=config.get("weight_decay", 0.0),
                                static_weight_decay=config.get("static_weight_decay", 0.0))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_train.py -v`
Expected: PASS(전체 — 기존 테스트 포함 회귀 없음; `static_weight_decay` 미설정 시 `build_optimizer`가
기존과 완전히 동일한 단일 그룹 Adam을 반환).

- [ ] **Step 5: 커밋**

```bash
cd /home/user/AI_Gentle_Viking_RE
git add training/train.py training/test_train.py
git commit -m "feat(training): split optimizer into static/other param groups for static_weight_decay

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: training/run_e4_experiments.py — 좁은 스윕 레시피 3개

**Files:**
- Modify: `training/run_e4_experiments.py:240-255`(`recipe_opts`), `:181-199`(`RECIPES` 딕셔너리),
  `:623-637`(`build_tft_config`/`run_training` 호출부)
- Test: `training/test_run_e4_experiments.py`(기존 파일에 추가)

**Interfaces:**
- Consumes: Task 2의 `build_tft_config(..., static_dropout=...)`, Task 3의
  `run_training({..., "static_weight_decay": ...})`.
- Produces: 레시피 이름 3개 — `v3_structure_nowd_cs_staticdrop`, `v3_structure_nowd_cs_staticwd`,
  `v3_structure_nowd_cs_staticboth` — `--recipes` 인자로 실행 가능.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_run_e4_experiments.py 에 추가
def test_recipe_opts_passes_static_dropout_and_weight_decay_through():
    from training.run_e4_experiments import recipe_opts
    base = {"hparams": {"state_size": 32, "attention_heads": 8, "lstm_layers": 2, "dropout": 0.1658, "lr": 2.9e-4}}
    ropts = recipe_opts("v3_structure_nowd_cs_staticdrop", base)
    assert ropts["hparams"]["static_dropout"] == 0.3
    assert ropts["hparams"].get("static_weight_decay", 0.0) == 0.0


def test_recipe_opts_defaults_static_hparams_to_zero_for_existing_recipes():
    """기존 챔피언 레시피는 static_dropout/static_weight_decay를 전혀 지정하지 않으므로, 둘 다
    0이어야 한다 -- 이게 깨지면 기존 레시피의 재현성이 깨진다."""
    from training.run_e4_experiments import recipe_opts
    base = {"hparams": {"state_size": 32, "attention_heads": 8, "lstm_layers": 2, "dropout": 0.1658, "lr": 2.9e-4}}
    ropts = recipe_opts("v3_structure_nowd_cs", base)
    assert ropts["hparams"]["static_dropout"] == 0.0
    assert ropts["hparams"]["static_weight_decay"] == 0.0


def test_three_new_static_recipes_exist_with_distinct_combos():
    from training.run_e4_experiments import RECIPES
    combos = {
        "v3_structure_nowd_cs_staticdrop": (0.3, 0.0),
        "v3_structure_nowd_cs_staticwd": (0.0, 1e-3),
        "v3_structure_nowd_cs_staticboth": (0.2, 5e-4),
    }
    for name, (drop, wd) in combos.items():
        assert RECIPES[name]["static_dropout"] == drop
        assert RECIPES[name]["static_weight_decay"] == wd
        assert RECIPES[name]["label_source"] == "cs"  # 현재 챔피언과 라벨/구조는 동일, 이 두 값만 다름


def test_e4_drop_cols_env_does_not_interact_with_static_hparams(monkeypatch):
    """E4_DROP_COLS(입력 열 제외)는 main()에서 champion["columns"]만 건드리고 recipe_opts()의
    static_dropout/static_weight_decay 처리와는 완전히 다른 코드 경로다. 새 레시피가 이 둘을 섞어
    써도 서로 영향이 없는지 확인한다(둘 다 독립적으로 hp에 반영되는지)."""
    import os
    from training.run_e4_experiments import recipe_opts
    monkeypatch.setenv("E4_DROP_COLS", "lev_total_aum,lev_aum_to_mktcap")
    base = {"hparams": {"state_size": 32, "attention_heads": 8, "lstm_layers": 2, "dropout": 0.1658, "lr": 2.9e-4}}
    ropts = recipe_opts("v3_structure_nowd_cs_staticdrop", base)
    # E4_DROP_COLS는 champion["columns"] 필터링(main()의 역할)이지 recipe_opts의 책임이 아니므로,
    # 여기서는 static_dropout이 env 변수와 무관하게 레시피 값 그대로 반영되는지만 본다.
    assert ropts["hparams"]["static_dropout"] == 0.3
    assert os.environ["E4_DROP_COLS"] == "lev_total_aum,lev_aum_to_mktcap"  # recipe_opts가 지우지 않음
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_e4_experiments.py -v -k static`
Expected: FAIL — `KeyError`(레시피 없음) / `recipe_opts`가 `static_dropout` 키를 안 돌려줌.

- [ ] **Step 3: 최소 구현**

`training/run_e4_experiments.py`의 `RECIPES` 딕셔너리에 `v3_structure_nowd_cs` 항목 바로 뒤에 추가:

```python
    "v3_structure_nowd_cs_staticdrop": dict(tag="V3wd0csSD3", label_source="cs", static_const=False, vol_rank=False,
                                            weight_decay=0.0, static_dropout=0.3,
                                            desc="v3_structure_nowd_cs + 정적 임베딩 전용 드롭아웃 0.3 -- "
                                                 "2026-10-13 발견: 207940/068270이 입력 변화에도 출력이 거의 "
                                                 "안 움직임(종목 고정효과가 동적 신호를 압도). static_const=True "
                                                 "(완전 제거)는 과적합이었으니 억제만 시도."),
    "v3_structure_nowd_cs_staticwd": dict(tag="V3wd0csSW3", label_source="cs", static_const=False, vol_rank=False,
                                          weight_decay=0.0, static_weight_decay=1e-3,
                                          desc="v3_structure_nowd_cs + 정적 임베딩 전용 weight_decay 1e-3 "
                                               "(동적 경로는 그대로 0)."),
    "v3_structure_nowd_cs_staticboth": dict(tag="V3wd0csSB", label_source="cs", static_const=False, vol_rank=False,
                                            weight_decay=0.0, static_dropout=0.2, static_weight_decay=5e-4,
                                            desc="v3_structure_nowd_cs + 정적 드롭아웃 0.2 + 정적 weight_decay "
                                                 "5e-4 동시 적용(완화된 조합)."),
```

`recipe_opts()`(240-255줄)를 수정 — `hp = {**base["hparams"], "weight_decay": ...}` 다음 줄에 추가:

```python
    hp["static_dropout"] = spec.get("static_dropout", 0.0)
    hp["static_weight_decay"] = spec.get("static_weight_decay", 0.0)
```

학습 호출부(623-637줄)를 수정:

```python
    tft_config = build_tft_config(
        ...,
        lstm_layers=hp["lstm_layers"], dropout=hp["dropout"], static_dropout=hp["static_dropout"])
    result = run_training({
        ...,
        "epochs": ropts["epochs"], "lr": hp["lr"], "weight_decay": hp.get("weight_decay", 0.0),
        "static_weight_decay": hp.get("static_weight_decay", 0.0),
        ...  # 기존 나머지 키는 그대로 유지
    })
```

(`...`는 기존 코드의 다른 인자들 — 그대로 둔다. `build_tft_config`/`run_training` 호출의 나머지
인자는 수정하지 않는다.)

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_run_e4_experiments.py -v`
Expected: PASS(전체 — 기존 테스트 포함 회귀 없음).

- [ ] **Step 5: 커밋**

```bash
cd /home/user/AI_Gentle_Viking_RE
git add training/run_e4_experiments.py training/test_run_e4_experiments.py
git commit -m "feat(training): add 3 narrow-sweep recipes for static embedding dampening

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: training/responsiveness.py — 종목별 반응성 지표 (순수 함수)

**Files:**
- Create: `training/responsiveness.py`
- Test: `training/test_responsiveness.py`

**Interfaces:**
- Consumes: 없음(순수 함수).
- Produces: `compute_per_ticker_responsiveness(records: list[dict], min_samples: int = 10) -> dict[str, dict]`
  — `records`의 각 원소는 `{"ticker": str, "score": float, "actual_next_day_return": float}`.
  반환값은 `{ticker: {"score_std": float, "rank_corr": float | None, "n": int}}`.
  `rank_corr`은 해당 종목 내에서, `score`와 그날 전체 종목군 대비 `actual_next_day_return`의 교차단면
  백분위 순위 간 스피어만 상관계수다. Task 6이 OOT 실제 데이터로 이 함수를 호출한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/test_responsiveness.py
import math
import pytest


def test_score_std_near_zero_for_constant_score_matches_207940_symptom():
    from training.responsiveness import compute_per_ticker_responsiveness
    records = [{"ticker": "T", "score": 0.36 + i * 0.0001, "pct_rank": 0.1 * (i % 10)} for i in range(20)]
    out = compute_per_ticker_responsiveness(records)
    assert out["T"]["score_std"] < 0.001


def test_rank_corr_is_high_when_score_tracks_actual_rank():
    from training.responsiveness import compute_per_ticker_responsiveness
    # score가 pct_rank와 완전히 같은 순서로 움직이면 상관계수는 1에 가까워야 한다
    records = [{"ticker": "T", "score": i / 19, "pct_rank": i / 19} for i in range(20)]
    out = compute_per_ticker_responsiveness(records)
    assert out["T"]["rank_corr"] > 0.95


def test_rank_corr_is_none_when_score_is_constant():
    """score가 전혀 안 움직이면(207940처럼) 상관계수 자체가 정의되지 않는다 -- 0이나 NaN을
    조용히 내지 않고 None으로 명시한다."""
    from training.responsiveness import compute_per_ticker_responsiveness
    records = [{"ticker": "T", "score": 0.36, "pct_rank": 0.1 * (i % 10)} for i in range(20)]
    out = compute_per_ticker_responsiveness(records)
    assert out["T"]["rank_corr"] is None


def test_ticker_below_min_samples_is_excluded_not_silently_wrong():
    from training.responsiveness import compute_per_ticker_responsiveness
    records = [{"ticker": "T", "score": 0.1, "pct_rank": 0.1}, {"ticker": "T", "score": 0.2, "pct_rank": 0.9}]
    out = compute_per_ticker_responsiveness(records, min_samples=10)
    assert "T" not in out


def test_each_ticker_computed_independently():
    from training.responsiveness import compute_per_ticker_responsiveness
    records = (
        [{"ticker": "A", "score": i / 19, "pct_rank": i / 19} for i in range(20)]
        + [{"ticker": "B", "score": 0.5, "pct_rank": 0.1 * (i % 10)} for i in range(20)]
    )
    out = compute_per_ticker_responsiveness(records)
    assert out["A"]["rank_corr"] > 0.95
    assert out["B"]["rank_corr"] is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_responsiveness.py -v`
Expected: FAIL — `ModuleNotFoundError: training.responsiveness`.

- [ ] **Step 3: 최소 구현**

```python
# training/responsiveness.py
"""종목별로, 예측 score가 날마다 실제로 움직이는지와 실제 교차단면 순위를 따라가는지를 잰다.
2026-10-13: 207940/068270이 입력이 크게 바뀌어도 출력이 거의 고정돼 있던 걸 수치로 잡기 위해 추가.
레코드는 이미 '그날의 교차단면 백분위 순위(pct_rank)'까지 계산해서 넘겨받는다(원시 수익률에서
순위를 매기는 건 이 함수의 책임이 아니다 -- 호출부가 scripts/return_report.py의
forward_returns류 로직으로 미리 구한다)."""
import statistics as st
from collections import defaultdict


def compute_per_ticker_responsiveness(records: list[dict], min_samples: int = 10) -> dict[str, dict]:
    by_ticker = defaultdict(list)
    for r in records:
        by_ticker[r["ticker"]].append(r)

    out = {}
    for ticker, rows in by_ticker.items():
        if len(rows) < min_samples:
            continue
        scores = [r["score"] for r in rows]
        ranks = [r["pct_rank"] for r in rows]
        score_std = st.pstdev(scores)
        rank_corr = _spearman(scores, ranks) if score_std > 1e-9 else None
        out[ticker] = {"score_std": score_std, "rank_corr": rank_corr, "n": len(rows)}
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    ra = _ranks(a)
    rb = _ranks(b)
    n = len(a)
    mean_r = (n - 1) / 2
    cov = sum((x - mean_r) * (y - mean_r) for x, y in zip(ra, rb))
    var_a = sum((x - mean_r) ** 2 for x in ra)
    var_b = sum((y - mean_r) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a * var_b) ** 0.5


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, i in enumerate(order):
        ranks[i] = position
    return ranks
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python -m pytest training/test_responsiveness.py -v`
Expected: PASS(전체 5개).

- [ ] **Step 5: 커밋**

```bash
cd /home/user/AI_Gentle_Viking_RE
git add training/responsiveness.py training/test_responsiveness.py
git commit -m "feat(training): per-ticker responsiveness metric (score variance + rank correlation)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: 스윕 실행 + 비교 스크립트 (수동 실행 단계)

**Files:**
- Create: `scripts/compare_static_dampening.py`

**Interfaces:**
- Consumes: Task 4의 3개 레시피(`training/run_e4_experiments.py --recipes`), Task 5의
  `compute_per_ticker_responsiveness`.
- Produces: 터미널에 찍는 비교표(타이밍 IC + 종목별 반응성, 챔피언 대비). 이 작업의 산출물은
  "결정"이지 커밋 가능한 코드가 아니므로, 이 스크립트 자체 외에는 다음 작업(Task 7, 교체 또는
  접근 2 전환)으로 넘어가는 분기점이다.

- [ ] **Step 1: 3개 후보 학습 실행** (GPU, 각 약 1~1.5시간 — tmux에서 순차 실행)

```bash
cd /home/user/AI_Gentle_Viking_RE
set -a; . ./.env; set +a
export FEATURE_POOL_TABLE=feature_pool_evtfix   # 지금 챔피언과 같은 테이블 기준으로 비교
S=/tmp/claude-1000/-home-user/1de18e8e-f244-4909-8b34-b508a4a09cc1/scratchpad/static_dampening
mkdir -p $S
tmux new-session -d -s static_sweep "cd /home/user/AI_Gentle_Viking_RE && set -a && . ./.env && set +a && \
  export FEATURE_POOL_TABLE=feature_pool_evtfix && \
  /home/user/miniconda3/envs/dl_env/bin/python -m training.run_e4_experiments \
  --recipes v3_structure_nowd_cs_staticdrop v3_structure_nowd_cs_staticwd v3_structure_nowd_cs_staticboth \
  --no-wandb --artifacts-dir training/artifacts/static_dampening \
  --results-path $S/e4_results.json --doc-path $S/e4_experiments.md \
  --model-versions-path $S/model_versions.md --ref-dir training/artifacts --max-minutes 300 \
  2>&1 | tee $S/train.log"
```

- [ ] **Step 2: 학습 완료 확인**

Run: `tail -20 /tmp/claude-1000/-home-user/1de18e8e-f244-4909-8b34-b508a4a09cc1/scratchpad/static_dampening/train.log`
Expected: 3개 레시피 전부 `recorded` 로그, `invocation complete: 3 recorded`.

- [ ] **Step 3: 비교 스크립트 작성**

```python
# scripts/compare_static_dampening.py
"""Task 6: 정적 임베딩 억제 후보 3개를 챔피언과 비교한다. 타이밍 IC는 e4_results.json에서 읽고,
종목별 반응성은 OOT 구간 실제 예측을 다시 계산해 training.responsiveness로 잰다."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from training.responsiveness import compute_per_ticker_responsiveness

RESULTS_PATH = "/tmp/claude-1000/-home-user/1de18e8e-f244-4909-8b34-b508a4a09cc1/scratchpad/static_dampening/e4_results.json"
CANDIDATES = ["v3_structure_nowd_cs_staticdrop", "v3_structure_nowd_cs_staticwd", "v3_structure_nowd_cs_staticboth"]
CHAMPION_VAL_TIMING_IC = 0.0723  # training/champion_config.json 기준 현재 서빙 체크포인트 값
WATCH_TICKERS = ["207940", "068270"]  # 2026-10-13 발견: 반응성이 바닥이던 종목


def main() -> int:
    results = json.load(open(RESULTS_PATH))
    print(f"{'레시피':35s} {'검증 타이밍 IC':>14s} {'챔피언 대비':>10s}")
    for name in CANDIDATES:
        ic = results[name]["val_2024"]["timing_ic"]
        print(f"{name:35s} {ic:14.4f} {ic - CHAMPION_VAL_TIMING_IC:+10.4f}")
    print("\n(종목별 반응성은 해당 체크포인트로 OOT 재추론한 뒤 별도로 확인 -- "
          "이 스크립트는 1차 스크리닝용. 통과한 후보만 Task 7에서 과거 재생 테스트로 반응성까지 본다.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 실행하고 1차 스크리닝**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/dl_env/bin/python scripts/compare_static_dampening.py`
Expected: 3개 후보의 검증 타이밍 IC가 출력됨. `CHAMPION_VAL_TIMING_IC`(0.0723) 대비 −0.01 이내인
후보만 다음 단계(Task 7) 대상으로 남긴다. 전부 탈락하면 접근 2(사후 보정, 스펙 §접근 2)로 전환한다
— 이 계획 밖의 별도 작업이다.

- [ ] **Step 5: 커밋** (스크립트 자체만 — 학습 산출물/로그는 scratchpad에 남기고 레포에 커밋하지 않음)

```bash
cd /home/user/AI_Gentle_Viking_RE
git add scripts/compare_static_dampening.py
git commit -m "feat(scripts): compare static-dampening candidates against champion val timing IC

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## 이 계획 다음 (별도 작업, 이 문서 범위 밖)

- **Task 6에서 통과한 후보가 있으면**: 10/5 모델 교체 때와 같은 절차(과거 거래일 재생 테스트,
  `feature_pool` 테이블 선택, `serving/best_model_state_dict.pt` 교체, `TFT_MODEL_VERSION` 갱신,
  207940/068270을 포함한 실제 과거 급락 구간(10/7→10/8, 10/8→10/12)에서 반응성이 실제로 개선됐는지
  `training.responsiveness`로 최종 확인) — 10/16(금) 장 마감 후~10/18(일) 사이.
- **전부 탈락하면**: 스펙의 접근 2(종목별 사후 보정, 재학습 없음)로 전환 — 새 브레인스토밍/계획 필요.
- **둘 다 10/16까지 어려우면**: 지금 모델(`v3_structure_nowd_cs-seed0-evtfix`) 유지, 11/13 이후로 미룸.
