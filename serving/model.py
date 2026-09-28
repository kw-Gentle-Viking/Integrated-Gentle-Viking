"""
serving/model.py
=================
Task 18 Step 5. Shared model-loading + feature-column plumbing for serving/api_server.py and
serving/inference_pipeline.py, so the two entry points can't silently drift apart on which
columns/architecture they use (both must match training/champion_config.json exactly).

Loads:
    training/champion_config.json  -- architecture + hparams + the 33-column historical feature
                                       list (Task 16's Stage-1 champion)
    serving/best_model_state_dict.pt -- Task 17's Stage-2 final trained weights
"""

import json
import os
import sys
import threading

import torch

from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SERVING_DIR)

# Some conda envs (e.g. kis_collector, used to run uvicorn -- see api_server.py's docstring)
# have an older `tft_torch` pip-installed in site-packages that predates this project's
# classification-task additions (task_type="classification" / class_logits output). Force this
# repo's own tft-torch/tft_torch (the one training/train.py already uses via
# `sys.path.insert(0, "tft-torch")`) to take priority, regardless of which env runs this module.
_TFT_TORCH_PATH = os.path.join(REPO_ROOT, "tft-torch")
if _TFT_TORCH_PATH not in sys.path:
    sys.path.insert(0, _TFT_TORCH_PATH)
CHAMPION_CONFIG_PATH = os.environ.get(
    "CHAMPION_CONFIG_PATH", os.path.join(os.path.dirname(SERVING_DIR), "training", "champion_config.json")
)
MODEL_PATH = os.environ.get("TFT_MODEL_PATH", os.path.join(SERVING_DIR, "best_model_state_dict.pt"))

# sector_id (0-20, incl. SECTOR_ID_UNCLASSIFIED=20), market_id (0-2, incl.
# MARKET_ID_UNCLASSIFIED=2) -- must match training/run_stage1_search.py's STATIC_CARDINALITIES,
# duplicated here rather than imported to avoid pulling in run_stage1_search's heavier
# (Optuna/W&B) import chain into the serving path.
STATIC_CARDINALITIES = [21, 3]

_champion = None
_model = None
_lock = threading.Lock()


def get_model_version() -> str:
    """백엔드로 push 하는 예측에 붙이는 모델 버전. TFT_MODEL_VERSION 이 있으면 그것, 없으면 모델 파일명(확장자 제외)."""
    return os.environ.get("TFT_MODEL_VERSION") or os.path.splitext(os.path.basename(MODEL_PATH))[0]


def load_champion_config(path: str = CHAMPION_CONFIG_PATH) -> dict:
    global _champion
    if _champion is None:
        with open(path) as f:
            _champion = json.load(f)
    return _champion


def get_feature_columns(champion: dict | None = None) -> dict:
    champion = champion or load_champion_config()
    return {
        "historical": champion["columns"],
        "future": KNOWN_FUTURE_COLS,
        "static": STATIC_COLS,
    }


def get_model(model_path: str = MODEL_PATH, champion: dict | None = None, device: str = "cpu"):
    """Process-wide singleton load, mirroring /home/user/api_server.py's get_model() pattern."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from tft_torch.tft import TemporalFusionTransformer

                champion = champion or load_champion_config()
                feature_columns = {
                    "historical": champion["columns"], "future": KNOWN_FUTURE_COLS,
                    "static_cardinalities": STATIC_CARDINALITIES,
                }
                tft_config = build_tft_config(
                    feature_columns, num_classes=3,
                    state_size=champion["state_size"], attention_heads=champion["attention_heads"],
                    lstm_layers=champion["lstm_layers"], dropout=champion["dropout"],
                )
                m = TemporalFusionTransformer(tft_config)
                m.load_state_dict(torch.load(model_path, map_location=device))
                m.eval()
                _model = m
    return _model
