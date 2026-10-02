"""백엔드 /ai/realtime 은 model_version 을 받는다 (백엔드 스키마 기본값 'unknown'). 서빙이 실제 버전을 채워 보낸다."""
from serving import model as sm


def test_model_version_env_overrides(monkeypatch):
    monkeypatch.setenv("TFT_MODEL_VERSION", "v3_wd-seed0")
    assert sm.get_model_version() == "v3_wd-seed0"


def test_model_version_defaults_to_model_file_stem(monkeypatch):
    monkeypatch.delenv("TFT_MODEL_VERSION", raising=False)
    monkeypatch.setattr(sm, "MODEL_PATH", "/x/y/e2e3-v3_wd.pt")
    assert sm.get_model_version() == "e2e3-v3_wd"
