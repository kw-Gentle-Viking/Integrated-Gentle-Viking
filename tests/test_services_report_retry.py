from types import SimpleNamespace

import pytest

import app.services_report as sr


class _Err(Exception):
    def __init__(self, code):
        super().__init__(f"code {code}")
        self.code = code


class _FakeModels:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(text="ok")


def _client(models):
    return SimpleNamespace(models=models)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sr.time, "sleep", lambda s: None)


def test_retries_transient_503_then_succeeds():
    models = _FakeModels([_Err(503), _Err(503)])
    resp = sr._call_gemini(_client(models), model="m", contents="c")
    assert resp.text == "ok" and models.calls == 3


def test_gives_up_after_attempts_on_persistent_503(monkeypatch):
    monkeypatch.setenv("GEMINI_RETRY_ATTEMPTS", "2")
    models = _FakeModels([_Err(503), _Err(503), _Err(503)])
    with pytest.raises(_Err):
        sr._call_gemini(_client(models), model="m", contents="c")
    assert models.calls == 2


def test_does_not_retry_client_errors():
    models = _FakeModels([_Err(404)])
    with pytest.raises(_Err):
        sr._call_gemini(_client(models), model="m", contents="c")
    assert models.calls == 1
