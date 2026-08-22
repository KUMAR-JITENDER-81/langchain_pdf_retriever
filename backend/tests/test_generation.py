import json

import pytest

from app.core.config import settings
from app.core.errors import ProviderUnavailableError
from app.services import generation_service


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_generation_model_falls_back_to_an_installed_text_model(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_FAST_MODEL", "missing-fast:latest")
    monkeypatch.setattr(settings, "OLLAMA_CHAT_MODEL", "available-deep:latest")

    balanced = generation_service._resolve_installed_generation_model(
        "balanced",
        ["available-deep:latest"],
    )
    deep = generation_service._resolve_installed_generation_model(
        "deep",
        ["missing-fast:latest"],
    )

    assert balanced == "available-deep:latest"
    assert deep == "missing-fast:latest"


def test_warmup_loads_effective_balanced_model(monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_FAST_MODEL", "qwen-test:latest")
    monkeypatch.setattr(settings, "OLLAMA_CHAT_MODEL", "qwen-deep:latest")
    monkeypatch.setattr(
        generation_service,
        "ollama_status",
        lambda force=False: {
            "available": True,
            "installed_models": ["qwen-test:latest"],
        },
    )
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"done": True, "done_reason": "load"})

    monkeypatch.setattr(generation_service, "urlopen", fake_urlopen)

    result = generation_service.warmup_ollama_model("balanced")

    assert result["state"] == "ready"
    assert result["model"] == "qwen-test:latest"
    assert captured["url"].endswith("/api/generate")
    assert captured["payload"]["prompt"] == ""
    assert captured["payload"]["keep_alive"] == settings.OLLAMA_KEEP_ALIVE


def test_answer_modes_have_separate_time_budgets(monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_TIMEOUT_SECONDS", 90.0)
    monkeypatch.setattr(settings, "OLLAMA_BALANCED_TIMEOUT_SECONDS", 25.0)
    monkeypatch.setattr(settings, "OLLAMA_DEEP_TIMEOUT_SECONDS", 70.0)

    assert generation_service.generation_timeout_for_mode("balanced") == 25.0
    assert generation_service.generation_timeout_for_mode("deep") == 70.0


def test_output_budget_expands_for_tasks_with_many_required_items():
    assert generation_service._mode_output_tokens("balanced", 900) == 480
    assert generation_service._mode_output_tokens("deep", 900) == 700
    assert generation_service._mode_output_tokens(
        "balanced",
        900,
        minimum=860,
    ) == 860
    assert generation_service._mode_output_tokens(
        "quick",
        700,
        minimum=900,
    ) == 700
    assert generation_service._mode_output_tokens(
        "deep",
        900,
        minimum=700,
        maximum=228,
    ) == 228


def test_overlapping_generation_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_QUEUE_TIMEOUT_SECONDS", 0.0)
    generation_service._GENERATION_SLOTS.acquire()
    try:
        with pytest.raises(ProviderUnavailableError) as captured:
            with generation_service.ollama_generation_slot():
                pass
    finally:
        generation_service._GENERATION_SLOTS.release()

    assert captured.value.code == "ollama_busy"
