from app.core.config import settings
from app.services import reranker_service


class FakeCrossEncoder:
    def predict(self, pairs, batch_size):
        assert batch_size == 4
        return [0.05 if "unrelated" in passage else 0.95 for _, passage in pairs]


def test_cross_encoder_reorders_hybrid_candidates(monkeypatch):
    monkeypatch.setattr(settings, "RERANKER_ENABLED", True)
    monkeypatch.setattr(settings, "RERANKER_WEIGHT", 0.8)
    monkeypatch.setattr(settings, "RERANKER_BATCH_SIZE", 4)
    monkeypatch.setattr(reranker_service, "get_reranker", lambda: FakeCrossEncoder())
    candidates = [
        {"text": "an unrelated passage", "relevance": 0.9, "metadata": {}},
        {"text": "the exact answer passage", "relevance": 0.4, "metadata": {}},
    ]

    results, diagnostics = reranker_service.rerank_candidates(
        "What is the exact answer?",
        candidates,
    )

    assert results[0]["text"] == "the exact answer passage"
    assert results[0]["reranker_score"] == 0.95
    assert results[0]["heuristic_relevance"] == 0.4
    assert diagnostics["ranker"] == "cross-encoder"
    assert diagnostics["reranked_count"] == 2


def test_reranker_failure_preserves_hybrid_order(monkeypatch):
    monkeypatch.setattr(settings, "RERANKER_ENABLED", True)

    def fail_to_load():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reranker_service, "get_reranker", fail_to_load)
    candidates = [
        {"text": "first", "relevance": 0.8, "metadata": {}},
        {"text": "second", "relevance": 0.7, "metadata": {}},
    ]

    results, diagnostics = reranker_service.rerank_candidates("question", candidates)

    assert results == candidates
    assert diagnostics["ranker"] == "heuristic-fallback"
    assert diagnostics["warning"]
