from fastapi.testclient import TestClient

from app.main import app
from app.services.quality_service import (
    answer_quality_metrics,
    quality_summary,
    recent_answer_runs,
    record_answer_run,
)


ANSWER_ID = "a" * 32
SOURCES = [
    {
        "source_id": 1,
        "document_id": "b" * 32,
        "filename": "guide.pdf",
        "page": 3,
        "relevance": 0.9,
        "text_quality": 0.8,
    }
]


def record_sample_answer() -> None:
    record_answer_run(
        answer_id=ANSWER_ID,
        question="What is the launch code?",
        mode="quick",
        document_ids=["b" * 32],
        answer="The launch code is ORCHID-4829 [Source 1].",
        engine="extractive",
        model="extractive-v1",
        sources=SOURCES,
        warnings=[],
        diagnostics={"retrieval_ms": 20.0, "generation_ms": 5.0, "total_ms": 25.0},
    )


def test_quality_metrics_validate_citations_and_claim_coverage():
    metrics = answer_quality_metrics(
        "The launch code is ORCHID-4829 [Source 1]. A second unsupported fact exists.",
        SOURCES,
    )

    assert metrics["citation_count"] == 1
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_coverage"] == 0.5
    assert metrics["retrieval_confidence"] == 0.9


def test_citation_after_sentence_period_stays_attached_to_claim():
    metrics = answer_quality_metrics(
        "Overview without a citation.\n- A grounded bullet ends here. [Source 1]",
        SOURCES,
    )

    assert metrics["citation_coverage"] == 0.5


def test_non_factual_list_leadin_is_not_counted_as_uncited_claim():
    metrics = answer_quality_metrics(
        "The strongest extracted points are:\n- A grounded bullet ends here. [Source 1]",
        SOURCES,
    )

    assert metrics["citation_coverage"] == 1.0


def test_feedback_is_saved_and_summarized():
    record_sample_answer()
    client = TestClient(app)

    response = client.post(
        "/quality/feedback",
        json={
            "answer_id": ANSWER_ID,
            "rating": "not_helpful",
            "reasons": ["wrong_source", "missing_information"],
            "comment": "The cited page does not contain the full explanation.",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["rating"] == "not_helpful"
    summary = quality_summary()
    assert summary["answer_count"] == 1
    assert summary["feedback_count"] == 1
    assert summary["not_helpful_count"] == 1
    assert summary["failure_reasons"]["wrong_source"] == 1
    run = recent_answer_runs(1)[0]
    assert run["answer_id"] == ANSWER_ID
    assert run["feedback"]["rating"] == "not_helpful"


def test_feedback_rejects_unknown_answer():
    client = TestClient(app)
    response = client.post(
        "/quality/feedback",
        json={"answer_id": "f" * 32, "rating": "helpful"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "answer_not_found"
