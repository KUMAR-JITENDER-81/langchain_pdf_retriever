from app.services.evaluation_service import score_evaluation_case, summarize_evaluation


def test_evaluation_scores_expected_terms_pages_and_citations():
    case = {
        "expected_terms": ["ORCHID-4829", "Maya"],
        "expected_pages": [2],
        "minimum_score": 0.8,
    }
    result = {
        "answer": "The code is ORCHID-4829 and the owner is Maya [Source 1].",
        "sources": [{"source_id": 1, "page": 2}],
        "diagnostics": {"citation_validity": 1.0, "citation_coverage": 1.0, "total_ms": 50},
    }

    metrics = score_evaluation_case(case, result)

    assert metrics["score"] == 1.0
    assert metrics["passed"] is True
    assert metrics["missing_terms"] == []
    assert metrics["missing_pages"] == []


def test_evaluation_summary_reports_regressions():
    summary = summarize_evaluation(
        [
            {"metrics": {"score": 1.0, "passed": True, "total_ms": 100}},
            {"metrics": {"score": 0.4, "passed": False, "total_ms": 300}},
        ]
    )

    assert summary["case_count"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["average_score"] == 0.7
    assert summary["average_total_ms"] == 200.0
