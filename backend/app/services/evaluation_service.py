from __future__ import annotations

from statistics import mean
from typing import Any


def score_evaluation_case(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    answer = str(result.get("answer") or "")
    answer_lower = answer.casefold()
    expected_terms = [
        str(term).strip() for term in case.get("expected_terms") or [] if str(term).strip()
    ]
    matched_terms = [term for term in expected_terms if term.casefold() in answer_lower]
    term_recall = len(matched_terms) / len(expected_terms) if expected_terms else None

    expected_pages = {int(page) for page in case.get("expected_pages") or []}
    source_pages = {
        int(source["page"])
        for source in result.get("sources") or []
        if source.get("page") is not None
    }
    matched_pages = sorted(expected_pages & source_pages)
    page_recall = len(matched_pages) / len(expected_pages) if expected_pages else None

    diagnostics = dict(result.get("diagnostics") or {})
    component_scores = [
        float(value)
        for value in (
            term_recall,
            page_recall,
            diagnostics.get("citation_validity"),
            diagnostics.get("citation_coverage"),
        )
        if value is not None
    ]
    score = mean(component_scores) if component_scores else 0.0
    minimum_score = float(case.get("minimum_score", 0.65))
    return {
        "score": round(score, 4),
        "passed": score >= minimum_score,
        "minimum_score": minimum_score,
        "term_recall": round(term_recall, 4) if term_recall is not None else None,
        "matched_terms": matched_terms,
        "missing_terms": [term for term in expected_terms if term not in matched_terms],
        "page_recall": round(page_recall, 4) if page_recall is not None else None,
        "matched_pages": matched_pages,
        "missing_pages": sorted(expected_pages - source_pages),
        "source_pages": sorted(source_pages),
        "citation_validity": diagnostics.get("citation_validity"),
        "citation_coverage": diagnostics.get("citation_coverage"),
        "retrieval_confidence": diagnostics.get("retrieval_confidence"),
        "total_ms": diagnostics.get("total_ms"),
    }


def summarize_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["metrics"]["score"]) for item in results]
    durations = [
        float(item["metrics"]["total_ms"])
        for item in results
        if item["metrics"].get("total_ms") is not None
    ]
    return {
        "case_count": len(results),
        "passed_count": sum(bool(item["metrics"]["passed"]) for item in results),
        "failed_count": sum(not bool(item["metrics"]["passed"]) for item in results),
        "pass_rate": round(
            sum(bool(item["metrics"]["passed"]) for item in results) / len(results), 4
        )
        if results
        else 0.0,
        "average_score": round(mean(scores), 4) if scores else 0.0,
        "average_total_ms": round(mean(durations), 1) if durations else 0.0,
    }
