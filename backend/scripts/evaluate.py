"""Run a free, local regression benchmark against indexed PDFs.

Example:
    python scripts/evaluate.py evaluation/cases.json --output evaluation/report.json
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.rag.chain import answer_question  # noqa: E402
from app.services.evaluation_service import (  # noqa: E402
    score_evaluation_case,
    summarize_evaluation,
)
from app.services.metadata_service import (  # noqa: E402
    initialize_metadata_store,
    list_document_records,
)
from app.services.quality_service import initialize_quality_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local PDF answer quality")
    parser.add_argument("dataset", type=Path, help="JSON file containing an array named 'cases'")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--use-cache", action="store_true", help="Allow cached answers")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit with code 1 when the overall pass rate is below this 0-1 value",
    )
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    initialize_metadata_store()
    initialize_quality_store()
    settings.ANSWER_CACHE_ENABLED = bool(args.use_cache)
    documents = list_document_records()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(dataset["cases"], start=1):
        document_ids = _resolve_document_ids(case, documents)
        result = answer_question(
            str(case["question"]),
            k=int(case.get("k", settings.DEFAULT_RETRIEVAL_K)),
            document_ids=document_ids,
            mode=str(case.get("mode", "balanced")),
            task=str(case.get("task", "answer")),
            response_language=str(case.get("response_language", "English")),
            history=[],
        )
        metrics = score_evaluation_case(case, result)
        results.append(
            {
                "id": str(case.get("id") or f"case-{index}"),
                "question": case["question"],
                "document_ids": document_ids,
                "answer": result["answer"],
                "sources": result["sources"],
                "metrics": metrics,
            }
        )
        marker = "PASS" if metrics["passed"] else "FAIL"
        print(f"[{marker}] {results[-1]['id']}: score={metrics['score']:.3f}")

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "cache_enabled": settings.ANSWER_CACHE_ENABLED,
        "summary": summarize_evaluation(results),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    else:
        print(rendered)

    print(
        f"Pass rate: {report['summary']['pass_rate']:.1%} · "
        f"average score: {report['summary']['average_score']:.3f}"
    )
    return 1 if report["summary"]["pass_rate"] < max(0.0, args.fail_under) else 0


def _load_dataset(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read evaluation dataset: {exc}") from exc
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise SystemExit("The dataset must contain a non-empty 'cases' array")
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not str(case.get("question") or "").strip():
            raise SystemExit(f"Evaluation case {index} needs a question")
    return payload


def _resolve_document_ids(
    case: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[str]:
    explicit_ids = [str(value) for value in case.get("document_ids") or []]
    if explicit_ids:
        return explicit_ids
    filenames = [str(value).casefold() for value in case.get("filenames") or []]
    if not filenames:
        raise SystemExit(
            f"Case '{case.get('id', case['question'])}' needs document_ids or filenames"
        )
    resolved = [
        str(document["document_id"])
        for document in documents
        if str(document.get("original_filename") or "").casefold() in filenames
        and document.get("status") == "ready"
    ]
    if len(resolved) != len(set(filenames)):
        raise SystemExit(
            f"Case '{case.get('id', case['question'])}' could not resolve every ready filename"
        )
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
