from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
import re
import sqlite3
from threading import RLock
from typing import Any, Iterator

from app.core.config import settings
from app.core.errors import AppError
from app.core.logger import logger
from app.services.metadata_service import initialize_metadata_store, metadata_db_path


_QUALITY_LOCK = RLock()
_INITIALIZED_DATABASES: set[str] = set()
_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
_CLAIM_PATTERN = re.compile(
    r".+?(?:[.!?](?:\s*\[Source\s+\d+\])*(?=\s|$)|$)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    initialize_metadata_store()
    database_path = metadata_db_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_quality_store() -> None:
    database_key = str(metadata_db_path().resolve())
    with _QUALITY_LOCK:
        if database_key in _INITIALIZED_DATABASES and metadata_db_path().is_file():
            return
        with _connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_runs (
                    answer_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    document_ids_json TEXT NOT NULL DEFAULT '[]',
                    answer TEXT NOT NULL DEFAULT '',
                    engine TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    diagnostics_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'completed',
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_answer_runs_created ON answer_runs(created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_answer_runs_status ON answer_runs(status)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_feedback (
                    answer_id TEXT PRIMARY KEY,
                    rating TEXT NOT NULL CHECK(rating IN ('helpful', 'not_helpful')),
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(answer_id) REFERENCES answer_runs(answer_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON answer_feedback(rating)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_answer_cache_accessed ON answer_cache(last_accessed_at DESC)"
            )
        _INITIALIZED_DATABASES.add(database_key)


def answer_quality_metrics(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Return inexpensive, deterministic checks suitable for every local answer."""
    valid_source_ids = {
        int(source["source_id"])
        for source in sources
        if source.get("source_id") is not None
    }
    citations = [int(value) for value in _CITATION_PATTERN.findall(answer)]
    valid_citations = [value for value in citations if value in valid_source_ids]

    claim_sentences = _claim_units(answer)
    cited_claims = sum(bool(_CITATION_PATTERN.search(sentence)) for sentence in claim_sentences)
    relevances = [
        max(0.0, min(float(source.get("relevance") or 0.0), 1.0))
        for source in sources
    ]
    text_qualities = [
        max(0.0, min(float(source["text_quality"]), 1.0))
        for source in sources
        if source.get("text_quality") is not None
    ]

    citation_validity = len(valid_citations) / len(citations) if citations else 0.0
    citation_coverage = cited_claims / len(claim_sentences) if claim_sentences else 0.0
    retrieval_confidence = (
        sum(relevances[:3]) / min(len(relevances), 3) if relevances else 0.0
    )
    source_quality = sum(text_qualities) / len(text_qualities) if text_qualities else None
    overall = (
        citation_validity * 0.35
        + citation_coverage * 0.35
        + retrieval_confidence * 0.30
    )
    return {
        "citation_count": len(citations),
        "citation_validity": round(citation_validity, 4),
        "citation_coverage": round(citation_coverage, 4),
        "retrieval_confidence": round(retrieval_confidence, 4),
        "source_text_quality": round(source_quality, 4) if source_quality is not None else None,
        "quality_score": round(overall, 4),
    }


def record_answer_run(
    *,
    answer_id: str,
    question: str,
    mode: str,
    document_ids: list[str],
    answer: str,
    engine: str,
    model: str,
    sources: list[dict[str, Any]],
    warnings: list[str],
    diagnostics: dict[str, Any],
    status: str = "completed",
    error_code: str | None = None,
) -> None:
    if not settings.QUALITY_TRACKING_ENABLED:
        return
    try:
        initialize_quality_store()
        now = utc_now()
        enriched_diagnostics = {
            **diagnostics,
            **answer_quality_metrics(answer, sources),
        }
        values = (
            answer_id,
            question,
            mode,
            json.dumps(document_ids, ensure_ascii=False),
            answer,
            engine,
            model,
            json.dumps(sources, ensure_ascii=False),
            json.dumps(list(dict.fromkeys(warnings)), ensure_ascii=False),
            json.dumps(enriched_diagnostics, ensure_ascii=False),
            status,
            error_code,
            now,
            now,
        )
        with _QUALITY_LOCK, _connection() as connection:
            connection.execute(
                """
                INSERT INTO answer_runs (
                    answer_id, question, mode, document_ids_json, answer, engine,
                    model, sources_json, warnings_json, diagnostics_json, status,
                    error_code, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(answer_id) DO UPDATE SET
                    answer = excluded.answer,
                    sources_json = excluded.sources_json,
                    warnings_json = excluded.warnings_json,
                    diagnostics_json = excluded.diagnostics_json,
                    status = excluded.status,
                    error_code = excluded.error_code,
                    completed_at = excluded.completed_at
                """,
                values,
            )
    except Exception:
        # Quality logging must never prevent a user from receiving an answer.
        logger.exception("Could not persist answer quality diagnostics")


def save_feedback(
    answer_id: str,
    rating: str,
    reasons: list[str],
    comment: str,
) -> dict[str, Any]:
    initialize_quality_store()
    now = utc_now()
    with _QUALITY_LOCK, _connection() as connection:
        answer_exists = connection.execute(
            "SELECT 1 FROM answer_runs WHERE answer_id = ?",
            (answer_id,),
        ).fetchone()
        if answer_exists is None:
            raise AppError(
                "This answer is no longer available for feedback",
                code="answer_not_found",
                status_code=404,
            )
        connection.execute(
            """
            INSERT INTO answer_feedback (
                answer_id, rating, reasons_json, comment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(answer_id) DO UPDATE SET
                rating = excluded.rating,
                reasons_json = excluded.reasons_json,
                comment = excluded.comment,
                updated_at = excluded.updated_at
            """,
            (
                answer_id,
                rating,
                json.dumps(reasons, ensure_ascii=False),
                comment,
                now,
                now,
            ),
        )
    return {
        "answer_id": answer_id,
        "rating": rating,
        "reasons": reasons,
        "comment": comment,
        "updated_at": now,
    }


def quality_summary() -> dict[str, Any]:
    initialize_quality_store()
    with _QUALITY_LOCK, _connection() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS answer_count,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                AVG(CAST(json_extract(diagnostics_json, '$.total_ms') AS REAL)) AS average_total_ms,
                AVG(CAST(json_extract(diagnostics_json, '$.quality_score') AS REAL)) AS average_quality_score
            FROM answer_runs
            """
        ).fetchone()
        feedback = connection.execute(
            """
            SELECT
                COUNT(*) AS feedback_count,
                SUM(CASE WHEN rating = 'helpful' THEN 1 ELSE 0 END) AS helpful_count,
                SUM(CASE WHEN rating = 'not_helpful' THEN 1 ELSE 0 END) AS not_helpful_count
            FROM answer_feedback
            """
        ).fetchone()
        reason_rows = connection.execute(
            "SELECT reasons_json FROM answer_feedback WHERE rating = 'not_helpful'"
        ).fetchall()

    answer_count = int(totals["answer_count"] or 0)
    feedback_count = int(feedback["feedback_count"] or 0)
    helpful_count = int(feedback["helpful_count"] or 0)
    reasons: dict[str, int] = {}
    for row in reason_rows:
        for reason in _json_list(row["reasons_json"]):
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "answer_count": answer_count,
        "completed_count": int(totals["completed_count"] or 0),
        "failed_count": int(totals["failed_count"] or 0),
        "feedback_count": feedback_count,
        "feedback_rate": round(feedback_count / answer_count, 4) if answer_count else 0.0,
        "helpful_count": helpful_count,
        "not_helpful_count": int(feedback["not_helpful_count"] or 0),
        "helpful_rate": round(helpful_count / feedback_count, 4) if feedback_count else None,
        "average_total_ms": round(float(totals["average_total_ms"] or 0.0), 1),
        "average_quality_score": round(float(totals["average_quality_score"] or 0.0), 4),
        "failure_reasons": dict(
            sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def recent_answer_runs(limit: int = 20) -> list[dict[str, Any]]:
    initialize_quality_store()
    safe_limit = max(1, min(limit, settings.QUALITY_RECENT_RUN_LIMIT))
    with _QUALITY_LOCK, _connection() as connection:
        rows = connection.execute(
            """
            SELECT
                runs.answer_id, runs.question, runs.mode, runs.answer, runs.engine,
                runs.model, runs.sources_json, runs.diagnostics_json, runs.status,
                runs.error_code, runs.created_at, feedback.rating,
                feedback.reasons_json, feedback.comment
            FROM answer_runs AS runs
            LEFT JOIN answer_feedback AS feedback USING(answer_id)
            ORDER BY runs.created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [
        {
            "answer_id": row["answer_id"],
            "question": row["question"],
            "mode": row["mode"],
            "answer_excerpt": str(row["answer"])[:500],
            "engine": row["engine"],
            "model": row["model"],
            "source_count": len(_json_list(row["sources_json"])),
            "diagnostics": _json_object(row["diagnostics_json"]),
            "status": row["status"],
            "error_code": row["error_code"],
            "created_at": row["created_at"],
            "feedback": (
                {
                    "rating": row["rating"],
                    "reasons": _json_list(row["reasons_json"]),
                    "comment": row["comment"],
                }
                if row["rating"]
                else None
            ),
        }
        for row in rows
    ]


def get_cached_answer(cache_key: str) -> dict[str, Any] | None:
    if not settings.ANSWER_CACHE_ENABLED:
        return None
    initialize_quality_store()
    with _QUALITY_LOCK, _connection() as connection:
        row = connection.execute(
            "SELECT payload_json, created_at FROM answer_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            created_at = datetime.fromisoformat(str(row["created_at"]))
        except ValueError:
            connection.execute("DELETE FROM answer_cache WHERE cache_key = ?", (cache_key,))
            return None
        age_hours = (datetime.now(UTC) - created_at).total_seconds() / 3600
        if age_hours > max(1, settings.ANSWER_CACHE_TTL_HOURS):
            connection.execute("DELETE FROM answer_cache WHERE cache_key = ?", (cache_key,))
            return None
        payload = _json_object(row["payload_json"])
        if not payload:
            connection.execute("DELETE FROM answer_cache WHERE cache_key = ?", (cache_key,))
            return None
        connection.execute(
            """
            UPDATE answer_cache
            SET last_accessed_at = ?, hit_count = hit_count + 1
            WHERE cache_key = ?
            """,
            (utc_now(), cache_key),
        )
    return payload


def store_cached_answer(cache_key: str, payload: dict[str, Any]) -> None:
    if not settings.ANSWER_CACHE_ENABLED:
        return
    initialize_quality_store()
    now = utc_now()
    with _QUALITY_LOCK, _connection() as connection:
        connection.execute(
            """
            INSERT INTO answer_cache (
                cache_key, payload_json, created_at, last_accessed_at, hit_count
            ) VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(cache_key) DO UPDATE SET
                payload_json = excluded.payload_json,
                created_at = excluded.created_at,
                last_accessed_at = excluded.last_accessed_at,
                hit_count = 0
            """,
            (cache_key, json.dumps(payload, ensure_ascii=False), now, now),
        )
        connection.execute(
            """
            DELETE FROM answer_cache
            WHERE cache_key IN (
                SELECT cache_key FROM answer_cache
                ORDER BY last_accessed_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (max(1, settings.ANSWER_CACHE_MAX_ENTRIES),),
        )


def _json_list(value: str | None) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _claim_units(answer: str) -> list[str]:
    claims: list[str] = []
    for line in answer.splitlines():
        normalized = line.strip(" -*\t")
        if not normalized:
            continue
        for match in _CLAIM_PATTERN.finditer(normalized):
            claim = match.group(0).strip()
            if len(claim) >= 25 and not claim.endswith(":"):
                claims.append(claim)
    return claims


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}
