from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterator
from uuid import uuid4

from app.core.config import settings


_DATABASE_LOCK = RLock()
_INITIALIZED_DATABASES: set[str] = set()

DOCUMENT_FIELDS = {
    "document_id",
    "original_filename",
    "stored_filename",
    "sha256",
    "size_bytes",
    "content_type",
    "page_count",
    "status",
    "stage",
    "progress",
    "extraction_method",
    "native_page_count",
    "ocr_page_count",
    "handwritten_page_count",
    "low_quality_page_count",
    "table_count",
    "average_text_quality",
    "extraction_warning_count",
    "character_count",
    "chunk_count",
    "embedding_provider",
    "embedding_model",
    "vector_collection",
    "index_fingerprint",
    "error_code",
    "error_message",
    "created_at",
    "updated_at",
    "indexed_at",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def metadata_db_path() -> Path:
    return Path(settings.METADATA_DB)


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
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


def initialize_metadata_store() -> None:
    database_key = str(metadata_db_path().resolve())
    with _DATABASE_LOCK:
        if database_key in _INITIALIZED_DATABASES and metadata_db_path().is_file():
            return
    with _DATABASE_LOCK, _connection() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                sha256 TEXT NOT NULL DEFAULT '',
                size_bytes INTEGER NOT NULL DEFAULT 0,
                content_type TEXT NOT NULL DEFAULT 'application/pdf',
                page_count INTEGER,
                status TEXT NOT NULL DEFAULT 'uploaded',
                stage TEXT NOT NULL DEFAULT 'uploaded',
                progress REAL NOT NULL DEFAULT 0,
                extraction_method TEXT,
                native_page_count INTEGER NOT NULL DEFAULT 0,
                ocr_page_count INTEGER NOT NULL DEFAULT 0,
                handwritten_page_count INTEGER NOT NULL DEFAULT 0,
                low_quality_page_count INTEGER NOT NULL DEFAULT 0,
                table_count INTEGER NOT NULL DEFAULT 0,
                average_text_quality REAL,
                extraction_warning_count INTEGER NOT NULL DEFAULT 0,
                character_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                embedding_provider TEXT,
                embedding_model TEXT,
                vector_collection TEXT,
                index_fingerprint TEXT,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                indexed_at TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS index_jobs (
                job_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_document ON index_jobs(document_id)"
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_job_per_document
            ON index_jobs(document_id)
            WHERE status IN ('queued', 'processing')
            """
        )
        _ensure_column(connection, "documents", "vector_collection", "TEXT")
        _ensure_column(
            connection, "documents", "handwritten_page_count", "INTEGER NOT NULL DEFAULT 0"
        )
        _ensure_column(
            connection, "documents", "low_quality_page_count", "INTEGER NOT NULL DEFAULT 0"
        )
        _ensure_column(connection, "documents", "table_count", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "documents", "average_text_quality", "REAL")
        _ensure_column(
            connection, "documents", "extraction_warning_count", "INTEGER NOT NULL DEFAULT 0"
        )
        _INITIALIZED_DATABASES.add(database_key)


def create_document(record: dict[str, Any]) -> dict[str, Any]:
    initialize_metadata_store()
    now = utc_now()
    values = {
        "document_id": record["document_id"],
        "original_filename": record["original_filename"],
        "stored_filename": record["stored_filename"],
        "sha256": record.get("sha256", ""),
        "size_bytes": record.get("size_bytes", 0),
        "content_type": record.get("content_type") or "application/pdf",
        "page_count": record.get("page_count"),
        "status": record.get("status", "uploaded"),
        "stage": record.get("stage", "uploaded"),
        "progress": record.get("progress", 0),
        "created_at": record.get("created_at", now),
        "updated_at": record.get("updated_at", now),
    }
    with _DATABASE_LOCK, _connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id, original_filename, stored_filename, sha256,
                size_bytes, content_type, page_count, status, stage, progress,
                created_at, updated_at
            ) VALUES (
                :document_id, :original_filename, :stored_filename, :sha256,
                :size_bytes, :content_type, :page_count, :status, :stage,
                :progress, :created_at, :updated_at
            )
            ON CONFLICT(document_id) DO UPDATE SET
                original_filename = excluded.original_filename,
                stored_filename = excluded.stored_filename,
                sha256 = excluded.sha256,
                size_bytes = excluded.size_bytes,
                content_type = excluded.content_type,
                page_count = COALESCE(excluded.page_count, documents.page_count),
                updated_at = excluded.updated_at
            """,
            values,
        )
    document = get_document(record["document_id"])
    assert document is not None
    return document


def get_document(document_id: str) -> dict[str, Any] | None:
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def find_document_by_hash(sha256: str) -> dict[str, Any] | None:
    if not sha256:
        return None
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE sha256 = ? ORDER BY created_at LIMIT 1",
            (sha256,),
        ).fetchone()
    return dict(row) if row else None


def list_document_records() -> list[dict[str, Any]]:
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def update_document(document_id: str, **changes: Any) -> dict[str, Any]:
    invalid_fields = set(changes) - DOCUMENT_FIELDS
    if invalid_fields:
        raise ValueError(f"Unsupported document fields: {sorted(invalid_fields)}")
    changes.pop("document_id", None)
    changes["updated_at"] = utc_now()
    assignments = ", ".join(f"{field} = :{field}" for field in changes)
    parameters = {**changes, "document_id": document_id}
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        cursor = connection.execute(
            f"UPDATE documents SET {assignments} WHERE document_id = :document_id",
            parameters,
        )
        if cursor.rowcount == 0:
            raise KeyError(document_id)
    document = get_document(document_id)
    assert document is not None
    return document


def delete_document_record(document_id: str) -> None:
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def mark_interrupted_work() -> None:
    """Make jobs left running by a previous process visibly retryable."""
    initialize_metadata_store()
    now = utc_now()
    with _DATABASE_LOCK, _connection() as connection:
        connection.execute(
            """
            UPDATE index_jobs
            SET status = 'failed', stage = 'interrupted', progress = 0,
                error_code = 'worker_interrupted',
                error_message = 'The server stopped before indexing completed',
                completed_at = ?
            WHERE status IN ('queued', 'processing')
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE documents
            SET status = 'failed', stage = 'interrupted', progress = 0,
                error_code = 'worker_interrupted',
                error_message = 'The server stopped before indexing completed',
                updated_at = ?
            WHERE status IN ('queued', 'processing')
            """,
            (now,),
        )


def create_index_job(document_id: str) -> tuple[dict[str, Any], bool]:
    """Create one queued job, returning an existing active job when present."""
    initialize_metadata_store()
    now = utc_now()
    with _DATABASE_LOCK, _connection() as connection:
        existing = connection.execute(
            """
            SELECT * FROM index_jobs
            WHERE document_id = ? AND status IN ('queued', 'processing')
            ORDER BY created_at DESC LIMIT 1
            """,
            (document_id,),
        ).fetchone()
        if existing:
            return dict(existing), False

        job_id = uuid4().hex
        connection.execute(
            """
            INSERT INTO index_jobs (
                job_id, document_id, status, stage, progress, created_at
            ) VALUES (?, ?, 'queued', 'queued', 0, ?)
            """,
            (job_id, document_id, now),
        )
        connection.execute(
            """
            UPDATE documents
            SET status = 'queued', stage = 'queued', progress = 0,
                error_code = NULL, error_message = NULL, updated_at = ?
            WHERE document_id = ?
            """,
            (now, document_id),
        )
        row = connection.execute(
            "SELECT * FROM index_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    assert row is not None
    return dict(row), True


def get_index_job(job_id: str) -> dict[str, Any] | None:
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        row = connection.execute(
            "SELECT * FROM index_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def get_latest_index_job(document_id: str) -> dict[str, Any] | None:
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM index_jobs WHERE document_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def update_index_job(job_id: str, **changes: Any) -> dict[str, Any]:
    allowed_fields = {
        "status",
        "stage",
        "progress",
        "error_code",
        "error_message",
        "started_at",
        "completed_at",
    }
    invalid_fields = set(changes) - allowed_fields
    if invalid_fields:
        raise ValueError(f"Unsupported job fields: {sorted(invalid_fields)}")
    assignments = ", ".join(f"{field} = :{field}" for field in changes)
    parameters = {**changes, "job_id": job_id}
    initialize_metadata_store()
    with _DATABASE_LOCK, _connection() as connection:
        cursor = connection.execute(
            f"UPDATE index_jobs SET {assignments} WHERE job_id = :job_id",
            parameters,
        )
        if cursor.rowcount == 0:
            raise KeyError(job_id)
    job = get_index_job(job_id)
    assert job is not None
    return job


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
