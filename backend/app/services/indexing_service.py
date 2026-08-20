from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from threading import RLock
from typing import Any

from app.core.config import settings
from app.core.errors import AppError, DocumentNotFoundError
from app.core.logger import logger
from app.services.metadata_service import (
    create_index_job,
    get_document,
    get_index_job,
    get_latest_index_job,
    update_document,
    update_index_job,
    utc_now,
)
from app.services.vector_service import index_document


_EXECUTOR: ThreadPoolExecutor | None = None
_FUTURES: dict[str, Future] = {}
_LOCK = RLock()


def submit_index_job(document_id: str, *, force: bool = False) -> tuple[dict[str, Any], bool]:
    if get_document(document_id) is None:
        raise DocumentNotFoundError()

    job, created = create_index_job(document_id)
    if not created:
        return job, False

    future = _executor().submit(_run_index_job, job["job_id"], document_id, force)
    with _LOCK:
        _FUTURES[job["job_id"]] = future
    future.add_done_callback(lambda _: _forget_future(job["job_id"]))
    return job, True


def wait_for_index_job(job_id: str, timeout: float = 300) -> dict[str, Any]:
    with _LOCK:
        future = _FUTURES.get(job_id)
    if future is not None:
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            return get_index_job(job_id) or {}
    job = get_index_job(job_id)
    if job is None:
        raise AppError("Index job not found", code="index_job_not_found", status_code=404)
    return job


def document_index_status(document_id: str) -> dict[str, Any]:
    document = get_document(document_id)
    if document is None:
        raise DocumentNotFoundError()
    return {
        "document": document,
        "latest_job": get_latest_index_job(document_id),
    }


def shutdown_index_executor() -> None:
    global _EXECUTOR
    with _LOCK:
        executor = _EXECUTOR
        _EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=max(1, settings.MAX_CONCURRENT_INDEX_JOBS),
                thread_name_prefix="pdf-indexer",
            )
        return _EXECUTOR


def _run_index_job(job_id: str, document_id: str, force: bool) -> None:
    started_at = utc_now()
    update_index_job(
        job_id,
        status="processing",
        stage="starting",
        progress=0.01,
        started_at=started_at,
        error_code=None,
        error_message=None,
    )
    update_document(
        document_id,
        status="processing",
        stage="starting",
        progress=0.01,
        error_code=None,
        error_message=None,
    )

    last_progress = {"stage": "starting", "value": 0.01}

    def report_progress(stage: str, progress: float) -> None:
        safe_progress = max(0.01, min(float(progress), 0.99))
        if (
            stage == last_progress["stage"]
            and safe_progress - float(last_progress["value"]) < 0.02
        ):
            return
        last_progress.update(stage=stage, value=safe_progress)
        update_index_job(job_id, stage=stage, progress=safe_progress)
        update_document(document_id, stage=stage, progress=safe_progress)

    try:
        result = index_document(
            document_id,
            force=force,
            progress_callback=report_progress,
        )
        completed_at = utc_now()
        update_index_job(
            job_id,
            status="ready",
            stage="ready",
            progress=1.0,
            completed_at=completed_at,
            error_code=None,
            error_message=None,
        )
        update_document(
            document_id,
            status="ready",
            stage="ready",
            progress=1.0,
            chunk_count=result["chunk_count"],
            embedding_provider=result["embedding_provider"],
            embedding_model=result["embedding_model"],
            vector_collection=result["collection"],
            index_fingerprint=result["index_fingerprint"],
            indexed_at=completed_at,
            error_code=None,
            error_message=None,
        )
    except AppError as exc:
        _mark_failed(job_id, document_id, exc.code, exc.message)
    except Exception:
        logger.exception("Unexpected indexing failure for document %s", document_id)
        _mark_failed(
            job_id,
            document_id,
            "indexing_failed",
            "An unexpected indexing error occurred",
        )


def _mark_failed(job_id: str, document_id: str, code: str, message: str) -> None:
    completed_at = utc_now()
    update_index_job(
        job_id,
        status="failed",
        stage="failed",
        progress=0,
        error_code=code,
        error_message=message,
        completed_at=completed_at,
    )
    try:
        update_document(
            document_id,
            status="failed",
            stage="failed",
            progress=0,
            error_code=code,
            error_message=message,
        )
    except KeyError:
        # The document may have been removed during shutdown.
        pass


def _forget_future(job_id: str) -> None:
    with _LOCK:
        _FUTURES.pop(job_id, None)
