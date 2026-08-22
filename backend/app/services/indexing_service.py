from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from threading import Event, RLock
from typing import Any

from app.core.config import settings
from app.core.errors import AppError, DocumentNotFoundError
from app.core.logger import logger
from app.services.metadata_service import (
    create_index_job,
    get_document,
    get_index_job,
    get_latest_index_job,
    list_document_records,
    update_document,
    update_index_job,
    utc_now,
)
from app.services.vector_service import index_document


_EXECUTOR: ThreadPoolExecutor | None = None
_FUTURES: dict[str, Future] = {}
_CANCEL_EVENTS: dict[str, Event] = {}
_LOCK = RLock()
_LEGACY_PROVIDER_ERRORS = {
    "provider_quota_exceeded",
    "openai_rate_limited",
    "openai_connection_error",
    "embedding_provider_error",
}


def submit_index_job(document_id: str, *, force: bool = False) -> tuple[dict[str, Any], bool]:
    if get_document(document_id) is None:
        raise DocumentNotFoundError()

    job, created = create_index_job(document_id)
    if not created:
        return job, False

    cancel_event = Event()
    future = _executor().submit(
        _run_index_job,
        job["job_id"],
        document_id,
        force,
        cancel_event,
    )
    with _LOCK:
        _FUTURES[job["job_id"]] = future
        _CANCEL_EVENTS[job["job_id"]] = cancel_event
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


def retry_legacy_provider_failures() -> list[str]:
    """Queue documents that only failed because the previous paid provider was unavailable."""
    if not settings.AUTO_RETRY_LEGACY_PROVIDER_FAILURES:
        return []
    queued: list[str] = []
    for document in list_document_records():
        if (
            document.get("status") == "failed"
            and document.get("error_code") in _LEGACY_PROVIDER_ERRORS
        ):
            _, created = submit_index_job(str(document["document_id"]), force=False)
            if created:
                queued.append(str(document["document_id"]))
    return queued


def retry_interrupted_jobs() -> list[str]:
    if not settings.AUTO_RETRY_INTERRUPTED_JOBS:
        return []
    queued: list[str] = []
    for document in list_document_records():
        if (
            document.get("status") == "failed"
            and document.get("error_code") == "worker_interrupted"
        ):
            _, created = submit_index_job(str(document["document_id"]), force=False)
            if created:
                queued.append(str(document["document_id"]))
    return queued


def cancel_index_job(job_id: str) -> dict[str, Any]:
    job = get_index_job(job_id)
    if job is None:
        raise AppError("Index job not found", code="index_job_not_found", status_code=404)
    if job["status"] not in {"queued", "processing"}:
        return job

    with _LOCK:
        cancel_event = _CANCEL_EVENTS.get(job_id)
        future = _FUTURES.get(job_id)
    if cancel_event is not None:
        cancel_event.set()
    if future is not None and future.cancel():
        _mark_cancelled(job_id, str(job["document_id"]))
    else:
        update_index_job(job_id, stage="cancelling")
        update_document(str(job["document_id"]), stage="cancelling")
    return get_index_job(job_id) or job


def cancel_document_index(document_id: str) -> dict[str, Any]:
    if get_document(document_id) is None:
        raise DocumentNotFoundError()
    job = get_latest_index_job(document_id)
    if job is None or job["status"] not in {"queued", "processing"}:
        raise AppError(
            "This document has no active indexing job",
            code="index_job_not_active",
            status_code=409,
        )
    return cancel_index_job(str(job["job_id"]))


def index_queue_status() -> dict[str, int]:
    with _LOCK:
        active = sum(not future.done() for future in _FUTURES.values())
        cancelling = sum(event.is_set() for event in _CANCEL_EVENTS.values())
    records = list_document_records()
    return {
        "active": active,
        "queued": sum(record.get("status") == "queued" for record in records),
        "processing": sum(record.get("status") == "processing" for record in records),
        "cancelling": cancelling,
        "max_workers": max(1, settings.MAX_CONCURRENT_INDEX_JOBS),
    }


def shutdown_index_executor() -> None:
    global _EXECUTOR
    with _LOCK:
        executor = _EXECUTOR
        _EXECUTOR = None
        for event in _CANCEL_EVENTS.values():
            event.set()
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


def _run_index_job(
    job_id: str,
    document_id: str,
    force: bool,
    cancel_event: Event,
) -> None:
    if cancel_event.is_set():
        _mark_cancelled(job_id, document_id)
        return
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
        if cancel_event.is_set():
            raise _IndexCancelled()
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
        if cancel_event.is_set():
            raise _IndexCancelled()
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
    except _IndexCancelled:
        _mark_cancelled(job_id, document_id)
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


def _mark_cancelled(job_id: str, document_id: str) -> None:
    completed_at = utc_now()
    update_index_job(
        job_id,
        status="cancelled",
        stage="cancelled",
        progress=0,
        error_code=None,
        error_message=None,
        completed_at=completed_at,
    )
    document = get_document(document_id)
    if document is None:
        return
    has_existing_index = int(document.get("chunk_count") or 0) > 0
    update_document(
        document_id,
        status="ready" if has_existing_index else "uploaded",
        stage="ready" if has_existing_index else "cancelled",
        progress=1.0 if has_existing_index else 0,
        error_code=None,
        error_message=None,
    )


def _forget_future(job_id: str) -> None:
    with _LOCK:
        _FUTURES.pop(job_id, None)
        _CANCEL_EVENTS.pop(job_id, None)


class _IndexCancelled(Exception):
    pass
