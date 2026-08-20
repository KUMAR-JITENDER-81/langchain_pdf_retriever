from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any

from langchain_chroma import Chroma

from app.core.config import settings
from app.core.errors import AppError
from app.services.embedding_service import translate_embedding_error
from app.services.metadata_service import get_document, list_document_records
from app.services.vector_service import current_collection_name, get_vector_store


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
RRF_CONSTANT = 60


def search_documents(
    question: str,
    k: int | None = None,
    document_ids: list[str] | None = None,
    *,
    hybrid: bool = True,
) -> dict[str, object]:
    """Run scoped dense + BM25 retrieval and return reranked, filtered chunks."""
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise AppError("Question cannot be empty", code="empty_question")

    result_limit = k or settings.DEFAULT_RETRIEVAL_K
    if result_limit < 1 or result_limit > settings.MAX_RETRIEVAL_K:
        raise AppError(
            f"Result count must be between 1 and {settings.MAX_RETRIEVAL_K}",
            code="invalid_result_count",
        )

    selected_documents = _select_documents(document_ids or [])
    selected_ids = [str(document["document_id"]) for document in selected_documents]
    metadata_filter = _document_filter(selected_ids)
    candidate_limit = min(
        max(result_limit * settings.RETRIEVAL_CANDIDATE_MULTIPLIER, result_limit),
        100,
    )

    candidates: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    dense_error: AppError | None = None

    try:
        dense_matches = get_vector_store().similarity_search_with_score(
            normalized_question,
            k=candidate_limit,
            filter=metadata_filter,
        )
        for rank, (document, distance) in enumerate(dense_matches, start=1):
            key = _candidate_key(document.metadata, document.page_content)
            candidates[key] = {
                "text": document.page_content,
                "metadata": dict(document.metadata),
                "distance": float(distance),
                "dense_rank": rank,
                "keyword_rank": None,
                "bm25_score": 0.0,
            }
    except Exception as exc:
        dense_error = translate_embedding_error(exc)
        warnings.append(dense_error.message)

    corpus_truncated = False
    if hybrid and settings.HYBRID_SEARCH_ENABLED:
        keyword_candidates, corpus_truncated = _keyword_candidates(
            normalized_question,
            metadata_filter,
            candidate_limit,
        )
        for candidate in keyword_candidates:
            key = _candidate_key(candidate["metadata"], candidate["text"])
            if key in candidates:
                candidates[key].update(
                    keyword_rank=candidate["keyword_rank"],
                    bm25_score=candidate["bm25_score"],
                )
            else:
                candidates[key] = candidate

    if dense_error and not candidates:
        raise dense_error

    reranked = _rerank(normalized_question, list(candidates.values()))
    results = _deduplicate(reranked, result_limit)
    if corpus_truncated:
        warnings.append(
            "Keyword search used a limited corpus; narrow the request to fewer documents"
        )

    return {
        "question": normalized_question,
        "document_ids": selected_ids,
        "result_count": len(results),
        "strategy": (
            "hybrid" if hybrid and settings.HYBRID_SEARCH_ENABLED else "dense"
        ),
        "results": results,
        "warnings": warnings,
    }


def _select_documents(requested_ids: list[str]) -> list[dict[str, Any]]:
    collection = current_collection_name()
    if requested_ids:
        records: list[dict[str, Any]] = []
        for document_id in requested_ids:
            record = get_document(document_id)
            if record is None:
                raise AppError(
                    f"Document {document_id} was not found",
                    code="document_not_found",
                    status_code=404,
                )
            records.append(record)
    else:
        records = [
            record
            for record in list_document_records()
            if record.get("status") == "ready"
            and record.get("vector_collection") == collection
        ]

    if not records:
        raise AppError(
            "No indexed documents are ready. Upload and index a PDF first.",
            code="no_indexed_documents",
            status_code=409,
        )

    not_ready = [
        str(record.get("original_filename") or record["document_id"])
        for record in records
        if record.get("status") != "ready"
    ]
    if not_ready:
        raise AppError(
            f"These documents are not ready: {', '.join(not_ready)}",
            code="documents_not_ready",
            status_code=409,
            retryable=True,
        )

    incompatible = [
        str(record.get("original_filename") or record["document_id"])
        for record in records
        if record.get("vector_collection") != collection
    ]
    if incompatible:
        raise AppError(
            "Re-index these documents with the currently configured embedding model: "
            + ", ".join(incompatible),
            code="document_index_incompatible",
            status_code=409,
        )
    return records


def _document_filter(document_ids: list[str]) -> dict[str, Any]:
    if len(document_ids) == 1:
        return {"document_id": document_ids[0]}
    return {"document_id": {"$in": document_ids}}


def _keyword_candidates(
    question: str,
    metadata_filter: dict[str, Any],
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    store = Chroma(
        collection_name=current_collection_name(),
        persist_directory=str(Path(settings.CHROMA_DIR)),
        embedding_function=None,
    )
    maximum = max(settings.HYBRID_MAX_CORPUS_CHUNKS, limit)
    payload = store.get(
        where=metadata_filter,
        limit=maximum + 1,
        include=["documents", "metadatas"],
    )
    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    ids = list(payload.get("ids") or [])
    truncated = len(documents) > maximum
    documents = documents[:maximum]
    metadatas = metadatas[:maximum]
    ids = ids[:maximum]

    scores = _bm25_scores(question, documents)
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )
    candidates: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices[:limit], start=1):
        if scores[index] <= 0:
            continue
        metadata = dict(metadatas[index] or {})
        metadata.setdefault("chunk_id", ids[index])
        candidates.append(
            {
                "text": documents[index],
                "metadata": metadata,
                "distance": None,
                "dense_rank": None,
                "keyword_rank": rank,
                "bm25_score": float(scores[index]),
            }
        )
    return candidates, truncated


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    tokenized_documents = [_tokens(document) for document in documents]
    query_tokens = _tokens(query)
    if not query_tokens:
        return [0.0] * len(documents)

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))
    average_length = sum(len(tokens) for tokens in tokenized_documents) / max(
        len(tokenized_documents), 1
    )
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            containing = document_frequency[token]
            inverse_frequency = math.log(
                1 + (len(documents) - containing + 0.5) / (containing + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(average_length, 1)
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _rerank(question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_tokens = set(_tokens(question))
    max_bm25 = max((float(item["bm25_score"]) for item in candidates), default=0.0)
    lowered_question = question.casefold()

    reranked: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate["text"])
        text_tokens = set(_tokens(text))
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        distance = candidate["distance"]
        dense_relevance = 0.0 if distance is None else 1 / (1 + max(float(distance), 0))
        keyword_relevance = (
            float(candidate["bm25_score"]) / max_bm25 if max_bm25 > 0 else 0.0
        )
        phrase_match = 1.0 if lowered_question in text.casefold() else 0.0
        dense_rrf = (
            1 / (RRF_CONSTANT + int(candidate["dense_rank"]))
            if candidate["dense_rank"]
            else 0.0
        )
        keyword_rrf = (
            1 / (RRF_CONSTANT + int(candidate["keyword_rank"]))
            if candidate["keyword_rank"]
            else 0.0
        )
        rrf = (dense_rrf + keyword_rrf) * RRF_CONSTANT / 2
        relevance = (
            dense_relevance * 0.48
            + keyword_relevance * 0.27
            + overlap * 0.15
            + phrase_match * 0.05
            + rrf * 0.05
        )

        if (
            distance is not None
            and float(distance) > settings.MAX_VECTOR_DISTANCE
            and keyword_relevance < 0.25
            and not phrase_match
        ):
            continue
        if relevance < settings.MIN_RETRIEVAL_RELEVANCE:
            continue

        metadata = dict(candidate["metadata"])
        if metadata.get("ocr_confidence") == -1.0:
            metadata["ocr_confidence"] = None
        reranked.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
                "relevance": round(relevance, 6),
                "dense_rank": candidate["dense_rank"],
                "keyword_rank": candidate["keyword_rank"],
            }
        )

    reranked.sort(key=lambda item: item["relevance"], reverse=True)
    return reranked


def _deduplicate(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_token_sets: list[set[str]] = []
    seen_hashes: set[int] = set()
    for candidate in candidates:
        normalized = " ".join(str(candidate["text"]).casefold().split())
        content_hash = hash(normalized)
        if content_hash in seen_hashes:
            continue
        tokens = set(_tokens(normalized))
        if any(_jaccard(tokens, existing) > 0.92 for existing in selected_token_sets):
            continue
        seen_hashes.add(content_hash)
        selected_token_sets.append(tokens)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def _candidate_key(metadata: dict[str, Any], text: str) -> str:
    return str(
        metadata.get("chunk_id")
        or f"{metadata.get('document_id')}:{metadata.get('page')}:{hash(text)}"
    )


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
