from __future__ import annotations

import math
import os
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logger import logger


_MODEL_LOCK = RLock()
_MODEL: OnnxCrossEncoder | None = None
_MODEL_IDENTITY: tuple[str, str, int] | None = None
_LAST_ERROR: str | None = None


class OnnxCrossEncoder:
    """Small CPU reranker loaded directly with ONNX Runtime and Tokenizers."""

    def __init__(self, model_path: Path, tokenizer_path: Path, max_length: int) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(os.cpu_count() or 1, 4))
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), max(1, batch_size)):
            batch = pairs[start : start + max(1, batch_size)]
            encodings = self.tokenizer.encode_batch(batch)
            encoded = {
                "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encodings], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [item.type_ids for item in encodings], dtype=np.int64
                ),
            }
            inputs = {name: value for name, value in encoded.items() if name in self.input_names}
            logits = np.asarray(self.session.run(None, inputs)[0]).reshape(-1)
            scores.extend(_sigmoid(float(value)) for value in logits)
        return scores


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not settings.RERANKER_ENABLED or len(candidates) < 2:
        return candidates, {
            "ranker": "heuristic",
            "model": None,
            "reranked_count": 0,
            "warning": None,
        }

    candidate_limit = max(
        settings.MAX_RETRIEVAL_K,
        min(settings.RERANKER_MAX_CANDIDATES, len(candidates)),
    )
    head = candidates[:candidate_limit]
    try:
        model = get_reranker()
        scores = model.predict(
            [(question, str(candidate.get("text") or "")) for candidate in head],
            settings.RERANKER_BATCH_SIZE,
        )
        if len(scores) != len(head):
            raise RuntimeError("The local reranker returned an unexpected score count")
    except Exception as exc:
        global _LAST_ERROR
        _LAST_ERROR = _safe_error(exc)
        logger.warning("Semantic reranker unavailable: %s", _LAST_ERROR)
        return candidates, {
            "ranker": "heuristic-fallback",
            "model": settings.RERANKER_MODEL,
            "reranked_count": 0,
            "warning": "The semantic reranker was unavailable; hybrid ranking was used instead",
        }

    weight = max(0.0, min(float(settings.RERANKER_WEIGHT), 1.0))
    reranked: list[dict[str, Any]] = []
    for candidate, cross_score in zip(head, scores, strict=True):
        heuristic_score = max(0.0, min(float(candidate.get("relevance") or 0.0), 1.0))
        combined_score = cross_score * weight + heuristic_score * (1.0 - weight)
        reranked.append(
            {
                **candidate,
                "heuristic_relevance": round(heuristic_score, 6),
                "reranker_score": round(cross_score, 6),
                "relevance": round(combined_score, 6),
            }
        )
    reranked.sort(key=lambda item: float(item["relevance"]), reverse=True)
    return reranked + candidates[candidate_limit:], {
        "ranker": "cross-encoder",
        "model": settings.RERANKER_MODEL,
        "reranked_count": len(reranked),
        "warning": None,
    }


def get_reranker() -> OnnxCrossEncoder:
    identity = (
        settings.RERANKER_MODEL,
        settings.RERANKER_ONNX_FILE,
        settings.RERANKER_MAX_LENGTH,
    )
    global _MODEL, _MODEL_IDENTITY, _LAST_ERROR
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_IDENTITY == identity:
            return _MODEL
        model_path, tokenizer_path = _model_files()
        _MODEL = OnnxCrossEncoder(model_path, tokenizer_path, settings.RERANKER_MAX_LENGTH)
        _MODEL_IDENTITY = identity
        _LAST_ERROR = None
        return _MODEL


def warmup_reranker() -> dict[str, Any]:
    if not settings.RERANKER_ENABLED:
        return reranker_status()
    started = __import__("time").perf_counter()
    try:
        model = get_reranker()
        model.predict([("warmup", "warmup document")], 1)
    except Exception as exc:
        global _LAST_ERROR
        _LAST_ERROR = _safe_error(exc)
    result = reranker_status()
    result["warmup_ms"] = round((__import__("time").perf_counter() - started) * 1000, 1)
    return result


def reranker_status() -> dict[str, Any]:
    model_path, tokenizer_path = _expected_model_files()
    return {
        "enabled": settings.RERANKER_ENABLED,
        "model": settings.RERANKER_MODEL,
        "installed": model_path.is_file() and tokenizer_path.is_file(),
        "loaded": _MODEL is not None,
        "auto_download": settings.RERANKER_AUTO_DOWNLOAD,
        "max_candidates": settings.RERANKER_MAX_CANDIDATES,
        "error": _LAST_ERROR,
    }


def clear_reranker_cache() -> None:
    global _MODEL, _MODEL_IDENTITY, _LAST_ERROR
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_IDENTITY = None
        _LAST_ERROR = None


def _model_files() -> tuple[Path, Path]:
    model_path, tokenizer_path = _expected_model_files()
    if model_path.is_file() and tokenizer_path.is_file():
        return model_path, tokenizer_path
    if not settings.RERANKER_AUTO_DOWNLOAD:
        raise FileNotFoundError(
            "Local reranker files are missing and automatic download is disabled"
        )

    from huggingface_hub import hf_hub_download

    cache_root = Path(settings.RERANKER_CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    local_directory = cache_root / _safe_model_directory(settings.RERANKER_MODEL)
    hf_hub_download(
        repo_id=settings.RERANKER_MODEL,
        filename=settings.RERANKER_ONNX_FILE,
        local_dir=local_directory,
    )
    hf_hub_download(
        repo_id=settings.RERANKER_MODEL,
        filename="tokenizer.json",
        local_dir=local_directory,
    )
    return _expected_model_files()


def _expected_model_files() -> tuple[Path, Path]:
    local_directory = Path(settings.RERANKER_CACHE_DIR) / _safe_model_directory(
        settings.RERANKER_MODEL
    )
    return local_directory / settings.RERANKER_ONNX_FILE, local_directory / "tokenizer.json"


def _safe_model_directory(model_name: str) -> str:
    return model_name.replace("/", "--").replace("\\", "--").replace(":", "-")


def _sigmoid(value: float) -> float:
    clipped = max(-30.0, min(value, 30.0))
    return 1.0 / (1.0 + math.exp(-clipped))


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return (message or exc.__class__.__name__)[:300]
