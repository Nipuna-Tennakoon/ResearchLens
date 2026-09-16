"""Cross-encoder reranking of the chunks returned by the vector search.

Vector search scores a question and a chunk independently, so it is fast but
approximate. A cross-encoder reads the question and the chunk *together* and
scores the pair directly, which is far more accurate but too slow to run over a
whole collection. Retrieving a wide candidate set and reranking it gets both.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from researchlens.config import Settings
from researchlens.model_loading import SharedModel, load_from_cache
from researchlens.store import SearchHit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_shared = SharedModel("reranker")


def _build_reranker(settings: Settings) -> "CrossEncoder":
    """Instantiate the cross-encoder. Slow; call through the cache."""
    # Imported lazily: it pulls in torch, which is slow to load.
    from sentence_transformers import CrossEncoder

    def construct(name: str, cache_folder: str, **kwargs):
        # CrossEncoder already reads and writes the HuggingFace cache that
        # cache_folder points at; forwarding it only triggers a deprecation
        # warning from transformers.
        return CrossEncoder(name, **kwargs)

    return load_from_cache(settings.rerank_model, construct, label="reranker")


def preload(settings: Settings) -> None:
    """Begin loading the reranker in the background."""
    if not settings.rerank:
        return
    _shared.preload(settings.rerank_model, lambda: _build_reranker(settings))


def is_ready(settings: Settings) -> bool:
    return _shared.is_ready(settings.rerank_model)


def get_reranker(settings: Settings) -> "CrossEncoder":
    """The shared cross-encoder, waiting for a background load if one is running."""
    return _shared.get(settings.rerank_model, lambda: _build_reranker(settings))


def reset_cache() -> None:
    """Drop the cached reranker. Used by tests."""
    _shared.reset()


def rerank(settings: Settings, question: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Score every (question, chunk) pair with the cross-encoder and re-sort.

    Returns hits ordered by cross-encoder score, each carrying its `rerank_score`.
    The vector similarity stays on `score` so both are visible to the caller.
    """
    if not hits:
        return []

    model = get_reranker(settings)
    pairs = [(question, hit.text) for hit in hits]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = [
        replace(hit, rerank_score=float(score))
        for hit, score in zip(hits, scores, strict=True)
    ]
    scored.sort(key=lambda hit: hit.rerank_score, reverse=True)
    logger.info("Reranked %d candidates with %s", len(scored), settings.rerank_model)
    return scored
