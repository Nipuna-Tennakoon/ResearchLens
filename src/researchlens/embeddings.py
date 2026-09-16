"""The embedding model: built once, in the background, shared by both pipelines.

A query must be embedded with the same model that embedded the stored chunks,
otherwise the vectors live in different spaces and the search is meaningless.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from researchlens.config import Settings
from researchlens.errors import ConfigError
from researchlens.model_loading import (  # re-exported for callers and tests
    SharedModel,
    hf_cache_dir,
    is_model_cached,
    load_from_cache,
)

if TYPE_CHECKING:  # heavy imports stay out of the CLI start-up path
    from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

__all__ = [
    "get_embed_model",
    "hf_cache_dir",
    "is_model_cached",
    "is_ready",
    "preload",
    "reset_cache",
]

_shared = SharedModel("embedding model")


def _load_huggingface(model_name: str) -> "BaseEmbedding":
    """Load from the local cache when possible, downloading only when needed."""
    # Imported lazily: it pulls in torch, which is slow to load.
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    def construct(name: str, **kwargs):
        return HuggingFaceEmbedding(model_name=name, **kwargs)

    return load_from_cache(model_name, construct, label="embedding model")


def _build_embed_model(settings: Settings) -> "BaseEmbedding":
    """Instantiate the configured embedding model. Slow; call through the cache."""
    provider = settings.embed_provider

    if provider == "huggingface":
        return _load_huggingface(settings.embed_model)

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        settings.require_openai_key("the OpenAI embedding model")
        return OpenAIEmbedding(model=settings.embed_model)

    raise ConfigError(
        f"Unknown EMBED_PROVIDER {provider!r}. Use 'huggingface' or 'openai'."
    )


def _key(settings: Settings) -> tuple[str, str]:
    return (settings.embed_provider, settings.embed_model)


def preload(settings: Settings) -> None:
    """Begin loading the model. Returns immediately; the work continues in a thread."""
    _shared.preload(_key(settings), lambda: _build_embed_model(settings))


def is_ready(settings: Settings) -> bool:
    """True if the model is already in memory, so callers can skip a spinner."""
    return _shared.is_ready(_key(settings))


def get_embed_model(settings: Settings) -> "BaseEmbedding":
    """The shared model, waiting for a background load to finish if one is running."""
    return _shared.get(_key(settings), lambda: _build_embed_model(settings))


def reset_cache() -> None:
    """Drop the cached model. Used by tests; the loaded copy is not reusable after."""
    _shared.reset()
