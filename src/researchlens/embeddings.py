"""The embedding model, built in one place so ingestion and retrieval always agree.

A query must be embedded with the same model that embedded the stored chunks,
otherwise the vectors live in different spaces and the search is meaningless.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from researchlens.config import Settings
from researchlens.errors import ConfigError

if TYPE_CHECKING:  # heavy imports stay out of the CLI start-up path
    from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


def build_embed_model(settings: Settings) -> "BaseEmbedding":
    """Instantiate the configured embedding model."""
    provider = settings.embed_provider

    if provider == "huggingface":
        # Imported lazily: it pulls in torch, which is slow to load.
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info("Loading local embedding model %s", settings.embed_model)
        return HuggingFaceEmbedding(model_name=settings.embed_model)

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        settings.require_openai_key("the OpenAI embedding model")
        return OpenAIEmbedding(model=settings.embed_model)

    raise ConfigError(
        f"Unknown EMBED_PROVIDER {provider!r}. Use 'huggingface' or 'openai'."
    )
