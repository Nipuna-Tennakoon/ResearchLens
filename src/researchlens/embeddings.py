"""The embedding model: built once, in the background, shared by both pipelines.

A query must be embedded with the same model that embedded the stored chunks,
otherwise the vectors live in different spaces and the search is meaningless.
Local models also take tens of seconds to load, so the CLI starts loading as
soon as it knows it will need one and everything else waits on that one copy.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from researchlens.config import Settings
from researchlens.errors import ConfigError

if TYPE_CHECKING:  # heavy imports stay out of the CLI start-up path
    from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


def _build_embed_model(settings: Settings) -> "BaseEmbedding":
    """Instantiate the configured embedding model. Slow; call through the cache."""
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


class _Loader:
    """Loads one model on a daemon thread and hands the same instance to everyone."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._done = threading.Event()
        self._model: "BaseEmbedding | None" = None
        self._error: BaseException | None = None
        # Daemon, so quitting mid-load exits straight away instead of blocking.
        self._thread = threading.Thread(
            target=self._run, name="embed-model-load", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self._model = _build_embed_model(self._settings)
        except BaseException as exc:  # re-raised on the calling thread by result()
            self._error = exc
        finally:
            self._done.set()

    @property
    def ready(self) -> bool:
        return self._done.is_set()

    def result(self) -> "BaseEmbedding":
        self._done.wait()
        if self._error is not None:
            raise self._error
        assert self._model is not None
        return self._model


_lock = threading.Lock()
_loader: _Loader | None = None
_loader_key: tuple[str, str] | None = None


def _key(settings: Settings) -> tuple[str, str]:
    return (settings.embed_provider, settings.embed_model)


def _loader_for(settings: Settings) -> _Loader:
    """Return the loader for these settings, starting one if needed."""
    global _loader, _loader_key

    key = _key(settings)
    with _lock:
        if _loader is None or _loader_key != key:
            logger.info("Starting background load of %s", settings.embed_model)
            _loader_key = key
            _loader = _Loader(settings)
        return _loader


def preload(settings: Settings) -> None:
    """Begin loading the model. Returns immediately; the work continues in a thread."""
    _loader_for(settings)


def is_ready(settings: Settings) -> bool:
    """True if the model is already in memory, so callers can skip a spinner."""
    with _lock:
        return _loader is not None and _loader_key == _key(settings) and _loader.ready


def get_embed_model(settings: Settings) -> "BaseEmbedding":
    """The shared model, waiting for a background load to finish if one is running."""
    return _loader_for(settings).result()


def reset_cache() -> None:
    """Drop the cached model. Used by tests; the loaded copy is not reusable after."""
    global _loader, _loader_key
    with _lock:
        _loader = None
        _loader_key = None
