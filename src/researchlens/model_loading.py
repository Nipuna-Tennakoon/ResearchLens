"""Shared machinery for loading HuggingFace models: cache lookup and background loading.

Both the embedding model and the reranker are slow to load and must be loaded
exactly once per process, so the mechanics live here rather than in each module.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


def hf_cache_dir() -> str:
    """The standard HuggingFace cache (~/.cache/huggingface/hub, or $HF_HOME).

    llama-index otherwise defaults to ~/.cache/llama_index, which would keep a
    second copy of every model and defeat the cache check below.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    return HF_HUB_CACHE


def is_model_cached(model_name: str) -> bool:
    """True if the model's files are already in the local HuggingFace cache.

    Without this check every start-up revalidates each file against the Hub,
    which costs about ninety seconds even though nothing needs downloading.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model_name, cache_dir=hf_cache_dir(), local_files_only=True)
        return True
    except Exception:  # not cached, or the cache is unusable
        return False


def load_from_cache(model_name: str, construct: Callable[..., Any], label: str) -> Any:
    """Build a model from the local cache when possible, downloading only if needed.

    `construct` is called with `cache_folder`, plus `local_files_only=True` on the
    cached path. A cache entry that turns out to be incomplete is repaired from
    the Hub rather than raising.
    """
    cache_folder = hf_cache_dir()

    if is_model_cached(model_name):
        logger.info("Loading %s %s from the local HuggingFace cache", label, model_name)
        try:
            return construct(
                model_name, cache_folder=cache_folder, local_files_only=True
            )
        except Exception:
            logger.warning(
                "Cached copy of %s is incomplete, fetching the missing files",
                model_name,
            )
    else:
        logger.info("Downloading %s %s from HuggingFace (first run only)", label, model_name)

    return construct(model_name, cache_folder=cache_folder)


class _Loader:
    """Loads one model on a daemon thread and hands the same instance to everyone."""

    def __init__(self, build: Callable[[], Any]):
        self._build = build
        self._done = threading.Event()
        self._model: Any = None
        self._error: BaseException | None = None
        # Daemon, so quitting mid-load exits straight away instead of blocking.
        self._thread = threading.Thread(
            target=self._run, name="model-load", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self._model = self._build()
        except BaseException as exc:  # re-raised on the calling thread by result()
            self._error = exc
        finally:
            self._done.set()

    @property
    def ready(self) -> bool:
        return self._done.is_set()

    def result(self) -> Any:
        self._done.wait()
        if self._error is not None:
            raise self._error
        return self._model


class SharedModel:
    """A process-wide, lazily built model, keyed so a config change rebuilds it."""

    def __init__(self, label: str):
        self._label = label
        self._lock = threading.Lock()
        self._loader: _Loader | None = None
        self._key: Any = None

    def _loader_for(self, key: Any, build: Callable[[], Any]) -> _Loader:
        with self._lock:
            if self._loader is None or self._key != key:
                logger.info("Starting background load of %s %s", self._label, key)
                self._key = key
                self._loader = _Loader(build)
            return self._loader

    def preload(self, key: Any, build: Callable[[], Any]) -> None:
        """Begin loading. Returns immediately; the work continues in a thread."""
        self._loader_for(key, build)

    def get(self, key: Any, build: Callable[[], Any]) -> Any:
        """The shared instance, waiting for a background load if one is running."""
        return self._loader_for(key, build).result()

    def is_ready(self, key: Any) -> bool:
        """True if the model is already in memory, so callers can skip a spinner."""
        with self._lock:
            return self._loader is not None and self._key == key and self._loader.ready

    def reset(self) -> None:
        """Drop the cached instance. Used by tests."""
        with self._lock:
            self._loader = None
            self._key = None
