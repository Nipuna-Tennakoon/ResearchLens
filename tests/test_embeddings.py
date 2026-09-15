"""The embedding model is built once and shared by ingestion and retrieval."""

import threading

import pytest

from researchlens import embeddings
from researchlens.config import Settings
from researchlens.embeddings import (
    _build_embed_model,
    get_embed_model,
    is_ready,
    preload,
    reset_cache,
)
from researchlens.errors import ConfigError


@pytest.fixture(autouse=True)
def clean_cache():
    reset_cache()
    yield
    reset_cache()


def test_unknown_provider_is_rejected():
    settings = Settings(embed_provider="word2vec")

    with pytest.raises(ConfigError, match="Unknown EMBED_PROVIDER"):
        _build_embed_model(settings)


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(embed_provider="openai", embed_model="text-embedding-3-small")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        _build_embed_model(settings)


# --------------------------------------------------------------------------- #
# Shared-instance behaviour
# --------------------------------------------------------------------------- #


class Counter:
    def __init__(self):
        self.calls = 0

    def build(self, settings):
        self.calls += 1
        return object()


def test_the_model_is_built_only_once(monkeypatch):
    counter = Counter()
    monkeypatch.setattr(embeddings, "_build_embed_model", counter.build)
    settings = Settings()

    first = get_embed_model(settings)
    second = get_embed_model(settings)

    assert counter.calls == 1
    assert first is second


def test_preload_then_get_reuses_the_background_load(monkeypatch):
    counter = Counter()
    monkeypatch.setattr(embeddings, "_build_embed_model", counter.build)
    settings = Settings()

    preload(settings)
    model = get_embed_model(settings)

    assert counter.calls == 1
    assert model is get_embed_model(settings)


def test_get_waits_for_a_slow_background_load(monkeypatch):
    released = threading.Event()
    sentinel = object()

    def slow_build(settings):
        released.wait(timeout=5)
        return sentinel

    monkeypatch.setattr(embeddings, "_build_embed_model", slow_build)
    settings = Settings()

    preload(settings)
    assert is_ready(settings) is False  # still loading

    released.set()
    assert get_embed_model(settings) is sentinel
    assert is_ready(settings) is True


def test_a_failure_in_the_background_surfaces_on_the_caller(monkeypatch):
    def failing_build(settings):
        raise ConfigError("no model for you")

    monkeypatch.setattr(embeddings, "_build_embed_model", failing_build)

    with pytest.raises(ConfigError, match="no model for you"):
        get_embed_model(Settings())


def test_changing_the_model_rebuilds(monkeypatch):
    counter = Counter()
    monkeypatch.setattr(embeddings, "_build_embed_model", counter.build)

    get_embed_model(Settings(embed_model="model-a"))
    get_embed_model(Settings(embed_model="model-b"))

    assert counter.calls == 2


def test_is_ready_is_false_before_any_load():
    assert is_ready(Settings()) is False


@pytest.mark.slow
def test_huggingface_provider_loads_the_bert_model():
    """Downloads the model on first run, so it is opt-in via `-m slow`."""
    settings = Settings()  # local all-MiniLM-L6-v2 defaults

    model = get_embed_model(settings)
    vector = model.get_query_embedding("wind farm energy forecasting")

    assert len(vector) == settings.embed_dim
    assert get_embed_model(settings) is model  # shared, not rebuilt
