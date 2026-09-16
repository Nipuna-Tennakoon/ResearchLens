"""Cache-aware loading shared by the embedding model and the reranker."""

import pytest

from researchlens import model_loading


def test_a_cached_model_loads_with_local_files_only(monkeypatch):
    """The whole point: a cached model must not be revalidated against the Hub."""
    seen = {}

    def construct(name, **kwargs):
        seen["name"] = name
        seen["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(model_loading, "is_model_cached", lambda name: True)

    assert model_loading.load_from_cache("some/model", construct, "thing") == "model"
    assert seen["name"] == "some/model"
    assert seen["kwargs"]["local_files_only"] is True
    # The loader and the cache check must agree on where models live.
    assert seen["kwargs"]["cache_folder"] == model_loading.hf_cache_dir()


def test_an_uncached_model_is_allowed_to_download(monkeypatch):
    seen = {}

    def construct(name, **kwargs):
        seen["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(model_loading, "is_model_cached", lambda name: False)

    model_loading.load_from_cache("some/model", construct, "thing")

    assert "local_files_only" not in seen["kwargs"]
    assert seen["kwargs"]["cache_folder"] == model_loading.hf_cache_dir()


def test_an_incomplete_cache_falls_back_to_downloading(monkeypatch):
    attempts = []

    def construct(name, **kwargs):
        attempts.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("missing file in cache")
        return "model"

    monkeypatch.setattr(model_loading, "is_model_cached", lambda name: True)

    assert model_loading.load_from_cache("some/model", construct, "thing") == "model"
    assert len(attempts) == 2
    assert attempts[0]["local_files_only"] is True
    assert "local_files_only" not in attempts[1]


def test_is_model_cached_is_false_for_an_unknown_model():
    assert model_loading.is_model_cached("definitely/not-a-real-model-xyz") is False


def test_the_cache_dir_is_the_huggingface_hub_cache():
    """Not llama-index's own cache, which would keep a second copy of the model."""
    assert "llama_index" not in model_loading.hf_cache_dir()
    assert model_loading.hf_cache_dir().endswith("hub")


# --------------------------------------------------------------------------- #
# SharedModel
# --------------------------------------------------------------------------- #


def test_shared_model_builds_once_and_returns_the_same_instance():
    calls = []
    shared = model_loading.SharedModel("thing")

    def build():
        calls.append(1)
        return object()

    first = shared.get("key", build)
    second = shared.get("key", build)

    assert len(calls) == 1
    assert first is second


def test_shared_model_rebuilds_when_the_key_changes():
    calls = []
    shared = model_loading.SharedModel("thing")

    shared.get("a", lambda: calls.append(1))
    shared.get("b", lambda: calls.append(1))

    assert len(calls) == 2


def test_shared_model_reraises_background_failures():
    shared = model_loading.SharedModel("thing")

    def build():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        shared.get("key", build)
