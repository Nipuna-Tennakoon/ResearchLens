import pytest

from researchlens.config import Settings
from researchlens.embeddings import build_embed_model
from researchlens.errors import ConfigError


def test_unknown_provider_is_rejected():
    settings = Settings(embed_provider="word2vec")

    with pytest.raises(ConfigError, match="Unknown EMBED_PROVIDER"):
        build_embed_model(settings)


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(embed_provider="openai", embed_model="text-embedding-3-small")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_embed_model(settings)


@pytest.mark.slow
def test_huggingface_provider_loads_the_bert_model():
    """Downloads the model on first run, so it is opt-in via `-m slow`."""
    settings = Settings()  # local all-MiniLM-L6-v2 defaults

    model = build_embed_model(settings)
    vector = model.get_query_embedding("wind farm energy forecasting")

    assert len(vector) == settings.embed_dim
