import pytest

from researchlens.config import Settings
from researchlens.errors import ConfigError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # The project's own .env must not decide what these tests see.
    monkeypatch.setattr("researchlens.config.load_dotenv", lambda *a, **k: False)
    for name in (
        "EMBED_PROVIDER", "EMBED_MODEL", "EMBED_DIM", "MILVUS_COLLECTION",
        "EXTRACT_TITLES", "CHUNK_SIZE", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_the_local_bert_model():
    settings = Settings.load()

    assert settings.embed_provider == "huggingface"
    assert settings.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embed_dim == 384


def test_switching_provider_switches_model_and_dimension(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "openai")

    settings = Settings.load()

    assert settings.embed_model == "text-embedding-3-small"
    assert settings.embed_dim == 1536


def test_an_explicit_model_overrides_the_provider_default(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    monkeypatch.setenv("EMBED_DIM", "768")

    settings = Settings.load()

    assert settings.embed_model == "sentence-transformers/all-mpnet-base-v2"
    assert settings.embed_dim == 768


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "cohere")

    with pytest.raises(ConfigError, match="Unknown EMBED_PROVIDER"):
        Settings.load()


def test_non_numeric_dimension_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "big")

    with pytest.raises(ConfigError, match="must be an integer"):
        Settings.load()


def test_loading_settings_does_not_require_an_openai_key():
    # Local embeddings with no title extraction need no key at all.
    assert Settings.load().embed_provider == "huggingface"


def test_local_ingestion_without_titles_needs_no_openai(monkeypatch):
    monkeypatch.setenv("EXTRACT_TITLES", "false")

    assert Settings.load().needs_openai_for_ingestion is False


def test_title_extraction_needs_openai():
    assert Settings.load().needs_openai_for_ingestion is True


def test_openai_embeddings_need_openai(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    monkeypatch.setenv("EXTRACT_TITLES", "false")

    assert Settings.load().needs_openai_for_ingestion is True


def test_require_openai_key_names_the_reason():
    with pytest.raises(ConfigError, match="needed for the answer model"):
        Settings.load().require_openai_key("the answer model")
