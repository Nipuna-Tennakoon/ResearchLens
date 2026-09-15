"""Runtime settings, read from the environment with notebook-compatible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from researchlens.errors import ConfigError

# Per-provider defaults, so switching EMBED_PROVIDER alone gives a working setup.
PROVIDER_DEFAULTS: dict[str, tuple[str, int]] = {
    "huggingface": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "openai": ("text-embedding-3-small", 1536),
}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Everything the ingestion and retrieval pipelines need to run."""

    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = "root:Milvus"
    collection_name: str = "paper_chunks_2"

    embed_provider: str = "huggingface"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 150
    chunk_overlap: int = 10
    extract_titles: bool = True

    search_limit: int = 10
    top_k: int = 3

    @property
    def needs_openai_for_ingestion(self) -> bool:
        """Title extraction and OpenAI embeddings are the paid parts of ingestion."""
        return self.embed_provider == "openai" or self.extract_titles

    def require_openai_key(self, reason: str) -> None:
        """Fail early, and say which part of the run needs the key."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigError(
                f"OPENAI_API_KEY is not set, but it is needed for {reason}. "
                "Add it to your .env file or export it before running ResearchLens."
            )

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        """Build settings from a .env file (if present) and the process environment."""
        load_dotenv(dotenv_path=env_file, override=False)

        provider = os.getenv("EMBED_PROVIDER", cls.embed_provider).strip().lower()
        if provider not in PROVIDER_DEFAULTS:
            raise ConfigError(
                f"Unknown EMBED_PROVIDER {provider!r}. "
                f"Use one of: {', '.join(sorted(PROVIDER_DEFAULTS))}."
            )
        default_model, default_dim = PROVIDER_DEFAULTS[provider]

        return cls(
            milvus_uri=os.getenv("MILVUS_URI", cls.milvus_uri),
            milvus_token=os.getenv("MILVUS_TOKEN", cls.milvus_token),
            collection_name=os.getenv("MILVUS_COLLECTION", cls.collection_name),
            embed_provider=provider,
            embed_model=os.getenv("EMBED_MODEL", default_model),
            embed_dim=_int_env("EMBED_DIM", default_dim),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            chunk_size=_int_env("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int_env("CHUNK_OVERLAP", cls.chunk_overlap),
            extract_titles=os.getenv("EXTRACT_TITLES", "true").lower() != "false",
            search_limit=_int_env("SEARCH_LIMIT", cls.search_limit),
            top_k=_int_env("TOP_K", cls.top_k),
        )
