"""Fakes that let the pipeline run without Milvus, OpenAI or network access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from llama_index.core.schema import TransformComponent
from pymilvus.exceptions import MilvusException

from researchlens import embeddings
from researchlens.config import Settings

EMBED_DIM = 8


def _hash_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(EMBED_DIM)]


class FakeEmbedding(TransformComponent):
    """Deterministic hash-based embeddings, so tests never call OpenAI."""

    def get_query_embedding(self, query: str) -> list[float]:
        return _hash_vector(query)

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.embedding = _hash_vector(node.get_content())
        return nodes


class FakeMilvusClient:
    """In-memory stand-in for MilvusClient covering the calls VectorStore makes.

    State is shared between instances, the way a real server is shared between
    connections, so data survives across separate CLI invocations in a test.
    """

    collections: dict[str, list[dict]] = {}
    schemas: dict[str, object] = {}

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def reset(cls) -> None:
        cls.collections = {}
        cls.schemas = {}

    # -- schema helpers ---------------------------------------------------- #
    def create_schema(self, **kwargs):
        class _Schema:
            def __init__(self):
                self.fields = []

            def add_field(self, field_name, datatype, **params):
                self.fields.append((field_name, datatype, params))

        return _Schema()

    def prepare_index_params(self):
        class _IndexParams:
            def __init__(self):
                self.indexes = []

            def add_index(self, **kwargs):
                self.indexes.append(kwargs)

        return _IndexParams()

    # -- collection lifecycle ---------------------------------------------- #
    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, schema=None, index_params=None):
        self.collections[collection_name] = []
        self.schemas[collection_name] = schema

    def drop_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def load_collection(self, collection_name):
        if collection_name not in self.collections:
            raise MilvusException(message=f"collection not found: {collection_name}")

    def describe_collection(self, collection_name):
        schema = self.schemas.get(collection_name)
        fields = []
        for name, _datatype, params in getattr(schema, "fields", []):
            fields.append({"name": name, "params": params})
        return {"fields": fields}

    # -- data -------------------------------------------------------------- #
    def upsert(self, collection_name, data):
        rows = self.collections.setdefault(collection_name, [])
        by_id = {row["id"]: row for row in rows}
        for entity in data:
            by_id[entity["id"]] = entity
        self.collections[collection_name] = list(by_id.values())
        return {"upsert_count": len(data)}

    def query(self, collection_name, filter="", output_fields=None, limit=None):
        rows = self.collections.get(collection_name, [])
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        selected = rows[:limit] if limit else rows
        fields = output_fields or []
        return [{field: row.get(field) for field in fields} for row in selected]

    def search(self, collection_name, data, limit, output_fields=None):
        query_vector = data[0]
        rows = self.collections.get(collection_name, [])
        fields = output_fields or []

        def similarity(row):
            return -sum(
                (a - b) ** 2 for a, b in zip(query_vector, row["embedding"], strict=False)
            )

        ranked = sorted(rows, key=similarity, reverse=True)[:limit]
        return [
            [
                {
                    "distance": similarity(row),
                    "entity": {field: row.get(field) for field in fields},
                }
                for row in ranked
            ]
        ]


class FakeLLM:
    """Records the prompt it was given and returns a canned answer."""

    last_prompt: str | None = None

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt: str):
        FakeLLM.last_prompt = prompt

        class _Message:
            text = "SARIMA uses the parameters p, d, q, P, D, Q and s."

        return _Message()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        collection_name="test_chunks",
        embed_provider="huggingface",
        embed_model="fake-test-embedding",
        embed_dim=EMBED_DIM,
        chunk_size=150,
        chunk_overlap=10,
        extract_titles=False,  # keeps the LLM out of ingestion
        search_limit=10,
        top_k=3,
    )


class BuildCounter:
    """Counts how many times the real model builder was invoked."""

    calls = 0

    @classmethod
    def build(cls, settings):
        cls.calls += 1
        return FakeEmbedding()


@pytest.fixture
def patched(monkeypatch):
    FakeMilvusClient.reset()
    FakeLLM.last_prompt = None
    BuildCounter.calls = 0
    embeddings.reset_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("researchlens.store.MilvusClient", FakeMilvusClient)
    # Patch the builder, not the call sites, so the shared-instance cache is real.
    monkeypatch.setattr("researchlens.embeddings._build_embed_model", BuildCounter.build)
    monkeypatch.setattr("researchlens.retrieval.ChatOpenAI", FakeLLM)
    yield FakeLLM
    embeddings.reset_cache()


@pytest.fixture
def sample_pdf() -> Path:
    pdfs = sorted(Path("data").glob("*.pdf"))
    if not pdfs:
        pytest.skip("no sample PDF in data/")
    return pdfs[0]
