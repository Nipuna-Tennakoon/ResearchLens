"""Milvus-backed vector store holding the embedded paper chunks."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from researchlens.config import Settings
from researchlens.errors import VectorStoreError

logger = logging.getLogger(__name__)

MAX_TITLE_BYTES = 500
MAX_TEXT_BYTES = 10_000
INSERT_BATCH_SIZE = 100


def _truncate(text: str, max_bytes: int) -> str:
    """Trim text so its UTF-8 encoding fits the VARCHAR limit Milvus enforces."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class Chunk:
    """One embedded slice of a paper, matching the collection schema."""

    id: str
    title: str
    page: int
    chunk_index: int
    text: str
    embedding: list[float]

    def to_entity(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": _truncate(self.title, MAX_TITLE_BYTES),
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": _truncate(self.text, MAX_TEXT_BYTES),
            "embedding": self.embedding,
        }


@dataclass(frozen=True)
class SearchHit:
    """A chunk returned by a similarity search, with its score."""

    score: float
    title: str
    page: int
    chunk_index: int
    text: str


def _batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class VectorStore:
    """Thin wrapper over MilvusClient: schema management, upserts and search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_name = settings.collection_name
        try:
            self._client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token,
            )
        except MilvusException as exc:
            raise VectorStoreError(self._connection_hint(exc)) from exc

    def _connection_hint(self, exc: Exception) -> str:
        return (
            f"Could not reach Milvus at {self.settings.milvus_uri}. "
            "Start the server with `docker compose up -d` or point MILVUS_URI at "
            f"the right address.\nDetails: {exc}"
        )

    def vector_dim(self) -> int | None:
        """Dimension of the stored vectors, or None if the collection is missing."""
        try:
            if not self._client.has_collection(self.collection_name):
                return None
            described = self._client.describe_collection(self.collection_name)
        except MilvusException as exc:
            raise VectorStoreError(self._connection_hint(exc)) from exc

        for field in described.get("fields", []):
            if field.get("name") == "embedding":
                dim = (field.get("params") or {}).get("dim")
                return int(dim) if dim is not None else None
        return None

    def check_dimension(self) -> None:
        """Refuse to touch a collection built with a differently sized embedding."""
        stored = self.vector_dim()
        if stored is None or stored == self.settings.embed_dim:
            return
        raise VectorStoreError(
            f"Collection '{self.collection_name}' stores {stored}-dimensional vectors, but "
            f"{self.settings.embed_model} produces {self.settings.embed_dim}. Point "
            "MILVUS_COLLECTION at a different collection, or run `researchlens reset` to "
            "drop this one and ingest again with the current model."
        )

    def ensure_collection(self) -> bool:
        """Create the collection and its index if missing. Returns True if created."""
        try:
            if self._client.has_collection(self.collection_name):
                self.check_dimension()
                return False

            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, max_length=100, is_primary=True)
            schema.add_field("title", DataType.VARCHAR, max_length=MAX_TITLE_BYTES)
            schema.add_field("page", DataType.INT64)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_BYTES)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.settings.embed_dim)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )

            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info("Created collection %s", self.collection_name)
            return True
        except MilvusException as exc:
            raise VectorStoreError(self._connection_hint(exc)) from exc

    def upsert(self, chunks: Iterable[Chunk], batch_size: int = INSERT_BATCH_SIZE) -> int:
        """Write chunks in batches; re-ingesting the same document replaces its rows."""
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        written = 0
        try:
            for batch in _batched(chunk_list, batch_size):
                result = self._client.upsert(
                    collection_name=self.collection_name,
                    data=[chunk.to_entity() for chunk in batch],
                )
                written += int(result.get("upsert_count", len(batch)))
        except MilvusException as exc:
            raise VectorStoreError(
                f"Failed to write chunks to {self.collection_name}: {exc}"
            ) from exc
        return written

    def _ensure_loaded(self) -> None:
        """Milvus only serves queries and searches from a loaded collection."""
        try:
            self._client.load_collection(self.collection_name)
        except MilvusException as exc:
            raise VectorStoreError(
                f"Could not load collection {self.collection_name}: {exc}"
            ) from exc

    def search(self, embedding: list[float], limit: int) -> list[SearchHit]:
        self._ensure_loaded()
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=limit,
                output_fields=["title", "page", "chunk_index", "text"],
            )
        except MilvusException as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

        if not results:
            return []
        return [
            SearchHit(
                score=float(hit["distance"]),
                title=hit["entity"].get("title", ""),
                page=int(hit["entity"].get("page", 0)),
                chunk_index=int(hit["entity"].get("chunk_index", 0)),
                text=hit["entity"].get("text", ""),
            )
            for hit in results[0]
        ]

    def exists(self) -> bool:
        try:
            return bool(self._client.has_collection(self.collection_name))
        except MilvusException as exc:
            raise VectorStoreError(self._connection_hint(exc)) from exc

    def count(self) -> int:
        if not self.exists():
            return 0
        self._ensure_loaded()
        try:
            rows = self._client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"],
            )
        except MilvusException as exc:
            raise VectorStoreError(f"Could not count entities: {exc}") from exc
        return int(rows[0]["count(*)"]) if rows else 0

    def titles(self, limit: int = 1000) -> list[str]:
        """Distinct document titles currently stored, for the status view."""
        if not self.exists():
            return []
        self._ensure_loaded()
        try:
            rows = self._client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["title"],
                limit=limit,
            )
        except MilvusException as exc:
            raise VectorStoreError(f"Could not list documents: {exc}") from exc
        return sorted({row["title"] for row in rows if row.get("title")})

    def drop(self) -> None:
        try:
            if self._client.has_collection(self.collection_name):
                self._client.drop_collection(self.collection_name)
                logger.info("Dropped collection %s", self.collection_name)
        except MilvusException as exc:
            raise VectorStoreError(f"Could not drop collection: {exc}") from exc
