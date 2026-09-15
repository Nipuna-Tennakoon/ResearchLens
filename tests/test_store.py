import pytest

from researchlens.store import MAX_TEXT_BYTES, Chunk, VectorStore


@pytest.fixture
def store(settings, patched):
    store = VectorStore(settings)
    store.ensure_collection()
    return store


def _chunk(index: int, text: str = "some text") -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        title="A paper",
        page=index,
        chunk_index=index,
        text=text,
        embedding=[float(index)] * 8,
    )


def test_ensure_collection_is_idempotent(settings, patched):
    store = VectorStore(settings)
    assert store.ensure_collection() is True
    assert store.ensure_collection() is False


def test_upsert_replaces_rows_with_the_same_id(store):
    store.upsert([_chunk(1, "first version")])
    store.upsert([_chunk(1, "second version")])

    assert store.count() == 1


def test_upsert_batches_large_inputs(store):
    written = store.upsert([_chunk(i) for i in range(250)], batch_size=100)

    assert written == 250
    assert store.count() == 250


def test_long_text_is_truncated_to_the_varchar_limit():
    entity = _chunk(1, "x" * (MAX_TEXT_BYTES + 500)).to_entity()

    assert len(entity["text"].encode("utf-8")) <= MAX_TEXT_BYTES


def test_search_returns_ranked_hits(store):
    store.upsert([_chunk(i) for i in range(5)])

    hits = store.search([3.0] * 8, limit=2)

    assert len(hits) == 2
    assert hits[0].page == 3
    assert hits[0].score >= hits[1].score


def test_drop_removes_the_collection(store):
    store.drop()

    assert store.exists() is False
    assert store.count() == 0
