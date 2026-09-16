"""Cross-encoder reranking of vector-search candidates."""

import dataclasses

import pytest

from researchlens import reranking
from researchlens.config import Settings
from researchlens.store import SearchHit


def hit(score: float, text: str, page: int = 1) -> SearchHit:
    return SearchHit(
        score=score, title="A paper", page=page, chunk_index=page, text=text
    )


@pytest.fixture
def settings():
    return Settings(rerank=True, top_k=3, search_limit=10)


def test_reranking_reorders_what_the_vector_search_returned(settings, patched):
    question = "sarima wind forecasting"
    candidates = [
        hit(0.99, "completely unrelated text about cooking", page=1),
        hit(0.50, "sarima wind forecasting results", page=2),
    ]

    ranked = reranking.rerank(settings, question, candidates)

    # The weak vector hit wins because the cross-encoder reads the pair together.
    assert [h.page for h in ranked] == [2, 1]


def test_every_hit_keeps_both_scores(settings, patched):
    candidates = [hit(0.9, "sarima results"), hit(0.8, "unrelated")]

    ranked = reranking.rerank(settings, "sarima", candidates)

    assert all(h.rerank_score is not None for h in ranked)
    assert {h.score for h in ranked} == {0.9, 0.8}  # vector scores preserved


def test_reranking_does_not_mutate_the_input(settings, patched):
    candidates = [hit(0.9, "sarima results")]
    before = dataclasses.asdict(candidates[0])

    reranking.rerank(settings, "sarima", candidates)

    assert dataclasses.asdict(candidates[0]) == before
    assert candidates[0].rerank_score is None


def test_reranking_an_empty_candidate_list_is_a_no_op(settings, patched):
    from conftest import FakeCrossEncoder

    assert reranking.rerank(settings, "sarima", []) == []
    assert FakeCrossEncoder.predictions == 0  # the model is never even consulted


def test_the_reranker_is_loaded_once_across_calls(settings, patched):
    from conftest import FakeCrossEncoder

    reranking.rerank(settings, "sarima", [hit(0.9, "sarima")])
    reranking.rerank(settings, "arima", [hit(0.9, "arima")])

    assert FakeCrossEncoder.builds == 1


def test_preload_is_skipped_when_reranking_is_disabled(patched):
    from conftest import FakeCrossEncoder

    reranking.preload(Settings(rerank=False))

    assert FakeCrossEncoder.builds == 0


def test_preload_starts_the_load_when_reranking_is_enabled(settings, patched):
    from conftest import FakeCrossEncoder

    reranking.preload(settings)
    reranking.get_reranker(settings)

    assert FakeCrossEncoder.builds == 1
