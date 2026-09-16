"""End-to-end ingestion and retrieval over the real sample PDF, with fake APIs."""

import dataclasses
from pathlib import Path

import pytest

from researchlens.errors import IngestionError, ResearchLensError
from researchlens.ingestion import Ingestor, discover_pdfs
from researchlens.retrieval import RagEngine
from researchlens.store import VectorStore


def test_discover_pdfs_finds_files_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "nested" / "b.PDF").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("ignored")

    assert [p.name for p in discover_pdfs(tmp_path)] == ["a.pdf", "b.PDF"]


def test_discover_pdfs_reports_an_empty_folder(tmp_path):
    with pytest.raises(IngestionError, match="No PDF files found"):
        discover_pdfs(tmp_path)


def test_discover_pdfs_reports_a_missing_folder(tmp_path):
    with pytest.raises(IngestionError, match="Folder not found"):
        discover_pdfs(tmp_path / "nope")


def test_ingest_then_answer(settings, patched, papers_dir):
    store = VectorStore(settings)
    report = Ingestor(settings, store).ingest_folder(papers_dir)

    assert report.files_processed == 1
    assert report.failures == []
    assert report.chunks_written > 0
    assert store.count() == report.chunks_written

    answer = RagEngine(settings, store).answer("What parameters does SARIMA use?")

    assert answer.text
    assert len(answer.sources) == settings.top_k
    assert all(hit.page > 0 for hit in answer.sources)
    assert all(hit.title for hit in answer.sources)
    # The answer must be grounded in the retrieved chunks, not the model's memory.
    for hit in answer.sources:
        assert hit.text in patched.last_prompt


def test_reingesting_the_same_folder_does_not_duplicate(settings, patched, papers_dir):
    store = VectorStore(settings)
    ingestor = Ingestor(settings, store)

    first = ingestor.ingest_folder(papers_dir)
    ingestor.ingest_folder(papers_dir)

    assert store.count() == first.chunks_written


def test_a_failing_file_does_not_abort_the_run(settings, patched, sample_pdf, tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    (folder / sample_pdf.name).write_bytes(sample_pdf.read_bytes())
    (folder / "broken.pdf").write_bytes(b"not really a pdf")

    report = Ingestor(settings, VectorStore(settings)).ingest_folder(folder)

    assert report.files_processed == 1
    assert [p.name for p, _ in report.failures] == ["broken.pdf"]


def test_answering_before_ingestion_explains_what_to_do(settings, patched):
    engine = RagEngine(settings, VectorStore(settings))

    with pytest.raises(ResearchLensError, match="Run data ingestion first"):
        engine.answer("anything")


def test_embedding_dimension_mismatch_is_reported(settings, patched, papers_dir):
    mismatched = type(settings)(**{**settings.__dict__, "embed_dim": 1536})
    store = VectorStore(mismatched)

    report = Ingestor(mismatched, store).ingest_folder(papers_dir)

    assert len(report.failures) == 1
    assert "dimensions" in report.failures[0][1]


def test_retrieval_reranks_the_candidate_set(settings, patched, papers_dir):
    """The reranker sees search_limit candidates and narrows them to top_k."""
    from conftest import FakeCrossEncoder

    store = VectorStore(settings)
    Ingestor(settings, store).ingest_folder(papers_dir)

    hits = RagEngine(settings, store).retrieve("What parameters does SARIMA use?")

    assert len(hits) == settings.top_k
    assert FakeCrossEncoder.predictions == 1
    assert all(h.rerank_score is not None for h in hits)
    # Ordered by cross-encoder score, not vector similarity.
    assert [h.rerank_score for h in hits] == sorted(
        [h.rerank_score for h in hits], reverse=True
    )


def test_reranking_can_be_turned_off(settings, patched, papers_dir):
    from conftest import FakeCrossEncoder

    no_rerank = dataclasses.replace(settings, rerank=False)
    store = VectorStore(no_rerank)
    Ingestor(no_rerank, store).ingest_folder(papers_dir)

    hits = RagEngine(no_rerank, store).retrieve("What parameters does SARIMA use?")

    assert len(hits) == no_rerank.top_k
    assert FakeCrossEncoder.predictions == 0
    assert all(h.rerank_score is None for h in hits)
    # Falls back to vector similarity order.
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
