"""Data ingestion: PDFs on disk -> cleaned text -> chunks -> embedded vectors."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.readers.file import PyMuPDFReader

from researchlens.config import Settings
from researchlens.embeddings import get_embed_model
from researchlens.errors import IngestionError
from researchlens.normalization import DataNormalizer
from researchlens.store import Chunk, VectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Path, int], None]
"""Called with (pdf_path, chunks_written) after each file is ingested."""


@dataclass
class IngestionReport:
    """Outcome of an ingestion run, used to render the CLI summary."""

    files_processed: int = 0
    chunks_written: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)


def discover_pdfs(folder: Path) -> list[Path]:
    """Return every PDF under `folder`, searched recursively and sorted by path."""
    if not folder.exists():
        raise IngestionError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise IngestionError(f"Not a folder: {folder}")

    pdfs = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise IngestionError(f"No PDF files found in {folder}")
    return pdfs


class Ingestor:
    """Runs the loading, cleaning, chunking and embedding pipeline over PDF files."""

    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.store = store
        self._reader_extractor = {".pdf": PyMuPDFReader()}

        transformations: list = [
            SentenceSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        ]
        if settings.extract_titles:
            transformations.append(TitleExtractor())
        transformations.append(get_embed_model(settings))

        self._pipeline = IngestionPipeline(transformations=transformations)

    def ingest_folder(
        self,
        folder: Path,
        on_file_done: ProgressCallback | None = None,
    ) -> IngestionReport:
        """Ingest every PDF in `folder`. A failing file is reported, not fatal."""
        pdfs = discover_pdfs(folder)
        self.store.ensure_collection()

        report = IngestionReport()
        for pdf in pdfs:
            try:
                written = self.ingest_file(pdf)
            except Exception as exc:  # one bad PDF must not abort the whole run
                logger.exception("Failed to ingest %s", pdf)
                report.failures.append((pdf, str(exc)))
                continue

            report.files_processed += 1
            report.chunks_written += written
            if on_file_done is not None:
                on_file_done(pdf, written)

        return report

    def ingest_file(self, pdf: Path) -> int:
        """Ingest a single PDF and return the number of chunks written."""
        chunks = self._build_chunks(pdf)
        if not chunks:
            raise IngestionError(f"No extractable text in {pdf.name}")
        return self.store.upsert(chunks)

    def _build_chunks(self, pdf: Path) -> list[Chunk]:
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(pdf)],
            file_extractor=self._reader_extractor,
        )
        documents = DataNormalizer(reader.load_data()).normalize()

        nodes: list[BaseNode] = []
        for document in documents:
            if not document.text.strip():
                continue  # blank page, nothing to embed
            nodes.extend(self._pipeline.run(documents=[document]))

        return [
            self._to_chunk(node, index=index, pdf=pdf)
            for index, node in enumerate(nodes)
        ]

    def _to_chunk(self, node: BaseNode, index: int, pdf: Path) -> Chunk:
        embedding = node.embedding
        if embedding is None:
            raise IngestionError(f"Chunk {index} of {pdf.name} has no embedding")
        if len(embedding) != self.settings.embed_dim:
            raise IngestionError(
                f"{self.settings.embed_model} returned {len(embedding)} dimensions but the "
                f"collection expects {self.settings.embed_dim}. Set EMBED_DIM={len(embedding)} "
                "and re-create the collection with `researchlens reset`."
            )

        metadata = node.metadata or {}
        return Chunk(
            id=node.id_,
            title=str(metadata.get("document_title") or pdf.stem),
            page=self._page_number(metadata),
            chunk_index=index,
            text=node.get_content(),
            embedding=list(embedding),
        )

    @staticmethod
    def _page_number(metadata: dict) -> int:
        """PyMuPDFReader records the page in `source`; fall back to 0 if absent."""
        raw = metadata.get("source", metadata.get("page_label", 0))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
