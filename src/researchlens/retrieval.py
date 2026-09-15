"""Retrieval: embed the question, rank chunks from Milvus, answer with an LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from researchlens.config import Settings
from researchlens.embeddings import get_embed_model
from researchlens.errors import ResearchLensError
from researchlens.store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


@dataclass(frozen=True)
class Answer:
    """An LLM answer together with the chunks it was grounded in."""

    text: str
    sources: list[SearchHit]


class RagEngine:
    """Question answering over the indexed paper chunks."""

    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.store = store
        settings.require_openai_key("the answer model")
        # Same instance ingestion used, or the query vectors would not be comparable.
        self._embed_model = get_embed_model(settings)
        self._llm = ChatOpenAI(model=settings.llm_model, temperature=0)

    def retrieve(self, question: str) -> list[SearchHit]:
        """Search the collection and keep the highest scoring chunks."""
        if not self.store.exists():
            raise ResearchLensError(
                f"Collection '{self.settings.collection_name}' does not exist yet. "
                "Run data ingestion first."
            )

        self.store.check_dimension()
        query_embedding = self._embed_model.get_query_embedding(question)
        hits = self.store.search(query_embedding, limit=self.settings.search_limit)
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[: self.settings.top_k]

    def answer(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ResearchLensError("The question is empty.")

        hits = self.retrieve(question)
        if not hits:
            return Answer(
                text="I don't know — no relevant passages were found in the indexed papers.",
                sources=[],
            )

        context = "\n\n".join(hit.text for hit in hits)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self._llm.invoke(prompt)
        return Answer(text=_message_text(response).strip(), sources=hits)


def _message_text(response) -> str:
    """Read the text off a LangChain message across minor API differences."""
    text = getattr(response, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return text
    content = getattr(response, "content", "")
    if isinstance(content, list):  # content blocks
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)
