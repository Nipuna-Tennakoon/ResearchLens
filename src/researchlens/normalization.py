"""Text clean-up applied to documents before they are chunked."""

from __future__ import annotations

import re
import unicodedata

from llama_index.core import Document


class DataNormalizer:
    """Normalizes text of Documents produced by SimpleDirectoryReader."""

    def __init__(self, documents: list[Document]):
        self.documents = documents

    def normalize(self) -> list[Document]:
        for doc in self.documents:
            doc.set_content(self._normalize_text(doc.text))
        return self.documents

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)  # convert characters to standard equivalents
        text = self._fix_hyphenation(text)  # hyphenated words become one word
        text = self._collapse_newlines(text)  # replace single newlines with a space
        text = self._collapse_whitespace(text)  # handle repeated spaces or tabs
        return text.strip()

    def _fix_hyphenation(self, text: str) -> str:
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def _collapse_newlines(self, text: str) -> str:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r"[ \t]{2,}", " ", text)
