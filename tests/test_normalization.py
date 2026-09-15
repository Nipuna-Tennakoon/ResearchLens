from llama_index.core import Document

from researchlens.normalization import DataNormalizer


def test_normalize_fixes_hyphenation_and_whitespace():
    doc = Document(text="fore-\ncasting   models\nare  useful\n\n\nNext section")
    (normalized,) = DataNormalizer([doc]).normalize()

    assert "forecasting models are useful" in normalized.text
    assert "  " not in normalized.text
    assert "\n\n\n" not in normalized.text


def test_normalize_strips_and_standardizes_unicode():
    doc = Document(text="  ﬁne-tuning  ")
    (normalized,) = DataNormalizer([doc]).normalize()

    assert normalized.text == "fine-tuning"
