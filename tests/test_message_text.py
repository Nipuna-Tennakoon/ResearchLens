"""Reading the answer text off a LangChain message, across API variations."""

import warnings

from researchlens.retrieval import _message_text


class TextAccessor(str):
    """Mimics LangChain's .text: a str that is also callable, and warns if called."""

    def __call__(self):
        warnings.warn("Calling .text() as a method is deprecated", DeprecationWarning)
        return str(self)


class Message:
    def __init__(self, text=None, content=""):
        if text is not None:
            self.text = text
        self.content = content


def test_the_text_property_is_read_without_calling_it():
    message = Message(text=TextAccessor("the answer"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _message_text(message)

    assert result == "the answer"
    assert type(result) is str  # not the str subclass
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_a_plain_string_text_attribute_works():
    assert _message_text(Message(text="hello")) == "hello"


def test_the_older_method_style_text_is_still_supported():
    class OldMessage:
        content = "ignored"

        def text(self):
            return "from the method"

    assert _message_text(OldMessage()) == "from the method"


def test_it_falls_back_to_content_when_text_is_empty():
    assert _message_text(Message(text="", content="from content")) == "from content"


def test_it_falls_back_to_content_when_there_is_no_text():
    assert _message_text(Message(content="from content")) == "from content"


def test_content_blocks_are_joined():
    message = Message(content=[{"text": "part one "}, {"text": "part two"}])

    assert _message_text(message) == "part one part two"
