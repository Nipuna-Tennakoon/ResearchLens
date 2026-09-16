"""The start-up animation and the stderr muting that goes with it."""

import io
import sys

from rich.console import Console

from researchlens import splash


def _plain(align) -> str:
    """Render a frame to plain text for assertions."""
    console = Console(file=io.StringIO(), width=80, no_color=True)
    console.print(align)
    return console.file.getvalue()


def test_the_frame_reveals_letters_one_at_a_time():
    assert "R e s" not in _plain(splash._frame(2, None, None))
    assert "R e" in _plain(splash._frame(2, None, None))
    assert "R e s e a r c h L e n s" in _plain(splash._frame(12, None, None))


def test_unrevealed_letters_are_blank_so_the_width_never_jumps():
    partial = _plain(splash._frame(3, None, None))
    full = _plain(splash._frame(12, None, None))

    assert len(partial.splitlines()[1]) == len(full.splitlines()[1])


def test_the_tagline_appears_only_once_the_word_is_complete():
    assert splash.TAGLINE not in _plain(splash._frame(5, None, None))
    assert splash.TAGLINE in _plain(splash._frame(len(splash.WORDMARK), None, None))


def test_the_patience_note_is_rendered_when_given():
    frame = _plain(splash._frame(12, None, "preparing your workspace..."))

    assert "preparing your workspace..." in frame


def test_no_model_wording_ever_reaches_a_frame():
    """The animation must not mention embeddings, rerankers or downloads."""
    frames = [
        _plain(splash._frame(n, 0, "preparing your workspace..."))
        for n in range(len(splash.WORDMARK) + 1)
    ]
    forbidden = ("model", "embed", "rerank", "download", "huggingface", "loading")

    for frame in frames:
        assert not any(word in frame.lower() for word in forbidden), frame


# --------------------------------------------------------------------------- #
# quiet_stderr
# --------------------------------------------------------------------------- #


def test_quiet_stderr_captures_writes_and_restores_the_stream():
    original = sys.stderr

    with splash.quiet_stderr() as buffer:
        print("noisy progress bar", file=sys.stderr)

    assert "noisy progress bar" in buffer.getvalue()
    assert sys.stderr is original


def test_quiet_stderr_restores_the_stream_after_an_error():
    original = sys.stderr

    try:
        with splash.quiet_stderr():
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert sys.stderr is original


def test_quiet_stderr_lets_exceptions_through():
    import pytest

    with pytest.raises(ValueError):
        with splash.quiet_stderr():
            raise ValueError("this must not be swallowed")


# --------------------------------------------------------------------------- #
# The animation loop
# --------------------------------------------------------------------------- #


def test_the_splash_stops_as_soon_as_the_work_is_ready(monkeypatch):
    monkeypatch.setattr(splash, "REVEAL_SECONDS", 0)
    monkeypatch.setattr(splash, "SHIMMER_SECONDS", 0)
    monkeypatch.setattr(splash, "SETTLE_SECONDS", 0)
    console = Console(file=io.StringIO(), width=80, force_terminal=True)

    calls = []

    def ready():
        calls.append(1)
        return len(calls) > 3  # busy for a few frames, then done

    splash.show_splash(console, ready)

    assert len(calls) == 4


def test_the_splash_ends_even_if_the_work_failed(monkeypatch):
    """A failed load still reports ready, so the animation cannot spin forever."""
    monkeypatch.setattr(splash, "REVEAL_SECONDS", 0)
    monkeypatch.setattr(splash, "SHIMMER_SECONDS", 0)
    monkeypatch.setattr(splash, "SETTLE_SECONDS", 0)
    console = Console(file=io.StringIO(), width=80, force_terminal=True)

    splash.show_splash(console, lambda: True)  # returns immediately
