"""Animated start-up wordmark, shown while the models load in the background.

Model loading takes seconds (much longer on a first run, when the weights are
downloaded), and the libraries doing it write progress bars and warnings to
stderr. The splash covers that wait with something worth looking at and keeps
the machinery out of sight.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from typing import Callable, Iterator

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text

WORDMARK = "ResearchLens"
TAGLINE = "retrieval-augmented research assistant"

# Letters appear at this rate, then the shimmer sweeps until loading finishes.
REVEAL_SECONDS = 0.055
SHIMMER_SECONDS = 0.045
SETTLE_SECONDS = 0.45
FRAME_RATE = 30

# Only after this long do we admit that something is taking a while.
PATIENCE_SECONDS = 8.0

DIM = "grey42"
BASE = "bold cyan"
BRIGHT = "bold white"


@contextlib.contextmanager
def quiet_stderr() -> Iterator[io.StringIO]:
    """Swallow library chatter (tqdm bars, hub warnings) for the duration.

    Errors still travel as exceptions; only the noise is dropped. The buffer is
    yielded so a caller can inspect it if something goes wrong.
    """
    buffer = io.StringIO()
    original = sys.stderr
    sys.stderr = buffer
    try:
        yield buffer
    finally:
        sys.stderr = original


def _frame(revealed: int, shimmer: int | None, note: str | None) -> Align:
    """One frame: the wordmark with `revealed` letters, a shimmer at `shimmer`."""
    word = Text(justify="center")
    for index, letter in enumerate(WORDMARK):
        if index >= revealed:
            word.append(" ", style=DIM)
        elif index == shimmer:
            word.append(letter, style=BRIGHT)
        else:
            word.append(letter, style=BASE)
        if index < len(WORDMARK) - 1:
            word.append(" ")

    block = Text("\n")
    block.append_text(word)
    if revealed >= len(WORDMARK):
        block.append("\n")
        block.append(TAGLINE, style=DIM)
    if note:
        block.append("\n\n")
        block.append(note, style=DIM)
    block.append("\n")
    return Align.center(block)


def show_splash(
    console: Console,
    ready: Callable[[], bool],
    *,
    patience: float = PATIENCE_SECONDS,
) -> None:
    """Animate the wordmark until `ready()` is true, then clear it away.

    `ready()` must eventually return true even if the work behind it failed,
    otherwise this spins forever; the caller surfaces the error afterwards.
    """
    started = time.monotonic()

    with Live(
        _frame(0, None, None),
        console=console,
        refresh_per_second=FRAME_RATE,
        transient=True,
        # Live hijacks stdout/stderr by default and replays what it captures
        # above the animation, which would undo quiet_stderr() and let the
        # libraries' progress bars through.
        redirect_stdout=False,
        redirect_stderr=False,
    ) as live:
        # Letters appear one at a time.
        for revealed in range(1, len(WORDMARK) + 1):
            live.update(_frame(revealed, revealed - 1, None))
            time.sleep(REVEAL_SECONDS)

        # Then a highlight sweeps back and forth until the work is done.
        position = 0
        direction = 1
        while not ready():
            note = None
            if time.monotonic() - started > patience:
                note = "preparing your workspace..."
            live.update(_frame(len(WORDMARK), position, note))
            position += direction
            if position in (0, len(WORDMARK) - 1):
                direction *= -1
            time.sleep(SHIMMER_SECONDS)

        live.update(_frame(len(WORDMARK), None, None))
        time.sleep(SETTLE_SECONDS)
