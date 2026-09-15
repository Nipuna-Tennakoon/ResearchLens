"""Command line interface for ResearchLens."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from researchlens import __version__
from researchlens.config import Settings
from researchlens.errors import ResearchLensError
from researchlens.ingestion import IngestionReport, Ingestor, discover_pdfs
from researchlens.retrieval import Answer, RagEngine
from researchlens.store import VectorStore

console = Console()

app = typer.Typer(
    name="researchlens",
    help="RAG over a folder of research papers: ingest PDFs, then ask questions.",
    no_args_is_help=False,
    add_completion=False,
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _fail(message: str) -> typer.Exit:
    console.print(f"[bold red]Error:[/bold red] {message}")
    return typer.Exit(code=1)


def _settings() -> Settings:
    try:
        return Settings.load()
    except ResearchLensError as exc:
        raise _fail(str(exc)) from exc


def _store(settings: Settings) -> VectorStore:
    try:
        return VectorStore(settings)
    except ResearchLensError as exc:
        raise _fail(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def _run_ingestion(folder: Path) -> IngestionReport:
    settings = _settings()
    store = _store(settings)

    try:
        pdfs = discover_pdfs(folder)
    except ResearchLensError as exc:
        raise _fail(str(exc)) from exc

    if settings.needs_openai_for_ingestion:
        try:
            settings.require_openai_key("title extraction and/or OpenAI embeddings")
        except ResearchLensError as exc:
            raise _fail(str(exc)) from exc

    console.print(
        f"Found [bold]{len(pdfs)}[/bold] PDF file(s) in [cyan]{folder}[/cyan] "
        f"-> collection [cyan]{settings.collection_name}[/cyan]"
    )
    console.print(
        f"Embedding with [cyan]{settings.embed_model}[/cyan] "
        f"({settings.embed_provider}, {settings.embed_dim}d)"
    )

    try:
        store.check_dimension()
    except ResearchLensError as exc:
        raise _fail(str(exc)) from exc

    ingestor = Ingestor(settings, store)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting", total=len(pdfs))

        def on_file_done(pdf: Path, chunks: int) -> None:
            progress.advance(task)
            progress.console.print(f"  [green]OK[/green] {pdf.name} — {chunks} chunks")

        try:
            report = ingestor.ingest_folder(folder, on_file_done=on_file_done)
        except ResearchLensError as exc:
            raise _fail(str(exc)) from exc

    for pdf, reason in report.failures:
        console.print(f"  [red]FAILED[/red] {pdf.name} — {reason}")

    console.print(
        Panel(
            f"Files ingested: [bold]{report.files_processed}[/bold]\n"
            f"Chunks written: [bold]{report.chunks_written}[/bold]\n"
            f"Failures: [bold]{len(report.failures)}[/bold]",
            title="Ingestion complete",
            border_style="green" if not report.failures else "yellow",
        )
    )
    return report


@app.command()
def ingest(
    path: Path = typer.Argument(  # noqa: B008
        None,
        help="Folder containing the PDF files. Prompted for when omitted.",
    ),
) -> None:
    """Load every PDF in a folder into the vector database."""
    folder = path if path is not None else _prompt_for_folder()
    report = _run_ingestion(folder)
    if report.files_processed == 0:
        raise typer.Exit(code=1)


def _prompt_for_folder() -> Path:
    raw = typer.prompt("Path to the data folder", default="data")
    return Path(raw).expanduser()


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def _render_answer(answer: Answer, show_sources: bool) -> None:
    console.print(Panel(answer.text or "(empty answer)", title="Answer", border_style="cyan"))

    if not show_sources or not answer.sources:
        return

    table = Table(title="Sources", show_lines=False, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("Score", width=7)
    table.add_column("Page", width=5)
    table.add_column("Document", overflow="fold")
    table.add_column("Excerpt", overflow="fold")
    for rank, hit in enumerate(answer.sources, start=1):
        excerpt = hit.text if len(hit.text) <= 160 else hit.text[:157] + "..."
        table.add_row(str(rank), f"{hit.score:.4f}", str(hit.page), hit.title, excerpt)
    console.print(table)


EXIT_WORDS = {"exit", "quit", "q"}
BACK_WORDS = {"back", "menu"}


@app.command()
def ask(
    question: str = typer.Argument(None, help="Question to answer. Omit for an interactive session."),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show the retrieved chunks."),
) -> None:
    """Ask a question about the indexed papers."""
    settings = _settings()
    store = _store(settings)
    engine = RagEngine(settings, store)

    if question:
        _answer_once(engine, question, sources)
        return

    _ask_loop(engine, sources, from_menu=False)


def _ask_loop(engine: RagEngine, sources: bool, from_menu: bool) -> bool:
    """Question-and-answer session. Returns True when the user wants to quit the app."""
    hint = "type a question, 'exit' to quit"
    if from_menu:
        hint += ", 'back' to return to the menu"
    console.print(f"Interactive retrieval — {hint}.\n")

    while True:
        try:
            user_question = typer.prompt("Question").strip()
        except (EOFError, KeyboardInterrupt, typer.Abort):
            console.print()
            return True

        command = user_question.lower()
        if command in EXIT_WORDS:
            console.print("Bye.")
            return True
        if from_menu and command in BACK_WORDS:
            return False
        if not user_question:
            continue

        _answer_once(engine, user_question, sources, fatal=False)
        console.print()


def _answer_once(engine: RagEngine, question: str, sources: bool, fatal: bool = True) -> None:
    try:
        with console.status("Retrieving and generating..."):
            answer = engine.answer(question)
    except ResearchLensError as exc:
        if fatal:
            raise _fail(str(exc)) from exc
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return
    _render_answer(answer, show_sources=sources)


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #


@app.command()
def status() -> None:
    """Show what is currently stored in the vector database."""
    settings = _settings()
    store = _store(settings)

    try:
        exists = store.exists()
        count = store.count() if exists else 0
        titles = store.titles() if exists else []
    except ResearchLensError as exc:
        raise _fail(str(exc)) from exc

    table = Table(show_header=False, box=None)
    table.add_row("Milvus URI", settings.milvus_uri)
    table.add_row("Collection", settings.collection_name)
    table.add_row("Exists", "yes" if exists else "no")
    table.add_row("Chunks", str(count))
    table.add_row(
        "Embedding model",
        f"{settings.embed_model} ({settings.embed_provider}, {settings.embed_dim}d)",
    )
    if exists:
        stored_dim = store.vector_dim()
        if stored_dim is not None and stored_dim != settings.embed_dim:
            table.add_row("[red]Stored vectors[/red]", f"[red]{stored_dim}d — mismatch[/red]")
    table.add_row("LLM", settings.llm_model)
    console.print(Panel(table, title="ResearchLens status", border_style="cyan"))

    if titles:
        console.print("Indexed documents:")
        for title in titles:
            console.print(f"  • {title}")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Drop the collection and everything indexed in it."""
    settings = _settings()
    store = _store(settings)

    if not yes and not typer.confirm(
        f"Drop collection '{settings.collection_name}' and all its chunks?"
    ):
        console.print("Cancelled.")
        return

    try:
        store.drop()
    except ResearchLensError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"[green]Dropped[/green] collection {settings.collection_name}")


# --------------------------------------------------------------------------- #
# Interactive menu (default entry point)
# --------------------------------------------------------------------------- #

MENU = """[bold]1[/bold]  Data ingestion — load PDFs from a folder into the vector database
[bold]2[/bold]  Retrieval — ask questions about the indexed papers
[bold]3[/bold]  Status — inspect the vector database
[bold]4[/bold]  Exit"""


def _menu() -> None:
    console.print(Panel(MENU, title=f"ResearchLens v{__version__}", border_style="cyan"))
    while True:
        try:
            choice = typer.prompt("Select an option", default="1").strip()
        except (EOFError, KeyboardInterrupt, typer.Abort):
            console.print()
            return

        if choice == "1":
            _run_ingestion(_prompt_for_folder())
        elif choice == "2":
            if _retrieval_from_menu():
                return
        elif choice == "3":
            status()
        elif choice in {"4", "exit", "quit", "q"}:
            return
        else:
            console.print("[yellow]Pick 1, 2, 3 or 4.[/yellow]")
            continue
        console.print()
        console.print(Panel(MENU, title="ResearchLens", border_style="cyan"))


def _retrieval_from_menu() -> bool:
    """Run a Q&A session from the menu. Returns True if the user asked to quit."""
    settings = _settings()
    store = _store(settings)
    return _ask_loop(RagEngine(settings, store), sources=True, from_menu=True)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO level logs."),
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    """Run the interactive menu when no subcommand is given."""
    _configure_logging(verbose)
    if version:
        console.print(f"researchlens {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _menu()


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
