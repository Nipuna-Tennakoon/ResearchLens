"""CLI-level tests: the commands and the interactive menu, with fake backends."""

import pytest
from typer.testing import CliRunner

from researchlens.cli import app

runner = CliRunner()


@pytest.fixture
def cli(monkeypatch, settings, patched):
    # Patch the raw loader so startup preloading sees the same settings too.
    monkeypatch.setattr("researchlens.cli._load_settings", lambda: settings)
    return settings


def test_ingest_then_status_then_ask(cli, sample_pdf):
    result = runner.invoke(app, ["ingest", str(sample_pdf.parent)])
    assert result.exit_code == 0, result.output
    assert "Ingestion complete" in result.output

    # A fresh process would lose the in-memory fake store, so assert within one run.
    result = runner.invoke(app, ["ask", "What parameters does SARIMA use?"])
    assert result.exit_code == 0, result.output
    assert "Answer" in result.output
    assert "Sources" in result.output


def test_ingest_reports_a_missing_folder(cli, tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "Folder not found" in result.output


def test_ingest_prompts_for_the_path_when_omitted(cli, sample_pdf):
    result = runner.invoke(app, ["ingest"], input=f"{sample_pdf.parent}\n")

    assert result.exit_code == 0, result.output
    assert "Path to the data folder" in result.output


def test_status_on_an_empty_database(cli):
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "ResearchLens status" in result.output


def test_reset_requires_confirmation(cli):
    result = runner.invoke(app, ["reset"], input="n\n")

    assert result.exit_code == 0
    assert "Cancelled" in result.output


def test_ask_without_ingestion_explains_what_to_do(cli):
    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "Run data ingestion first" in result.output


def test_menu_runs_ingestion_then_exits(cli, sample_pdf):
    result = runner.invoke(app, [], input=f"1\n{sample_pdf.parent}\n4\n")

    assert result.exit_code == 0, result.output
    assert "Data ingestion" in result.output
    assert "Ingestion complete" in result.output


def test_menu_rejects_an_unknown_option(cli):
    result = runner.invoke(app, [], input="9\n4\n")

    assert result.exit_code == 0
    assert "Pick 1, 2, 3 or 4." in result.output


def test_version_flag(cli):
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "researchlens" in result.output


def _menu_renders(output: str) -> int:
    return output.count("Status — inspect the vector database")


def test_exit_in_question_mode_leaves_the_application(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    result = runner.invoke(app, [], input="2\nexit\n")

    assert result.exit_code == 0, result.output
    assert "Bye." in result.output
    # The menu must not come back after 'exit'.
    assert _menu_renders(result.output) == 1


def test_back_in_question_mode_returns_to_the_menu(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    result = runner.invoke(app, [], input="2\nback\n4\n")

    assert result.exit_code == 0, result.output
    assert _menu_renders(result.output) == 2


def test_quit_and_q_also_exit(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    for word in ("quit", "q"):
        result = runner.invoke(app, [], input=f"2\n{word}\n")
        assert result.exit_code == 0, result.output
        assert _menu_renders(result.output) == 1


def test_exit_ends_the_standalone_ask_command(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    result = runner.invoke(app, ["ask"], input="exit\n")

    assert result.exit_code == 0, result.output
    assert "Bye." in result.output


def test_end_of_input_leaves_the_application(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    result = runner.invoke(app, [], input="2\n")  # no further input: EOF

    assert result.exit_code == 0, result.output
    assert _menu_renders(result.output) == 1


def test_the_question_prompt_advertises_exit(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    result = runner.invoke(app, [], input="2\nexit\n")

    assert "'exit' to quit" in result.output
    assert "'back'" in result.output


def test_a_question_is_still_answered_before_exiting(cli, sample_pdf):
    runner.invoke(app, ["ingest", str(sample_pdf.parent)])

    result = runner.invoke(app, [], input="2\nWhat is SARIMA?\nexit\n")

    assert result.exit_code == 0, result.output
    assert "Answer" in result.output
    assert "Bye." in result.output


def test_ingest_then_ask_loads_the_model_only_once(cli, sample_pdf, patched):
    from conftest import BuildCounter

    runner.invoke(app, ["ingest", str(sample_pdf.parent)])
    runner.invoke(app, ["ask", "What is SARIMA?"])

    assert BuildCounter.calls == 1


def test_repeated_retrieval_from_the_menu_reuses_the_model(cli, sample_pdf, patched):
    from conftest import BuildCounter

    runner.invoke(app, ["ingest", str(sample_pdf.parent)])
    # Enter retrieval, go back to the menu, enter it again, then quit.
    result = runner.invoke(app, [], input="2\nback\n2\nWhat is SARIMA?\nexit\n")

    assert result.exit_code == 0, result.output
    assert BuildCounter.calls == 1


def test_commands_that_never_embed_do_not_load_the_model(cli, patched):
    from conftest import BuildCounter

    runner.invoke(app, ["status"])
    runner.invoke(app, ["reset"], input="n\n")

    assert BuildCounter.calls == 0
