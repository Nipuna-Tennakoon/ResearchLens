"""CLI-level tests: the commands and the interactive menu, with fake backends."""

import pytest
from typer.testing import CliRunner

from researchlens.cli import app

runner = CliRunner()


@pytest.fixture
def cli(monkeypatch, settings, patched):
    monkeypatch.setattr("researchlens.cli._settings", lambda: settings)
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
