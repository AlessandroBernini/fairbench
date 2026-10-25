"""Tests for FairBench CLI."""

from click.testing import CliRunner

from fairbench.cli import cli


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "FairBench" in result.output


def test_run_command_help():
    """Test run command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_report_command_help():
    """Test report command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--help"])
    assert result.exit_code == 0
    assert "--latest" in result.output
