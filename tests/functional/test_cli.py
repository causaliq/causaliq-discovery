"""
Functional tests for the CLI.

These tests use Click's CliRunner to invoke the CLI commands

monkeypatch only works on curent process, so CLI runner must be invoked
using standalone=False
"""

import pandas as pd
from click.testing import CliRunner
from pytest import fixture

from causaliq_discovery.cli import cli

CLI_BASE_DIR = "tests/data/functional/cli"


# Provide a CLI runner for testing
@fixture
def cli_runner():
    return CliRunner()


# Test missing required --input argument exits with error.
def test_cli_missing_name_argument():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0  # Should fail
    assert "Missing argument" in result.output or "Usage:" in result.output


# Test no args shows usage information with required options listed.
def test_cli_no_args_shows_usage():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
    assert "--input" in result.output or "Usage:" in result.output


# Test CLI with a real CSV file exits with error for unknown algorithm.
def test_cli_with_csv_file_runs_until_not_implemented(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["-i", str(csv_path), "-a", "not-an-algo", "-o", "out"],
    )
    assert result.exit_code != 0
    assert "not-an-algo" in result.output.lower()


# Test CLI with a real CSV file and hyperparameter parses to dispatch.
def test_cli_with_csv_file_and_hyperparameter(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"X": [1, 0, 1], "Y": [0, 1, 0]})
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "-i",
            str(csv_path),
            "-a",
            "not-an-algo",
            "-o",
            "out",
            "-p",
            "alpha=0.01",
        ],
    )
    assert result.exit_code != 0
    assert "not-an-algo" in result.output.lower()


# Test that invoking script directly will run the CLI
def test_main_function(monkeypatch):
    called = {}

    def fake_cli(*args, **kwargs):
        called["cli"] = args != kwargs

    monkeypatch.setattr("causaliq_discovery.cli.cli", fake_cli)
    from causaliq_discovery.cli import main

    main()
    assert called.get("cli") is True
