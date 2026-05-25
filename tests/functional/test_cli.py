"""
Functional tests for the CLI.

These tests use Click's CliRunner to invoke the CLI commands

monkeypatch only works on curent process, so CLI runner must be invoked
using standalone=False
"""

import subprocess
import sys

import pandas as pd
import pytest
from click.testing import CliRunner
from pytest import fixture

from causaliq_discovery.cli import cli

_ALL_ALGORITHMS = [
    "gs",
    "h2pc",
    "hc",
    "hc-stable",
    "iiamb",
    "mmhc",
    "pc-stable",
    "tabu",
    "tabu-stable",
]


# Provide a CLI runner for testing
@fixture
def cli_runner():
    return CliRunner()


# Test learn with no args exits with error showing missing options.
def test_cli_missing_name_argument():
    runner = CliRunner()
    result = runner.invoke(cli, ["learn"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Usage:" in result.output


# Test no args shows group usage with subcommands listed.
def test_cli_no_args_shows_usage():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert "learn" in result.output or "Usage:" in result.output


# Test CLI with a real CSV file exits with error for unknown algorithm.
def test_cli_with_csv_file_runs_until_not_implemented(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "learn",
            "-i",
            str(csv_path),
            "-a",
            "not-an-algo",
            "-o",
            "out",
        ],
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
            "learn",
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


# CLI learn dispatches each algorithm to learn_graph when mock active.
@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_cli_dispatches_to_learn_graph_for_algorithm(
    tmp_path,
    mocker,
    algorithm: str,
) -> None:
    mocker.patch("causaliq_discovery.cli.learn_graph")
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["learn", "-i", str(csv_path), "-a", algorithm, "-o", "out"],
    )
    assert result.exit_code == 0


# --variant flag is forwarded as variant kwarg to learn_graph.
def test_cli_variant_flag_forwarded_to_learn_graph(tmp_path, mocker) -> None:
    mock_learn = mocker.patch("causaliq_discovery.cli.learn_graph")
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(csv_path, index=False)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "learn",
            "-i",
            str(csv_path),
            "-a",
            "hc",
            "-o",
            "out",
            "-V",
            "bnlearn",
        ],
    )
    assert mock_learn.call_args.kwargs["variant"] == "bnlearn"


# cqdisc list-algorithms lists all algorithms via subprocess.
def test_list_algorithms_via_subprocess() -> None:
    script = (
        "from causaliq_discovery.cli import main; "
        "import sys; "
        "sys.argv = ['cqdisc', 'list-algorithms']; "
        "main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for algo in _ALL_ALGORITHMS:
        assert algo in result.stdout


# describe shows algorithm name, hyperparameters and docs for each algo.
@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_describe_shows_info_for_algorithm(algorithm: str) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["describe", algorithm])
    assert result.exit_code == 0
    assert f"Algorithm:  {algorithm}" in result.output
    assert "Class:" in result.output
    assert "Hyperparameters:" in result.output
    assert "Docs:" in result.output


# generate-docs writes one markdown file per algorithm to output dir.
def test_generate_docs_produces_all_algorithm_pages(tmp_path) -> None:
    import shutil
    from pathlib import Path

    from causaliq_discovery.registry import AlgorithmRegistry

    docs_dir = Path(__file__).parents[2] / "docs" / "userguide" / "algorithms"
    shutil.copy(docs_dir / "_template.md.j2", tmp_path / "_template.md.j2")
    descriptions_dst = tmp_path / "_descriptions"
    if (docs_dir / "_descriptions").exists():
        shutil.copytree(docs_dir / "_descriptions", descriptions_dst)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["generate-docs", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    for algo in AlgorithmRegistry.algorithms():
        page = tmp_path / f"{algo}.md"
        assert page.exists(), f"Missing page for {algo}"
        content = page.read_text(encoding="utf-8")
        assert f"# {algo.upper()}" in content.upper()
        assert "## Overview" in content
        assert "## Hyperparameters" in content
        assert "## Variants" in content
