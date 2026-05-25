"""Unit tests for the cqdisc CLI."""

import pytest
from click.testing import CliRunner

from causaliq_discovery.cli import cli, help_cli, main
from causaliq_discovery.registry import AlgorithmRegistry


@pytest.fixture
def runner():
    return CliRunner()


# Version flag prints the package version.
def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# Help flag lists all options including required ones.
def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--algorithm" in result.output
    assert "--output" in result.output


# Missing required --input exits with error.
def test_cli_missing_input(runner):
    result = runner.invoke(cli, ["-a", "hc", "-o", "results"])
    assert result.exit_code != 0


# Missing required --algorithm exits with error.
def test_cli_missing_algorithm(runner):
    result = runner.invoke(cli, ["-i", "data.csv", "-o", "results"])
    assert result.exit_code != 0


# Missing required --output exits with error.
def test_cli_missing_output(runner):
    result = runner.invoke(cli, ["-i", "data.csv", "-a", "hc"])
    assert result.exit_code != 0


# Malformed hyperparameter raises UsageError.
def test_cli_bad_hyperparameter(runner):
    result = runner.invoke(
        cli,
        ["-i", "data.csv", "-a", "hc", "-o", "out", "-p", "noequals"],
    )
    assert result.exit_code != 0
    assert "key=value" in result.output.lower()


# Unknown algorithm name propagates as ClickException.
def test_cli_unknown_algorithm(runner):
    result = runner.invoke(
        cli,
        ["-i", "data.csv", "-a", "not-an-algo", "-o", "out"],
    )
    assert result.exit_code != 0


# Valid algorithm with no adapter raises ClickException (not UsageError).
def test_cli_valid_algorithm_raises_click_exception(runner, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError("not yet implemented"),
    )
    result = runner.invoke(
        cli,
        ["-i", "data.csv", "-a", "hc", "-o", "out"],
    )
    assert result.exit_code == 1
    assert "not yet implemented" in result.output.lower()


# Valid key=value hyperparameter is parsed successfully.
def test_cli_valid_hyperparameter_is_accepted(runner):
    result = runner.invoke(
        cli,
        [
            "-i",
            "data.csv",
            "-a",
            "hc",
            "-o",
            "out",
            "-p",
            "score=bic",
        ],
    )
    # Reaches learn_graph, which raises NotImplementedError → exit 1.
    assert result.exit_code == 1


# help_cli with 'algorithm' topic lists all algorithm names.
def test_help_cli_algorithm(runner):
    result = runner.invoke(help_cli, ["algorithm"])
    assert result.exit_code == 0
    assert "hc" in result.output
    assert "tabu-stable" in result.output


# help_cli with 'variant' topic lists variants for given algorithm.
def test_help_cli_variant(runner):
    result = runner.invoke(help_cli, ["variant", "hc-stable"])
    assert result.exit_code == 0
    assert "causaliq" in result.output


# help_cli with unknown algorithm under 'variant' exits with error.
def test_help_cli_variant_unknown_algorithm(runner):
    result = runner.invoke(help_cli, ["variant", "not-an-algo"])
    assert result.exit_code != 0


# help_cli with no topic displays help text.
def test_help_cli_no_topic(runner):
    result = runner.invoke(help_cli, [])
    assert result.exit_code == 0
    assert "algorithm" in result.output.lower()


# main() delegates to cli with prog_name set.
def test_main_delegates_to_cli(mocker):
    mock_cli = mocker.patch("causaliq_discovery.cli.cli")
    main()
    mock_cli.assert_called_once_with(prog_name="cqdisc")
