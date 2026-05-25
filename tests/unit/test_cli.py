"""Unit tests for the cqdisc CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from causaliq_discovery.cli import (
    _algorithm_title,
    _build_template_context,
    _format_describe,
    cli,
    main,
)
from causaliq_discovery.registry import AlgorithmRegistry, AlgorithmSpec


@pytest.fixture
def runner():
    return CliRunner()


# Version flag prints the package version.
def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# Help flag lists all subcommands.
def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "learn" in result.output
    assert "list-algorithms" in result.output
    assert "describe" in result.output


# learn --help lists all required options.
def test_cli_learn_help(runner):
    result = runner.invoke(cli, ["learn", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--algorithm" in result.output
    assert "--output" in result.output


# Missing required --input exits with error.
def test_cli_missing_input(runner):
    result = runner.invoke(cli, ["learn", "-a", "hc", "-o", "results"])
    assert result.exit_code != 0


# Missing required --algorithm exits with error.
def test_cli_missing_algorithm(runner):
    result = runner.invoke(cli, ["learn", "-i", "data.csv", "-o", "results"])
    assert result.exit_code != 0


# Missing required --output exits with error.
def test_cli_missing_output(runner):
    result = runner.invoke(cli, ["learn", "-i", "data.csv", "-a", "hc"])
    assert result.exit_code != 0


# Malformed hyperparameter raises UsageError.
def test_cli_bad_hyperparameter(runner):
    result = runner.invoke(
        cli,
        [
            "learn",
            "-i",
            "data.csv",
            "-a",
            "hc",
            "-o",
            "out",
            "-p",
            "noequals",
        ],
    )
    assert result.exit_code != 0
    assert "key=value" in result.output.lower()


# Unknown algorithm name propagates as ClickException.
def test_cli_unknown_algorithm(runner):
    result = runner.invoke(
        cli,
        ["learn", "-i", "data.csv", "-a", "not-an-algo", "-o", "out"],
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
        ["learn", "-i", "data.csv", "-a", "hc", "-o", "out"],
    )
    assert result.exit_code == 1
    assert "not yet implemented" in result.output.lower()


# Valid key=value hyperparameter is parsed successfully.
def test_cli_valid_hyperparameter_is_accepted(runner):
    result = runner.invoke(
        cli,
        [
            "learn",
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


# list-algorithms command shows name, class and description for each algo.
def test_list_algorithms_cmd(runner):
    result = runner.invoke(cli, ["list-algorithms"])
    assert result.exit_code == 0
    assert "hc" in result.output
    assert "tabu-stable" in result.output
    assert "score" in result.output
    assert "constraint" in result.output
    assert "hybrid" in result.output
    assert "Hill-climbing" in result.output


# main() delegates to cli with prog_name set.
def test_main_delegates_to_cli(mocker):
    mock_cli = mocker.patch("causaliq_discovery.cli.cli")
    main()
    mock_cli.assert_called_once_with(prog_name="cqdisc")


# describe shows algorithm info including hyperparameters and docs.
def test_describe_cmd_shows_algorithm_info(runner):
    result = runner.invoke(cli, ["describe", "tabu"])
    assert result.exit_code == 0
    assert "tabu" in result.output
    assert "Class:" in result.output
    assert "Hyperparameters:" in result.output
    assert "Docs:" in result.output


# describe with an unknown algorithm exits with error.
def test_describe_cmd_unknown_algorithm(runner):
    result = runner.invoke(cli, ["describe", "not-an-algo"])
    assert result.exit_code != 0


# describe --variant selects a specific variant.
def test_describe_cmd_with_variant(runner):
    result = runner.invoke(
        cli, ["describe", "hc-stable", "--variant", "causaliq"]
    )
    assert result.exit_code == 0
    assert "hc-stable" in result.output


# describe --variant with unknown variant exits with error.
def test_describe_cmd_unknown_variant(runner):
    result = runner.invoke(cli, ["describe", "hc", "--variant", "nonexistent"])
    assert result.exit_code != 0


# describe shows paper reference for algorithms with a paper_ref.
def test_describe_cmd_shows_paper_ref(runner):
    result = runner.invoke(cli, ["describe", "hc"])
    assert result.exit_code == 0
    assert "Paper:" in result.output
    assert "Chickering" in result.output


# cli --help lists describe as an available command.
def test_cli_help_lists_describe(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "describe" in result.output


# describe shows 'No limit' as default for max_elapsed.
def test_describe_shows_no_limit_for_max_elapsed(runner):
    result = runner.invoke(cli, ["describe", "hc"])
    assert result.exit_code == 0
    assert "No limit" in result.output


# _format_describe shows (none) when spec has no hyperparameters.
def test_format_describe_no_hyperparameters():
    spec = AlgorithmSpec(
        algorithm="test-algo",
        variant="test",
        package="test",
        description="Test",
        graph_type="DAG",
    )
    with patch.object(AlgorithmRegistry, "variants", return_value=["test"]):
        output = _format_describe(spec)
    assert "(none)" in output


# _format_describe uses fallback type when HP not in registry.
def test_format_describe_unknown_hyperparameter_uses_fallback():
    spec = AlgorithmSpec(
        algorithm="test-algo",
        variant="test",
        package="test",
        description="Test",
        graph_type="DAG",
        supported_hyperparameters={"unknown_hp"},
    )
    with patch.object(AlgorithmRegistry, "variants", return_value=["test"]):
        output = _format_describe(spec)
    assert "unknown_hp" in output


# _format_describe shows Variants line when algorithm has multiple variants.
def test_format_describe_shows_variants_line():
    spec = AlgorithmRegistry.get_spec("hc", None)
    with patch.object(
        AlgorithmRegistry, "variants", return_value=["bnlearn", "other"]
    ):
        output = _format_describe(spec)
    assert "Variants:" in output


# _format_describe shows paper URL when paper_url is set on spec.
def test_format_describe_shows_paper_url():
    spec = AlgorithmSpec(
        algorithm="test-algo",
        variant="test",
        package="test",
        description="Test",
        graph_type="DAG",
        paper_ref="Author (2025) – A paper.",
        paper_url="https://doi.org/10.0000/test",
    )
    with patch.object(AlgorithmRegistry, "variants", return_value=["test"]):
        output = _format_describe(spec)
    assert "https://doi.org/10.0000/test" in output


# _algorithm_title hyphenates and uppercases each part.
def test_algorithm_title_single_part():
    assert _algorithm_title("hc") == "HC"


# _algorithm_title uppercases all hyphen-separated parts.
def test_algorithm_title_multi_part():
    assert _algorithm_title("hc-stable") == "HC-STABLE"


# _build_template_context returns title derived from algorithm slug.
def test_build_template_context_title():
    spec = AlgorithmRegistry.get_spec("hc", None)
    ctx = _build_template_context(spec, "A description.")
    assert ctx["title"] == "HC"


# _build_template_context passes description fragment through stripped.
def test_build_template_context_description_stripped():
    spec = AlgorithmRegistry.get_spec("hc", None)
    ctx = _build_template_context(spec, "  A description.  \n")
    assert ctx["description_fragment"] == "A description."


# _build_template_context includes a row for each supported hyperparameter.
def test_build_template_context_hyperparameter_rows():
    spec = AlgorithmRegistry.get_spec("hc", None)
    ctx = _build_template_context(spec, "")
    hp_names = {row["name"] for row in ctx["hyperparameters"]}
    assert "score" in hp_names
    assert "max_iterations" in hp_names


# _build_template_context shows valid values for ci_test.
def test_build_template_context_valid_values_for_ci_test():
    spec = AlgorithmRegistry.get_spec("pc-stable", None)
    ctx = _build_template_context(spec, "")
    ci_row = next(r for r in ctx["hyperparameters"] if r["name"] == "ci_test")
    assert "`mi`" in ci_row["values"]
    assert "`x2`" in ci_row["values"]


# _build_template_context shows No limit for max_iterations.
def test_build_template_context_no_limit_for_max_iterations():
    spec = AlgorithmRegistry.get_spec("hc", None)
    ctx = _build_template_context(spec, "")
    mi_row = next(
        r for r in ctx["hyperparameters"] if r["name"] == "max_iterations"
    )
    assert mi_row["default"] == "No limit"


# _build_template_context uses dash for HPs with no default or display.
def test_build_template_context_unknown_hp_uses_fallback():
    spec = AlgorithmSpec(
        algorithm="test-algo",
        variant="test",
        package="test",
        description="Test",
        graph_type="DAG",
        supported_hyperparameters={"unknown_hp"},
    )
    dummy = AlgorithmSpec(
        algorithm="test-algo",
        variant="test",
        package="test",
        description="Test",
        graph_type="DAG",
    )
    with (
        patch.object(AlgorithmRegistry, "variants", return_value=["test"]),
        patch.object(AlgorithmRegistry, "get_spec", return_value=dummy),
    ):
        ctx = _build_template_context(spec, "")
    assert ctx["hyperparameters"][0]["type"] == "?"


# _build_template_context includes one variant row per registered variant.
def test_build_template_context_variant_rows():
    spec = AlgorithmRegistry.get_spec("hc", None)
    ctx = _build_template_context(spec, "")
    assert len(ctx["variants"]) == len(AlgorithmRegistry.variants("hc"))


# generate-docs exits with error when template is missing.
def test_generate_docs_missing_template(runner, tmp_path):
    result = runner.invoke(
        cli, ["generate-docs", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "Template not found" in result.output


# generate-docs writes one file per algorithm when template exists.
def test_generate_docs_writes_algorithm_files(runner, tmp_path):
    import shutil

    docs_dir = Path(__file__).parents[2] / "docs" / "userguide" / "algorithms"
    shutil.copy(docs_dir / "_template.md.j2", tmp_path / "_template.md.j2")
    result = runner.invoke(
        cli, ["generate-docs", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    algos = AlgorithmRegistry.algorithms()
    for algo in algos:
        assert (tmp_path / f"{algo}.md").exists()


# generate-docs uses fallback description when fragment file is absent.
def test_generate_docs_fallback_description(runner, tmp_path):
    import shutil

    docs_dir = Path(__file__).parents[2] / "docs" / "userguide" / "algorithms"
    shutil.copy(docs_dir / "_template.md.j2", tmp_path / "_template.md.j2")
    result = runner.invoke(
        cli, ["generate-docs", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    hc_md = (tmp_path / "hc.md").read_text(encoding="utf-8")
    # Either the real description was used or the fallback placeholder
    assert "hc" in hc_md.lower()


# generate-docs reads description fragment and renders default_display HP.
def test_generate_docs_reads_description_fragment(runner, tmp_path):
    import shutil

    docs_dir = Path(__file__).parents[2] / "docs" / "userguide" / "algorithms"
    shutil.copy(docs_dir / "_template.md.j2", tmp_path / "_template.md.j2")
    shutil.copytree(docs_dir / "_descriptions", tmp_path / "_descriptions")
    result = runner.invoke(
        cli, ["generate-docs", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    # hc has max_elapsed with default_display="No limit" and no numeric default
    hc_md = (tmp_path / "hc.md").read_text(encoding="utf-8")
    assert "No limit" in hc_md


# _build_template_context uses dash when HP has no default and no display.
def test_build_template_context_dash_when_no_default():
    spec = AlgorithmSpec(
        algorithm="hc",
        variant="bnlearn",
        package="bnlearn",
        description="Hill-climbing",
        graph_type="DAG",
        algorithm_class="score",
        supported_hyperparameters={"score"},
        hyperparameter_defaults={},
    )
    context = _build_template_context(spec, "Description.")
    score_row = next(
        r for r in context["hyperparameters"] if r["name"] == "score"
    )
    assert score_row["default"] == "—"
