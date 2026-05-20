"""Command-line interface for causaliq-discovery."""

from typing import Optional, Tuple

import click

from causaliq_discovery import __version__, learn_graph
from causaliq_discovery.registry import AlgorithmRegistry


def _parse_hyperparameters(
    ctx: click.Context,
    param: click.Parameter,
    value: Tuple[str, ...],
) -> Optional[dict]:
    """Parse key=value hyperparameter strings into a dict.

    Args:
        ctx: Click context (unused).
        param: Click parameter (unused).
        value: Tuple of ``"key=value"`` strings.

    Returns:
        Dict of parsed hyperparameters, or None if empty.

    Raises:
        click.BadParameter: If any entry is not in ``key=value``
            form.
    """
    if not value:
        return None
    result = {}
    for entry in value:
        if "=" not in entry:
            raise click.BadParameter(
                f"Expected key=value format, got '{entry}'.",
                param_hint="'-p'",
            )
        key, _, val = entry.partition("=")
        result[key.strip()] = val.strip()
    return result


@click.command(name="cqdisc")
@click.version_option(version=__version__)
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    metavar="FILE",
    help="CSV data file to learn from.",
)
@click.option(
    "-a",
    "--algorithm",
    required=True,
    metavar="NAME",
    help=(
        "Structure learning algorithm. "
        "Use '--help algorithm' to list supported names."
    ),
)
@click.option(
    "-o",
    "--output",
    required=True,
    metavar="PATH",
    help="Directory or workflow cache file for result output.",
)
@click.option(
    "-p",
    "--hyperparameter",
    "hyperparameters",
    multiple=True,
    metavar="KEY=VALUE",
    callback=_parse_hyperparameters,
    is_eager=False,
    expose_value=True,
    help=(
        "Hyperparameter as key=value. "
        "May be repeated, e.g. -p score=bdeu -p max_iterations=100."
    ),
)
@click.option(
    "-d",
    "--trace",
    is_flag=True,
    default=False,
    help="Include a step-by-step execution trace in the output.",
)
@click.option(
    "-T",
    "--variable-types",
    "variable_types",
    default=None,
    metavar="FILE",
    help="Network context JSON file defining variable types.",
)
@click.option(
    "-N",
    "--sample-size",
    "sample_size",
    default=None,
    type=int,
    metavar="N",
    help="Number of data rows to use.",
)
@click.option(
    "-V",
    "--variant",
    default=None,
    metavar="NAME",
    help=(
        "Algorithm variant, e.g. 'bnlearn' or 'causaliq'. "
        "Use '--help variant ALGORITHM' to list options."
    ),
)
@click.option(
    "-k",
    "--knowledge",
    default=None,
    metavar="FILE",
    help="Knowledge JSON file guiding structure learning.",
)
@click.option(
    "-r",
    "--randomise",
    multiple=True,
    metavar="OPTION",
    help=(
        "Randomise aspect of the input data. "
        "Supported: row_order, column_order, column_names, "
        "row_subsample. May be repeated."
    ),
)
@click.option(
    "-S",
    "--seed",
    default=None,
    type=int,
    metavar="N",
    help="Deterministic randomisation seed (0–1000).",
)
def cli(
    input_path: str,
    algorithm: str,
    output: str,
    hyperparameters: Optional[dict],
    trace: bool,
    variable_types: Optional[str],
    sample_size: Optional[int],
    variant: Optional[str],
    knowledge: Optional[str],
    randomise: tuple,
    seed: Optional[int],
) -> None:
    """Learn a causal graph from data.

    \b
    Examples:
      cqdisc -i data.csv -a tabu-stable -o results/
      cqdisc -i data.csv -a hc -o results/ -p score=bdeu -d
    """
    randomise_list = list(randomise) if randomise else None
    try:
        learn_graph(
            data=input_path,
            algorithm=algorithm,
            output=output,
            hyperparameters=hyperparameters,
            trace=trace,
            variable_types=variable_types,
            sample_size=sample_size,
            variant=variant,
            knowledge=knowledge,
            randomise=randomise_list,
            seed=seed,
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    except NotImplementedError as exc:
        raise click.ClickException(str(exc)) from exc


@click.group(name="cqdisc-help", invoke_without_command=True)
@click.argument("topic", required=False)
@click.argument("name", required=False)
def help_cli(topic: Optional[str], name: Optional[str]) -> None:
    """Display help on supported algorithms and variants.

    \b
    Topics:
      algorithm          List all supported algorithm names.
      variant ALGORITHM  List variants for a specific algorithm.
    """
    if topic == "algorithm":
        click.echo("Supported algorithms:")
        for alg in AlgorithmRegistry.algorithms():
            click.echo(f"  {alg}")
    elif topic == "variant" and name:
        try:
            variants = AlgorithmRegistry.variants(name)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        click.echo(f"Variants for '{name}':")
        for v in variants:
            click.echo(f"  {v}")
    else:
        click.echo(help_cli.get_help(click.Context(help_cli)))


def main() -> None:
    """Entry point for the cqdisc CLI."""
    cli(prog_name="cqdisc")


if __name__ == "__main__":  # pragma: no cover
    main()
