"""Command-line interface for causaliq-discovery."""

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from jinja2 import Environment, FileSystemLoader

from causaliq_discovery import __version__, learn_graph
from causaliq_discovery.registry import AlgorithmRegistry, AlgorithmSpec


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


@click.group(name="cqdisc")
@click.version_option(version=__version__)
def cli() -> None:
    """causaliq-discovery: structure learning from data."""


@cli.command(name="learn")
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
        "Run 'cqdisc list-algorithms' to list supported names."
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
        "Run 'cqdisc describe ALGORITHM' to see available variants."
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
def learn_cmd(
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
      cqdisc learn -i data.csv -a tabu-stable -o results/
      cqdisc learn -i data.csv -a hc -o results/ -p score=bdeu -d
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


@cli.command(name="list-algorithms")
def list_algorithms_cmd() -> None:
    """List all supported structure learning algorithms."""
    algorithms = AlgorithmRegistry.algorithms()
    name_w = max(len(a) for a in algorithms) + 2
    class_w = 11  # len("constraint") + 1
    click.echo("Supported algorithms:")
    for alg in algorithms:
        spec = AlgorithmRegistry.get_spec(alg, None)
        col_name = alg.ljust(name_w)
        col_class = spec.algorithm_class.ljust(class_w)
        click.echo(f"  {col_name}{col_class}{spec.description}")


def _format_describe(spec: AlgorithmSpec) -> str:
    """Format algorithm description for terminal display.

    Args:
        spec: AlgorithmSpec to describe.

    Returns:
        Formatted multi-line string for terminal output.
    """
    lines: List[str] = []
    lines.append(f"Algorithm:  {spec.algorithm}  \u2013  {spec.description}")
    lines.append(
        f"Class:      {spec.algorithm_class}  \u00b7  {spec.graph_type}"
    )
    lines.append(f"Package:    {spec.package}")
    all_variants = AlgorithmRegistry.variants(spec.algorithm)
    if len(all_variants) > 1:
        lines.append(f"Variants:   {', '.join(all_variants)}")

    if spec.paper_ref:
        lines.append("")
        wrapped = textwrap.fill(
            f"Paper:  {spec.paper_ref}",
            width=79,
            subsequent_indent="        ",
        )
        lines.append(wrapped)
        if spec.paper_url:
            lines.append(f"        {spec.paper_url}")

    hp_names = sorted(spec.supported_hyperparameters)
    lines.append("")
    lines.append("Hyperparameters:")
    if not hp_names:
        lines.append("  (none)")
    else:
        name_w = max(len(n) for n in hp_names) + 2
        type_w = 7
        indent = " " * (2 + name_w + type_w)
        for hp_name in hp_names:
            try:
                hp = AlgorithmRegistry.get_hyperparameter_spec(hp_name)
                hp_type = hp.type
                desc = hp.description
                valid = hp.valid_values
                hp_default_display = hp.default_display
            except KeyError:
                hp_type, desc, valid = "?", "", None
                hp_default_display = None
            default = spec.hyperparameter_defaults.get(hp_name)
            col_n = hp_name.ljust(name_w)
            col_t = hp_type.ljust(type_w)
            lines.append(f"  {col_n}{col_t}{desc}")
            if default is not None:
                lines.append(f"{indent}Default: {default}")
            elif hp_default_display is not None:
                lines.append(f"{indent}Default: {hp_default_display}")
            if valid:
                vals = ", ".join(str(v) for v in valid)
                lines.append(f"{indent}Values:  {vals}")

    algo_slug = spec.algorithm
    lines.append("")
    lines.append(
        "Docs:   https://causaliq.github.io/causaliq-discovery/"
        f"userguide/algorithms/{algo_slug}/"
    )
    return "\n".join(lines)


@cli.command(name="describe")
@click.argument("algorithm")
@click.option(
    "--variant",
    default=None,
    metavar="NAME",
    help=("Variant to describe. Defaults to the first registered " "variant."),
)
def describe_cmd(algorithm: str, variant: Optional[str]) -> None:
    """Show description, hyperparameters, and reference for ALGORITHM."""
    try:
        spec = AlgorithmRegistry.get_spec(algorithm, variant)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(_format_describe(spec))


def _algorithm_title(algorithm: str) -> str:
    """Return a title-cased display name for an algorithm slug.

    Args:
        algorithm: Hyphen-separated algorithm identifier, e.g. ``hc-stable``.

    Returns:
        Display name, e.g. ``HC-Stable``.
    """
    return "-".join(part.upper() for part in algorithm.split("-"))


def _build_template_context(
    spec: AlgorithmSpec,
    description_fragment: str,
) -> Dict[str, Any]:
    """Build the Jinja2 template context for one algorithm page.

    Args:
        spec: AlgorithmSpec for the default (first) variant.
        description_fragment: Raw markdown text from the description
            fragment file.

    Returns:
        Dictionary of template variables.
    """
    hp_rows = []
    for hp_name in sorted(spec.supported_hyperparameters):
        try:
            hp = AlgorithmRegistry.get_hyperparameter_spec(hp_name)
            hp_type = hp.type
            hp_desc = hp.description
            hp_values = (
                ", ".join(f"`{v}`" for v in hp.valid_values)
                if hp.valid_values
                else "—"
            )
            raw_default = spec.hyperparameter_defaults.get(hp_name)
            if raw_default is not None:
                hp_default = str(raw_default)
            elif hp.default_display is not None:
                hp_default = hp.default_display
            else:
                hp_default = "—"
        except KeyError:
            hp_type, hp_desc, hp_values, hp_default = "?", "", "—", "—"
        hp_rows.append(
            {
                "name": hp_name,
                "type": hp_type,
                "default": hp_default,
                "values": hp_values,
                "description": hp_desc,
            }
        )

    variant_rows = [
        {
            "variant": vs.variant,
            "package": vs.package,
        }
        for vs in (
            AlgorithmRegistry.get_spec(spec.algorithm, v)
            for v in AlgorithmRegistry.variants(spec.algorithm)
        )
    ]

    paper_ref_wrapped = (
        textwrap.fill(spec.paper_ref, width=79) if spec.paper_ref else ""
    )

    return {
        "title": _algorithm_title(spec.algorithm),
        "spec": spec,
        "description_fragment": description_fragment.strip(),
        "hyperparameters": hp_rows,
        "variants": variant_rows,
        "paper_ref_wrapped": paper_ref_wrapped,
    }


@cli.command(name="generate-docs")
@click.option(
    "--output-dir",
    default=None,
    metavar="DIR",
    help=(
        "Directory to write algorithm pages into. "
        "Defaults to docs/userguide/algorithms/ relative to the "
        "current working directory."
    ),
)
def generate_docs_cmd(output_dir: Optional[str]) -> None:
    """Generate a user-guide markdown page for each registered algorithm."""
    cwd = Path.cwd()
    out_path = (
        Path(output_dir)
        if output_dir
        else (cwd / "docs" / "userguide" / "algorithms")
    )
    template_dir = out_path
    descriptions_dir = out_path / "_descriptions"
    template_name = "_template.md.j2"

    if not (template_dir / template_name).exists():
        raise click.ClickException(
            f"Template not found: {template_dir / template_name}"
        )

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template(template_name)

    out_path.mkdir(parents=True, exist_ok=True)

    for algorithm in AlgorithmRegistry.algorithms():
        spec = AlgorithmRegistry.get_spec(algorithm, None)
        desc_file = descriptions_dir / f"{algorithm}.md"
        if desc_file.exists():
            description_fragment = desc_file.read_text(encoding="utf-8")
        else:
            description_fragment = (
                f"*No description available for {algorithm}.*"
            )

        context = _build_template_context(spec, description_fragment)
        rendered = template.render(**context)
        out_file = out_path / f"{algorithm}.md"
        out_file.write_text(rendered, encoding="utf-8")
        try:
            display = out_file.relative_to(cwd)
        except ValueError:
            display = out_file
        click.echo(f"  wrote {display}")

    click.echo(
        f"Generated {len(AlgorithmRegistry.algorithms())} algorithm pages."
    )


def main() -> None:
    """Entry point for the cqdisc CLI."""
    cli(prog_name="cqdisc")


if __name__ == "__main__":  # pragma: no cover
    main()
