# CLI

## Overview

The command-line interface is exposed as the `cqdisc` command.

In addition to end-user commands such as `learn`, the CLI includes a
hidden contributor command, `generate-docs`, for regenerating algorithm
documentation pages from Jinja templates.

## Generate Algorithm Pages

Run this from the repository root:

```text
.\scripts\activate.ps1; cqdisc generate-docs
```

This command:

- renders one markdown page per registered algorithm under
    `docs/userguide/algorithms/`
- updates `docs/userguide/algorithms.md` when the index template is present
- updates the Algorithms navigation block in `mkdocs.yml`

Optional arguments:

- `--output-dir DIR` to write pages to a different directory
- `--mkdocs FILE` to explicitly set the mkdocs config path

Example:

```text
cqdisc generate-docs --output-dir docs/userguide/algorithms --mkdocs mkdocs.yml
```

## API Docs

::: causaliq_discovery.cli
        options:
                show_root_heading: true
                show_source: false
                heading_level: 3