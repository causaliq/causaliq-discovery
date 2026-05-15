# causaliq-discovery

![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

Advanced algorithms for learning causal structures from data and human & LLM-derived knowledge. This is part of the [CausalIQ ecosystem](https:https://causaliq.org/projects/)

## Status

🚧 **Active Development** - This repository is currently in active development, which involves:

- defining an intuitive, flexible, interface which supports all the algorithms supported by LLM and human knowledge.
- migrating score-based greedy algorithms from the legacy monolithic [discovery repo](https://github.com/causaliq/discovery)
- supporting bnlearn and Tetrad algorithms used in existing [CausalIQ papers](https://causaliq.org/papers/)
- extending the breadth and depth of algorithm coverage to cover: a larger selection of score, constraint and hybrid algorithms as well as continuous optimisation and time-series approaches 
- enabling all algorithms to be used within [CausalIQ Workflows](https://causaliq.org/projects/workflow/)

## Features

✅ **Implemented Releases**

- none

*See Git commit history for detailed implementation progress*

🛣️ Upcoming Releases

- **Release v1.0.0 Foundation**: Support for stable Tabu and HC, bnlearn and Tetrad/FGES without knowledge

- **Release v2.0.0 Knowledge**: Structure learning supported by simulated human experts

- **Release v3.0.0 More Algorithms**: Expand the range and type of algorithms supported


### Feature Overview

- 🧩 **Standardised API**: for running all structure learning algorithms
- 🧠 **Knowledge injection**: common framework for injecting knowledge into the learning process
- 🕵️ **Full reporting**: complete reporting of algorithm hypeparameters and learning process
- 🤖 **Workflows**: inclusion in causal discovery, analysis and inference workflows



### Usage

_to be defined_

---

## Integration with CausalIQ Ecosystem

- ⚙️ **CausalIQ Core** defines graph and BN objects
- 🔢 **CausalIQ Data** provides data objects that support randomisation and subsampling and capabilities to score graphs
- 🧠 **CausalIQ Knowledge** provides knowledge objects that guide or constrain the causal discovery
- 🤖 **CausalIQ Workflows**: causal discovery can be included in workflows
- 🔮 **CausalIQ Whatif** is called by this package to perform causal prediction.
- 🔄 **Zenodo Synchronisation** is used by this package to download datasets and upload results.
- 🧪 CausalIQ Papers are defined in terms of CausalIQ Workflows allowing the reproduction of experiments, results and published paper assets created by the CausalIQ ecosystem.

## LLM Support

The following provides project-specific context for this repo which should be provided after the [personal and ecosystem context](https://github.com/causaliq/causaliq/blob/main/LLM_DEVELOPMENT_GUIDE.md):

```text
I wish to migrate the code in legacy/core/metrics.py following all CausalIQ development guidelines
so that the legacy repo can use the migrated code instead. 
```

## Quick Start

```python
# to be completed
```

## Getting started

### Prerequisites

- Git 
- Latest stable versions of Python 3.9, 3.10. 3.11, 3.12 and 3.13.


### Clone the new repo locally and check that it works

Clone the causaliq-discovery repo locally as normal

```bash
git clone https://github.com/causaliq/causaliq-discovery.git
```

Set up the Python virtual environments and activate the default Python virtual environment. You may see
messages from VSCode (if you are using it as your IDE) that new Python environments are being created
as the scripts/setup-env runs - these messages can be safely ignored at this stage.

```text
scripts/setup-env -Install
scripts/activate
```

Check that the causaliq-discovery CLI is working, check that all CI tests pass, and start up the local mkdocs webserver. There should be no errors  reported in any of these.

```text
causaliq-newcapability --help
scripts/check_ci
mkdocs serve
```

Enter **http://127.0.0.1:8000/** in a browser and check that the 
causaliq-data documentation is visible.

If all of the above works, this confirms that the code is working successfully on your system.


## Documentation

Full API documentation is available at: **http://127.0.0.1:8000/** (when running `mkdocs serve`)

## Contributing

This repository is part of the CausalIQ ecosystem. For development setup:

1. Clone the repository
2. Run `scripts/setup-env -Install` to set up environments  
3. Run `scripts/check_ci` to verify all tests pass
4. Start documentation server with `mkdocs serve`

---

**Supported Python Versions**: 3.9, 3.10, 3.11, 3.12, 3.13
**Default Python Version**: 3.11  
**License**: MIT

