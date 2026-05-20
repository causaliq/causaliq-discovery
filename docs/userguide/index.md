# CausalIQ Discovery User Guide

The causaliq-discovery package provides advanced algorithms for learning causal structures from data and human & LLM-derived knowledge. The initial focus is on learning graphs from time-independent observational data alone, but this will be extended to include knowledge-based guidance, interventional and time-series data.


## Getting Started

### From the command line

This is a minimal command line invocation with the Tabu-Stable algorithm learning from a CSV data file and producing results in folder "results" which includes the learnt graph in GraphML format. Default algorithm hyperparameters, including the score would be used.

```bash
cqdisc -i data.csv -a tabu-stable -o results
```


### From Python code

This is the same simple learning process implemented in Python.

```python
from causaliq_discovery import learn_graph

# Simple call to learn a graph from a CSV data file using default parameters

result = learn_graph(data='data.csv', algorithm='tabu-stable')

# result is a DiscoveryResult which includes the learnt graph, diagnostics
# which helps a human or LLM understand the learning process, and details
# of all default and explicit hyperparameters used
```


### As a CausalIQ Workflow Action

Causal Discovery may be included in [CausalIQ Workflows](https://causaliq.github.io/causaliq-workflow/userguide/), an approach which provides a number of benefits:

 * Causal discovery can be part of a **pipeline**, for instance, where LLM knowledge provides a prior graph, causal discovery is performed, and the resulting learnt graph is analysed, and Bayesian Network is constructed from it which is used for causal inference.
 * CausalIQ Workflows facilitate repeatedly calling causal discovery with different
 algorithms, networks, data files or hyperparameter values facilitating research **sensitivity analyses**
 * Algorithm and score hyperparameters can be specified in an **intuitive, transparent** way

The figure below shows a causal discovery action that could be embedded in
a CausalIQ Workflow. There are several points to note:

* although not done in this example, the input, algorithm, and hyperparameters can all
be parameterised to support wide ranging sensitivity analysis
* in this case, the output is a Workflow Cache


```yaml
steps:
  - name: "Learn Graph from Data"
    uses: "causaliq-discovery"
    with:
      action: "learn_graph"
      input: "data.csv"
      algorithm: "tabu-stable"
      output: "results/learnt_graphs.db"
      hyperparameters: 
        score: "bdeu"
        iss: 5
        max_iterations: 5
```

## Parameters

The table below gives an overview of the supported parameters - which are supported by
the CLI, Python and Workflow Action interfaces. Parameter names are the same across all three interfaces. Additionally the specified CLI shortform flag can be used in the CLI.

| Parameter                                       | CLI Flag | Required    | Description |
|-------------------------------------------------|----------|-------------|-------------|
| [`input`](#input-parameter)                     | `-i`     | Yes         | Data file to learn from |
| [`algorithm`](#algorithm-parameter)             | `-a`     | Yes         | Structure learning algorithm to use |
| [`output`](#output-parameter)                   | `-o`     | (see below) | Output folder/database where results will be placed, including the learnt graph |
| [`hyperparameters`](#hyperparameters-parameter) | `-p`     | No          | Structure learning hyperparameters |
| [`trace`](#trace-parameter)                     | `-d`     | No          | Whether a detailed structure learning trace should be included in the result output |
| [`variable_types`](#variable_types-parameter)   | `-T`     | No          | Information regarding the data variable types (`CONTINUOUS`, `DISCRETE` etc.) |
| [`sample_size`](#sample_size-parameter)         | `-N`     | No          | Sample size to learn from |
| [`variant`](#variant-parameter)                 | `-V`     | No          | Variant of a algorithm if there are several, for instance,  provided in different packages |
| [`knowledge`](#knowledge-parameter)             | `-k`     | No          | Knowledge which guides or constrains the learning process, for instance required or forbidden arcs |
| [`randomise`](#randomise-parameter)             | `-r`     | No          | Properties of the dataset to randomise: column names, column order, row order and/or row selection |
| [`seed`](#seed-parameter)                       | `-S`     | No          | Randomise seed number |


### `input` parameter

When run from the CLI or an action this is a CSV file with the first row being the variable names. If the Python function is being used this can be:

 * a string of a CSV file containing the data
 * a Pandas data frame
 * a [CausalIQ Data object](https://causaliq.github.io/causaliq-data/)


### `algorithm` parameter

This mandatory parameter is the algorithm name. Supported algorithm names can be inspected from the command line using the command `cqdisc --help algorithm`. Details of the supported algorithms are provided in the [Algorithms section](./algorithms.md).


### `output` parameter

This must be one of: 

- an existing directory name where the structure learning results will be placed as individual files
- the filename of a CausalIQ Workflow cache, where the results will be stored as a cache entry
- `None` value indicating that results will not be stored to disk

This parameter is mandatory except on the Python `learn_graph` method where a default `None` value is provided. Additionally,  `learn_graph` method return is a `DiscoveryResult` object. Whichever form(s) the result is returned in it will contain the following elements:

- the **learnt graph** - either a DAG, PDAG, CPDAG, MAG or PAG depending upon the algorithm used.
- **structure learning metadata** which provides full details of the structure learning run, including the algorithm, date and time of execution and its elapsed time and *all* relevant hyperparameters - whether specified explictly using the `hyperparameters` parameter, or whether default values specific to each algorithm.
- if the `trace` parameter is set true, then the result will also contain an **execution trace** which provide full detail of the steps the algorithm took in creating the learnt graph, sufficient for a human or LLM to understand how the properties of the learnt graph arose from the data.

The format of these elements will be dependent on the form that the result is returned in - a file, element within a Workflow Cache object or `DiscoveryResult` Python object as follows:

|                 | File          | Workflow cache            | `DiscoveryResult` |
|-----------------|---------------|---------------------------|-------------------|
| Learnt Graph    | .graphML file | compressed `SDG` object   | [`SDG` object](https://causaliq.github.io/causaliq-core/api/graph/)    |
| Metadata        | JSON file     | compressed JSON           | Python `dict`     |
| Execution trace | JSON file     | compressed JSON           | Python `list[dict]` |

#### Execution trace format

The trace is a JSON object containing a `trace_type` field identifying the schema, and a `steps` array with one entry per algorithm step. The `trace_type` allows consumers (including LLMs) to interpret the steps correctly without needing access to the accompanying metadata.

For score-based algorithms (`hc`, `hc-stable`, `tabu`, `tabu-stable`) the trace type is `score_steps` and each step contains:

| Field | Description |
|-------|-------------|
| `time` | Elapsed time in seconds at this step |
| `arc_change` | The arc addition, removal or reversal chosen at this step, e.g. `"A→B"` |
| `score_increase` | The score improvement achieved by this change |
| `alternative_arc_change` | The next-best arc change considered |
| `alternative_score_increase` | The score improvement that alternative would have achieved |

Example:

```json
{
  "trace_type": "score_steps",
  "steps": [
    {
      "time": 0.12,
      "arc_change": "Age→Cancer",
      "score_increase": 1.34,
      "alternative_arc_change": "Smoking→Cancer",
      "alternative_score_increase": 0.87
    }
  ]
}
```

Trace formats for other algorithm types (constraint-based, hybrid) will be defined when those algorithms are added.


### `hyperparameters` parameter

 This optional parameter defines the structure learning hyperparameters, such as the score used, a limit on the number of iterations, the BDeu Imaginary Sample Size parameter and so on. Hyperparameter names will be unique, and structured so it is intuitive what element of the structure learning process it is applying to, for example:

| Hyperparameter name        | Description                         |
|----------------------------|-------------------------------------|
| `max_iterations`           | A limit on the number of iterations |
| `max_elapsed`              | A limit on the execution time       |
| `score`                    | Score being used for score-based algorithms, e.g., `bdeu` |
| `iss`                      | The ISS hyperparameter for the BDeu score |
| `alpha`                    | The p-value used in constraint-based algorithms |

The supported hyperparameters and their allowed values will be **specific to each algorithm** which is detailed in the section devoted to each algorithm. A full list of hyperparameters supported by one or more algorithms can be found in the [Hyperparameters section](./hyperparameters.md).


### `trace` parameter

This optional parameter determines whether the results should contain a trace object or not; this is a simple boolean flag. The default is `false`.


### `variable_types` parameter

This optional parameter provides information about the variable types. It can be a string identifying a [CausalIQ Knowledge network context](https://causaliq.github.io/causaliq-knowledge/userguide/model_specification/) JSON file, or a `dict` in the Python `learn_graph` call. If a `dict` is used, the keys are variable names which must match the column headers in the `input` data file exactly. The supported variable types are defined in the `VariableTypes` `enum` defined in the [CausalIQ Core](https://causaliq.github.io/causaliq-core/) package, and are: `CONTINUOUS`, `DISCRETE`, `ORDINAL`, `BINARY` and `COUNT`. If this parameter is not specified, variable types would be imputed from their values in the `input` data file using standard Pandas rules.


### `sample_size` parameter

This optional parameter determines the number of rows in the input data file that the algorithm will use to learn from. This must be no more than the number of rows in the input data file, or if subsampling randomisation is being used (see below), no more than 10% of the number of rows in the input data file. The default is to use all rows in the input data file, or 10% of the rows if subsampling randomisation is being used.


### `variant` parameter

This optional parameter is used to specify a variant of a particular algorithm. This can be used to define which package to used (e.g. Tetrad or bnlearn) in cases where the same algorithm is supported by multiple packages, or a specific version of an algorithm to exactly replicate historical runs. This parameter is only valid for those algorithms where multiple variants exist, and the supported variants may be inspected using CLI command `cqdisc --help variant <algorithm>` The default variant is defined for each algorithm.


### `knowledge` parameter

This optional parameters supplies human, simulated human, interventional, or LLM-derived knowledge which guides or constrains the structure learning process so that it learns better causal graphs. The exact form is yet to be defined, but will likely depend upon the source of the knowledge and class of algorithm it is being applied to. It will be provided as a `Knowledge' object defined in the [CausalIQ Knowledge](https://causaliq.github.io/causaliq-knowledge/api/overview/) package, or as a JSON file representing the same. The default is that the strucyutre learning will not make use of any knowledge but rely on the data alone.


### `randomise` parameter

This optional parameter can be used to randomise aspects of the input dataset to explore the stability of the algorithm. Any combinations of the following values can be used for this randomisation:

- `row_order` a random ordering of rows is used
- `column_order` a random ordering of the columns is used
- `column_names` column names will be randomised 
- `row_subsample` a random selection of rows is used

The default is to use the row and column order, and column names present in the input data fie, and to use the first `sample_size` rows in the data file. 

### `seed` parameter

This parameter defines the randomisation seed when the `randomise` parameter is specified, and is an integer between 0 and 1000. All randomisation in the CausalIQ ecosystem is deterministic - so that a specific `seed` value guarantees the same randomisation on every occasion and every platform so that specific experiments will always produce the same results. Default value is `None` which will trigger an error if the `randomise` parameter is specified.