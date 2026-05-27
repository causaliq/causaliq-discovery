"""One-off script to extract integration test fixtures from legacy data.

Reads legacy experiment results from the causaliq/discovery repo and
saves reference GraphML files plus a 1 000-row Asia dataset slice into
the causaliq-discovery integration test data area.

Run once from the causaliq-discovery root (venv active):

    python tests/integration/setup/extract_integration_fixtures.py

Requires:
    - causaliq-analysis installed in the active venv
    - causaliq-data installed in the active venv
    - The legacy discovery repo at c:/dev/causaliq/discovery
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LEGACY_EXPERIMENTS = Path(r"c:\dev\causaliq\discovery\experiments")
LEGACY_DATASETS = LEGACY_EXPERIMENTS / "datasets"

OUT_ROOT = Path(__file__).parent.parent.parent / "data" / "integration"
DATASETS_OUT = OUT_ROOT / "datasets"
REFERENCE_OUT = OUT_ROOT / "reference"

# Series to extract: (partial_id, algorithm_label, variant_label)
_SERIES = [
    ("BNLEARN/HC_STD/asia", "hc", "bnlearn"),
    ("BNLEARN/TABU_STD/asia", "tabu", "bnlearn"),
    ("TETRAD/FGES_STD/asia", "fges", "tetrad"),
]

_SAMPLE_KEY = "N1000"
_DATASET_ROWS = 1000


def _extract_dataset() -> None:
    """Save first 1 000 rows of asia.data.gz as CSV."""
    from causaliq_data import NumPy

    data_path = str(LEGACY_DATASETS / "asia.data.gz")
    data = NumPy.read(data_path, dstype="categorical")
    df = data.as_df().iloc[:_DATASET_ROWS]

    DATASETS_OUT.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_OUT / "asia_N1000.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset: {out_path}  ({len(df)} rows)")


def _extract_reference(
    partial_id: str,
    algo_label: str,
    variant_label: str,
) -> None:
    """Load a legacy trace and save the N=1000 result as GraphML."""
    from causaliq_analysis.migrate import trace_to_graphml
    from causaliq_analysis.trace import Trace

    traces = Trace.read(partial_id, str(LEGACY_EXPERIMENTS))
    if traces is None or _SAMPLE_KEY not in traces:
        print(
            f"ERROR: key '{_SAMPLE_KEY}' not found in {partial_id}",
            file=sys.stderr,
        )
        sys.exit(1)

    trace = traces[_SAMPLE_KEY]
    graphml_str = trace_to_graphml(trace)

    out_dir = REFERENCE_OUT / variant_label / algo_label / "asia"
    out_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = out_dir / "graph.graphml"
    graphml_path.write_text(graphml_str, encoding="utf-8")
    print(f"Saved reference: {graphml_path}")

    # Save the params used so the integration test can verify them.
    params = {
        "algorithm": algo_label,
        "variant": variant_label,
        "network": "asia",
        "N": trace.context.get("N"),
        "score": trace.context.get("params", {}).get("score"),
        "penalty_weight": trace.context.get("params", {}).get("k"),
        "iss": trace.context.get("params", {}).get("iss"),
    }
    params_path = out_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Saved params:    {params_path}")


def main() -> None:
    """Extract all fixtures."""
    _extract_dataset()
    for partial_id, algo_label, variant_label in _SERIES:
        _extract_reference(partial_id, algo_label, variant_label)
    print("\nDone. Commit the generated files under tests/data/integration/")


if __name__ == "__main__":
    main()
