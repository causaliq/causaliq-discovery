"""DiscoveryResult model and output serialisation."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from causaliq_core.graph import SDG
from causaliq_core.graph.io import graphml


@dataclass
class DiscoveryResult:
    """Result of a structure learning run.

    Attributes:
        graph: Learnt causal graph as an SDG.
        metadata: Dictionary of run metadata, including algorithm
            name, variant, hyperparameters used, and any statistics
            reported by the algorithm (e.g. score, iterations).
        trace: Step-by-step execution trace as a list of dicts, or
            None when tracing was not requested.

    Example:
        >>> from causaliq_core.graph import SDG
        >>> graph = SDG(["A", "B"], [("A", "->", "B")])
        >>> result = DiscoveryResult(
        ...     graph=graph,
        ...     metadata={"algorithm": "tabu-stable"},
        ... )
        >>> result.graph.nodes
        ['A', 'B']
    """

    graph: SDG
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[List[Dict[str, Any]]] = None

    def save(self, output_dir: str) -> None:
        """Write the result to a directory.

        Creates ``output_dir`` if it does not exist, then writes:
        - ``graph.graphml`` — the learnt graph in GraphML format.
        - ``metadata.json`` — run metadata as indented JSON.
        - ``trace.json`` — execution trace as indented JSON, only
          written when ``self.trace`` is not None.

        Args:
            output_dir: Directory path to write the output files to.

        Raises:
            TypeError: If output_dir is not a string.
            ValueError: If output_dir is an empty string.
        """
        if not isinstance(output_dir, str):
            raise TypeError(
                "'output_dir' must be a string; "
                f"got {type(output_dir).__name__}."
            )
        if not output_dir:
            raise ValueError("'output_dir' must not be an empty string.")

        os.makedirs(output_dir, exist_ok=True)

        graphml_path = os.path.join(output_dir, "graph.graphml")
        graphml.write(self.graph, graphml_path)

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        if self.trace is not None:
            trace_path = os.path.join(output_dir, "trace.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(self.trace, f, indent=2)
