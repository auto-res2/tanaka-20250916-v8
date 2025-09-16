"""Evaluation, statistical analysis and plotting utilities.
This tiny module complements `src.train` by providing a standalone
interface that prints evaluation results to stdout and dumps them to
.research/iteration1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def output_results(name: str, metrics: Dict[str, Any]):
    """Persist *and* print evaluation metrics.

    Metrics are stored as JSON in `.research/iteration1/{name}.json` and
    also echoed to stdout so that the CI harness can capture them.
    """
    out_dir = Path(".research/iteration1")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{name}.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"===== Evaluation: {name} =====")
    print(json.dumps(metrics, indent=2))
