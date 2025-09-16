"""Orchestrator script with CLI flags.

Usage examples
--------------
Smoke-test only
$ uv run python -m src.main --smoke-test

Full experiment only
$ uv run python -m src.main --full-experiment

If both flags are omitted the script defaults to running the smoke test
first and, **provided it succeeds**, continues with the full experiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

from . import evaluate as _eval
from . import preprocess, train


# ----------------------------------------------------------------------------
# YAML helpers
# ----------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)
    except FileNotFoundError as exc:  # pragma: no cover
        print(f"Configuration file not found: {path}", file=sys.stderr)
        raise exc


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------

def _run(cfg_path: Path, tag: str):
    cfg = _load_yaml(cfg_path)
    preprocess.prepare_data(cfg)
    metrics = train.train_model(cfg)
    _eval.output_results(tag, metrics)


def cli():  # pragma: no cover
    parser = argparse.ArgumentParser(description="ReDi-Mix refactored experiment runner")
    parser.add_argument("--smoke-test", action="store_true", help="run quick sanity check experiment")
    parser.add_argument("--full-experiment", action="store_true", help="run full-scale experiment")
    args = parser.parse_args()

    root_cfg_dir = Path(__file__).parent.parent / "config"
    smoke_cfg = root_cfg_dir / "smoke_test.yaml"
    full_cfg = root_cfg_dir / "full_experiment.yaml"

    # Decision logic
    run_smoke = args.smoke_test or not (args.smoke_test or args.full_experiment)
    run_full = args.full_experiment or not (args.smoke_test or args.full_experiment)

    if run_smoke:
        _run(smoke_cfg, tag="smoke_test")

    if run_full:
        # guard: only proceed if smoke-test succeeded (very naive check based on file existence)
        if run_smoke and not (Path(".research/iteration2") / "smoke_test.json").exists():
            print("Smoke-test did not complete. Aborting full experiment.", file=sys.stderr)
            sys.exit(1)
        _run(full_cfg, tag="full_experiment")


if __name__ == "__main__":  # pragma: no cover
    cli()
