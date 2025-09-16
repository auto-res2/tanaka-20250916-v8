"""Data-loading / preprocessing stub.
For this minimal example all heavy lifting is done in `src.train` itself,
but we keep the module and a helper so that the refactor matches the
requested structure.
"""
from __future__ import annotations

from typing import Any, Dict


def prepare_data(cfg: Dict[str, Any]):  # noqa: D401
    """No-op wrapper kept for API completeness."""
    print(f"Data preparation step skipped — using torchvision datasets for {cfg['data']['name']}")
