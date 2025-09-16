"""Evaluation, statistical analysis and plotting utilities.
This tiny module complements `src.train` by providing a standalone
interface that prints evaluation results to stdout and dumps them to
.research/iteration2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt


def output_results(name: str, metrics: Dict[str, Any]):
    """Persist *and* print evaluation metrics with experiment details."""
    
    out_dir = Path(".research/iteration2")
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(['Accuracy', 'Loss'], [metrics.get('val_accuracy', 0), metrics.get('val_loss', 0)])
    ax.set_title(f'Experiment Results: {name}')
    ax.set_ylabel('Value')
    
    img_path = img_dir / f"{name}_results.png"
    plt.savefig(img_path)
    plt.close()
    
    enhanced_metrics = {
        **metrics,
        "experiment_name": name,
        "image_path": str(img_path),
        "timestamp": str(Path().cwd()),
        "concrete_data": {
            "validation_accuracy": f"{metrics.get('val_accuracy', 0):.4f}",
            "validation_loss": f"{metrics.get('val_loss', 0):.4f}",
            "total_parameters": metrics.get('total_params', 'N/A')
        }
    }
    
    file_path = out_dir / f"{name}.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(enhanced_metrics, fp, indent=2)

    print(f"===== Experiment Details: {name} =====")
    print(f"Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")
    print(f"Validation Loss: {metrics.get('val_loss', 0):.4f}")
    print(f"Image Path: {img_path}")
    print("===== JSON Results =====")
    print(json.dumps(enhanced_metrics, indent=2))
