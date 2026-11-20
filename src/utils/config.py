import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def setup_experiment_dir(
    output_root: str,
    experiment_id: str,
    config: Dict[str, Any],
    resume: bool = False,
) -> Dict[str, Path]:
    """Set up directory structure for an experiment and save a copy of the config.

    Guarantees that old experiment results are not overwritten unless resume=True.

    Args:
        output_root: Base output directory.
        experiment_id: Unique experiment name (e.g., 'E1_bottle_mse_seed42').
        config: Configuration dictionary to be saved as config_used.yaml.
        resume: If True, allow existing directory for resuming training.

    Returns:
        Dictionary mapping directory names to Path objects:
        {
            'root': ...,
            'checkpoints': ...,
            'thresholds': ...,
            'logs': ...,
            'metrics': ...,
            'plots': ...,
            'visualizations': ...
        }
    """
    exp_dir = Path(output_root) / experiment_id
    if exp_dir.exists() and not resume:
        raise FileExistsError(
            f"Experiment directory '{exp_dir}' already exists! "
            f"Please choose a new --experiment-id or pass --resume to continue."
        )

    dirs = {
        "root": exp_dir,
        "checkpoints": exp_dir / "checkpoints",
        "thresholds": exp_dir / "thresholds",
        "logs": exp_dir / "logs",
        "metrics": exp_dir / "metrics",
        "plots": exp_dir / "plots",
        "visualizations": exp_dir / "visualizations",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Save a snapshot of the config used for reproducibility
    config_copy_path = exp_dir / "config_used.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return dirs
