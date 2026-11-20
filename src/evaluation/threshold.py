import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.engine.predictor import CAEPredictor
from src.utils.logger import get_logger


def calibrate_thresholds(
    predictor: CAEPredictor,
    val_loader: DataLoader,
    rule: str = "percentile",
    image_percentile: float = 99.0,
    image_k_std: float = 3.0,
    pixel_percentile: float = 99.5,
) -> Dict[str, Any]:
    """Calibrate image and pixel anomaly thresholds using ONLY normal validation samples.

    Supports:
        - 'percentile': threshold = np.percentile(val_scores, p)
        - 'mean_k_std': threshold = mean(val_scores) + k * std(val_scores)

    Returns:
        Dictionary containing calibrated thresholds, statistics, and rule metadata.
    """
    logger = get_logger("Calibrator")
    logger.info(f"Starting threshold calibration on validation set ({len(val_loader.dataset)} normal images)...")

    val_image_scores: List[float] = []
    # For memory efficiency, sample a subset of pixel errors across validation batches
    pixel_error_sample: List[float] = []

    for batch in val_loader:
        images = batch["image"]
        batch_out = predictor.predict_tensor_batch(images)

        scores = batch_out["image_scores"].cpu().numpy().tolist()
        val_image_scores.extend(scores)

        # Subsample pixel errors (e.g. 5% random pixels per map) to avoid Out Of Memory
        maps_np = batch_out["anomaly_maps"].cpu().numpy().flatten()
        if len(maps_np) > 20000:
            sub = np.random.choice(maps_np, size=20000, replace=False)
        else:
            sub = maps_np
        pixel_error_sample.extend(sub.tolist())

    scores_arr = np.array(val_image_scores)
    pixels_arr = np.array(pixel_error_sample)

    val_count = len(scores_arr)
    score_min = float(np.min(scores_arr))
    score_mean = float(np.mean(scores_arr))
    score_std = float(np.std(scores_arr))
    score_max = float(np.max(scores_arr))

    logger.info(
        f"Validation Image Score Stats -> Count: {val_count}, "
        f"Min: {score_min:.6f}, Mean: {score_mean:.6f}, Std: {score_std:.6f}, Max: {score_max:.6f}"
    )

    # 1. Determine Image Threshold
    if rule == "mean_k_std":
        image_threshold = score_mean + image_k_std * score_std
        chosen_rule_info = {"rule": "mean_k_std", "k": image_k_std}
    elif rule == "percentile":
        image_threshold = float(np.percentile(scores_arr, image_percentile))
        chosen_rule_info = {"rule": "percentile", "percentile": image_percentile}
    else:
        raise ValueError(f"Unknown threshold rule: {rule}. Choose 'percentile' or 'mean_k_std'.")

    # 2. Determine Pixel Threshold (always percentile on normal pixel errors)
    pixel_threshold = float(np.percentile(pixels_arr, pixel_percentile))

    # Compute False Positive Rate on validation normal set with this threshold
    val_fp_rate = float(np.mean(scores_arr >= image_threshold))

    logger.info(f"Calibrated Image Threshold: {image_threshold:.6f} ({chosen_rule_info})")
    logger.info(f"Calibrated Pixel Threshold: {pixel_threshold:.6f} (Percentile: {pixel_percentile}%)")
    logger.info(f"Validation Normal False Positive Rate: {val_fp_rate * 100:.2f}%")

    thresholds_dict = {
        "image_threshold": float(image_threshold),
        "pixel_threshold": float(pixel_threshold),
        "validation_stats": {
            "num_samples": val_count,
            "min": score_min,
            "mean": score_mean,
            "std": score_std,
            "max": score_max,
            "val_fp_rate": val_fp_rate,
        },
        "image_threshold_rule": chosen_rule_info,
        "pixel_threshold_rule": {"rule": "percentile", "percentile": pixel_percentile},
    }

    return thresholds_dict


def save_thresholds(
    thresholds_dict: Dict[str, Any],
    save_path: Union[str, Path],
) -> None:
    """Save thresholds dictionary into a formatted JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)


def load_thresholds(thresholds_path: Union[str, Path]) -> Dict[str, Any]:
    """Load thresholds dictionary from JSON."""
    thresholds_path = Path(thresholds_path)
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"Thresholds file not found at: {thresholds_path}")
    with open(thresholds_path, "r", encoding="utf-8") as f:
        return json.load(f)
