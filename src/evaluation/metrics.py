from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_image_metrics(
    y_true: List[int],
    y_scores: List[float],
    image_threshold: float,
) -> Dict[str, Any]:
    """Compute Image-level anomaly detection metrics."""
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = (y_scores_arr >= image_threshold).astype(int)

    # AUROC
    try:
        auroc = float(roc_auc_score(y_true_arr, y_scores_arr))
        fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)
    except Exception:
        auroc = float("nan")
        fpr, tpr = np.array([]), np.array([])

    # Binary metrics at fixed calibrated threshold
    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "cm_matrix": cm,
        "roc_curve": (fpr, tpr),
    }


def compute_pixel_metrics(
    all_gt_masks: List[np.ndarray],
    all_anomaly_maps: List[np.ndarray],
    pixel_threshold: float,
    max_auc_samples: int = 500000,
) -> Dict[str, Any]:
    """Compute Pixel-level anomaly localization metrics (Pixel AUROC, IoU, Dice)."""
    gt_flat = np.concatenate([m.flatten() for m in all_gt_masks]).astype(np.uint8)
    maps_flat = np.concatenate([a.flatten() for a in all_anomaly_maps]).astype(np.float32)

    # Subsample if pixel count is massive to prevent RAM exhaustion when computing ROC AUC
    if len(gt_flat) > max_auc_samples:
        indices = np.random.choice(len(gt_flat), size=max_auc_samples, replace=False)
        sub_gt = gt_flat[indices]
        sub_maps = maps_flat[indices]
    else:
        sub_gt = gt_flat
        sub_maps = maps_flat

    try:
        pixel_auroc = float(roc_auc_score(sub_gt, sub_maps))
        pixel_fpr, pixel_tpr, _ = roc_curve(sub_gt, sub_maps)
    except Exception:
        pixel_auroc = float("nan")
        pixel_fpr, pixel_tpr = np.array([]), np.array([])

    # Binary localization metrics at calibrated pixel_threshold
    pred_bin = (maps_flat > pixel_threshold).astype(np.uint8)

    intersection = np.sum((pred_bin == 1) & (gt_flat == 1))
    total_pred = np.sum(pred_bin == 1)
    total_gt = np.sum(gt_flat == 1)
    union = np.sum((pred_bin == 1) | (gt_flat == 1))

    iou = float(intersection / union) if union > 0 else (1.0 if total_gt == 0 else 0.0)
    dice = float(2 * intersection / (total_pred + total_gt)) if (total_pred + total_gt) > 0 else (
        1.0 if total_gt == 0 else 0.0
    )

    return {
        "pixel_auroc": pixel_auroc,
        "pixel_iou": iou,
        "pixel_dice": dice,
        "roc_curve": (pixel_fpr, pixel_tpr),
    }
