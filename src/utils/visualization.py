from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Union[str, Path],
    title: str = "Training and Validation Loss",
) -> None:
    """Plot and save training and validation loss curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=3)
    plt.plot(epochs, val_losses, "r-s", label="Validation Loss", markersize=3)
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Union[str, Path],
    title: str = "ROC Curve",
) -> None:
    """Plot and save ROC curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:d}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_anomaly_overlay(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Blend red overlay on anomalous regions for visualization.

    Args:
        image_np: RGB array in [0, 1], shape (H, W, 3).
        mask_np: Binary mask {0, 1}, shape (H, W).
        alpha: Overlay opacity.
        color: RGB tuple for overlay color (default: red).

    Returns:
        Overlay image array in [0, 1], shape (H, W, 3).
    """
    overlay = image_np.copy()
    mask_bool = mask_np > 0.5
    for c in range(3):
        overlay[mask_bool, c] = (1 - alpha) * image_np[mask_bool, c] + alpha * color[c]
    return overlay


def save_prediction_sample(
    original_img: np.ndarray,
    recon_img: np.ndarray,
    anomaly_map: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    save_path: Union[str, Path],
    image_score: float,
    threshold: float,
    pred_label: int,
    true_label: int,
) -> None:
    """Save a comprehensive 5-panel or 6-panel prediction visualization.

    Panels:
        1. Original Image
        2. Reconstructed Image
        3. Anomaly Heatmap
        4. Predicted Mask
        5. Overlay (Image + Predicted Mask)
        6. Ground Truth Mask (if provided)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    has_gt = gt_mask is not None
    cols = 6 if has_gt else 5
    fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3.5))

    status = "TP" if (true_label == 1 and pred_label == 1) else (
        "TN" if (true_label == 0 and pred_label == 0) else (
            "FP" if (true_label == 0 and pred_label == 1) else "FN"
        )
    )

    pred_str = "Anomaly" if pred_label == 1 else "Normal"
    true_str = "Anomaly" if true_label == 1 else "Normal"
    title_main = f"Pred: {pred_str} | True: {true_str} [{status}] | Score: {image_score:.4f} (Thresh: {threshold:.4f})"
    fig.suptitle(title_main, fontsize=12, fontweight="bold")

    # 1. Original
    axes[0].imshow(np.clip(original_img, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 2. Reconstructed
    axes[1].imshow(np.clip(recon_img, 0, 1))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    # 3. Anomaly Map (Heatmap)
    im3 = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Anomaly Map")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Predicted Mask
    axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Predicted Mask")
    axes[3].axis("off")

    # 5. Overlay
    overlay = create_anomaly_overlay(original_img, pred_mask)
    axes[4].imshow(np.clip(overlay, 0, 1))
    axes[4].set_title("Overlay (Pred)")
    axes[4].axis("off")

    # 6. Ground Truth
    if has_gt:
        axes[5].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[5].set_title("Ground Truth Mask")
        axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
