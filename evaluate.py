import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from src.datasets.mvtec_dataset import get_mvtec_dataloaders
from src.engine.predictor import CAEPredictor, load_cae_from_checkpoint
from src.evaluation.metrics import compute_image_metrics, compute_pixel_metrics
from src.evaluation.threshold import load_thresholds
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    save_prediction_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAE on MVTec AD Test set.")
    parser.add_argument("--config", type=str, default="configs/bottle.yaml", help="Path to config YAML.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MVTec dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Path to outputs root directory.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth.")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to calibrated thresholds JSON.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    parser.add_argument("--max-vis-samples", type=int, default=5, help="Max visualizations per category.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    exp_id = args.experiment_id or config.get("experiment_id", "E1_bottle_mse_seed42")

    seed = config.get("seed", 42)
    set_seed(seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category = config.get("category", "bottle")
    exp_dir = Path(args.output_root) / exp_id

    # Checkpoint and thresholds paths
    ckpt_path = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best_model.pth"
    thresh_path = Path(args.thresholds) if args.thresholds else exp_dir / "thresholds" / f"{category}_thresholds.json"

    logger = get_logger("EvaluateCLI")
    logger.info(f"Loading best checkpoint from: {ckpt_path}")
    model, _ = load_cae_from_checkpoint(ckpt_path, device=device)

    logger.info(f"Loading calibrated thresholds from: {thresh_path}")
    thresh_data = load_thresholds(thresh_path)
    image_thresh = thresh_data["image_threshold"]
    pixel_thresh = thresh_data["pixel_threshold"]
    logger.info(f"Using Image Threshold: {image_thresh:.6f} | Pixel Threshold: {pixel_thresh:.6f}")

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    num_workers = 0 if (sys.platform.startswith("win") and device.type == "cpu") else data_cfg.get("num_workers", 2)

    _, _, test_loader = get_mvtec_dataloaders(
        data_root=args.data_root,
        category=category,
        image_size=image_size,
        batch_size=1,  # Batch size 1 for individual image-mask processing
        num_workers=num_workers,
        seed=seed,
    )

    inf_cfg = config.get("inference", {})
    predictor = CAEPredictor(
        model=model,
        device=device,
        image_size=image_size,
        error_metric=inf_cfg.get("error_metric", "l1"),
        top_percent=inf_cfg.get("top_percent", 0.01),
        image_threshold=image_thresh,
        pixel_threshold=pixel_thresh,
    )

    y_true: List[int] = []
    y_scores: List[float] = []
    all_gt_masks: List[np.ndarray] = []
    all_anomaly_maps: List[np.ndarray] = []
    records = []

    # Visualizations output directories
    vis_base = exp_dir / "visualizations" / category
    vis_dirs = {
        "normal": vis_base / "normal",
        "anomaly": vis_base / "anomaly",
        "false_positive": vis_base / "false_positive",
        "false_negative": vis_base / "false_negative",
    }
    for d in vis_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    vis_counts = {"normal": 0, "anomaly": 0, "false_positive": 0, "false_negative": 0}

    logger.info(f"Evaluating {len(test_loader.dataset)} test samples...")
    for idx, batch in enumerate(test_loader):
        images = batch["image"]
        masks = batch["mask"]
        label = batch["label"].item()
        defect_type = batch["defect_type"][0]
        img_path = batch["image_path"][0]

        batch_out = predictor.predict_tensor_batch(images)

        score = batch_out["image_scores"].item()
        pred_label = batch_out["pred_labels"].item()
        anomaly_map = batch_out["anomaly_maps"].squeeze().cpu().numpy()
        gt_mask = masks.squeeze().cpu().numpy()
        pred_mask = batch_out["pred_masks"].squeeze().cpu().numpy()

        y_true.append(label)
        y_scores.append(score)
        all_gt_masks.append(gt_mask)
        all_anomaly_maps.append(anomaly_map)

        # Classification status
        if label == 1 and pred_label == 1:
            status = "TP"
            category_group = "anomaly"
        elif label == 0 and pred_label == 0:
            status = "TN"
            category_group = "normal"
        elif label == 0 and pred_label == 1:
            status = "FP"
            category_group = "false_positive"
        else:
            status = "FN"
            category_group = "false_negative"

        records.append(
            {
                "sample_idx": idx,
                "image_path": img_path,
                "defect_type": defect_type,
                "true_label": label,
                "anomaly_score": score,
                "image_threshold": image_thresh,
                "pred_label": pred_label,
                "status": status,
            }
        )

        # Save qualitative visualization sample if under limit
        if vis_counts[category_group] < args.max_vis_samples:
            orig_np = images.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_np = batch_out["reconstructions"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            save_name = f"{Path(img_path).stem}_{status}.png"
            save_path = vis_dirs[category_group] / save_name

            save_prediction_sample(
                original_img=orig_np,
                recon_img=recon_np,
                anomaly_map=anomaly_map,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                save_path=save_path,
                image_score=score,
                threshold=image_thresh,
                pred_label=pred_label,
                true_label=label,
            )
            vis_counts[category_group] += 1

    # 1. Compute Metrics
    image_metrics = compute_image_metrics(y_true, y_scores, image_threshold=image_thresh)
    pixel_metrics = compute_pixel_metrics(all_gt_masks, all_anomaly_maps, pixel_threshold=pixel_thresh)

    logger.info("=" * 50)
    logger.info("FINAL EVALUATION RESULTS ON TEST SET:")
    logger.info(f"Image AUROC: {image_metrics['auroc']:.4f}")
    logger.info(f"Image F1:    {image_metrics['f1']:.4f}")
    logger.info(f"Precision:   {image_metrics['precision']:.4f}")
    logger.info(f"Recall:      {image_metrics['recall']:.4f}")
    logger.info(f"Confusion Matrix: {image_metrics['confusion_matrix']}")
    logger.info(f"Pixel AUROC: {pixel_metrics['pixel_auroc']:.4f}")
    logger.info(f"Pixel IoU:   {pixel_metrics['pixel_iou']:.4f}")
    logger.info(f"Pixel Dice:  {pixel_metrics['pixel_dice']:.4f}")
    logger.info("=" * 50)

    # 2. Save CSV of all sample predictions
    pred_csv_path = exp_dir / "metrics" / f"{category}_predictions.csv"
    pd.DataFrame(records).to_csv(pred_csv_path, index=False)
    logger.info(f"Saved sample predictions to: {pred_csv_path}")

    # 3. Save JSON summary metrics
    summary_metrics = {
        "category": category,
        "experiment_id": exp_id,
        "image_level": {
            "auroc": image_metrics["auroc"],
            "f1": image_metrics["f1"],
            "precision": image_metrics["precision"],
            "recall": image_metrics["recall"],
            "confusion_matrix": image_metrics["confusion_matrix"],
            "threshold": image_thresh,
        },
        "pixel_level": {
            "pixel_auroc": pixel_metrics["pixel_auroc"],
            "pixel_iou": pixel_metrics["pixel_iou"],
            "pixel_dice": pixel_metrics["pixel_dice"],
            "threshold": pixel_thresh,
        },
    }
    metrics_json_path = exp_dir / "metrics" / f"{category}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    logger.info(f"Saved summary metrics to: {metrics_json_path}")

    # 4. Plots
    plots_dir = exp_dir / "plots"
    # Image ROC Curve
    fpr, tpr = image_metrics["roc_curve"]
    if len(fpr) > 0:
        roc_plot_path = plots_dir / f"{category}_image_roc.png"
        plot_roc_curve(fpr, tpr, image_metrics["auroc"], roc_plot_path, title=f"Image ROC - {category}")
        logger.info(f"Saved Image ROC curve to: {roc_plot_path}")

    # Confusion Matrix Heatmap
    cm_plot_path = plots_dir / f"{category}_confusion_matrix.png"
    plot_confusion_matrix(
        image_metrics["cm_matrix"],
        class_names=["Normal", "Anomaly"],
        save_path=cm_plot_path,
        title=f"Confusion Matrix - {category}",
    )
    logger.info(f"Saved Confusion Matrix to: {cm_plot_path}")


if __name__ == "__main__":
    main()
