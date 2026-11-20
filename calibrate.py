import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.datasets.mvtec_dataset import get_mvtec_dataloaders
from src.engine.predictor import CAEPredictor, load_cae_from_checkpoint
from src.evaluation.threshold import calibrate_thresholds, save_thresholds
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Anomaly Detection Thresholds using Validation Normal set.")
    parser.add_argument("--config", type=str, default="configs/bottle.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MVTec dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Path to outputs root directory.")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique experiment ID. Defaults to config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to best_model.pth. Defaults to output_root/experiment_id/checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    exp_id = args.experiment_id or config.get("experiment_id", "E1_bottle_mse_seed42")

    seed = config.get("seed", 42)
    set_seed(seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category = config.get("category", "bottle")
    exp_dir = Path(args.output_root) / exp_id
    ckpt_path = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best_model.pth"

    logger = get_logger("CalibrateCLI")
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    model, ckpt_meta = load_cae_from_checkpoint(ckpt_path, device=device)

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    val_ratio = data_cfg.get("val_ratio", 0.2)
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = 0 if (sys.platform.startswith("win") and device.type == "cpu") else data_cfg.get("num_workers", 2)

    _, val_loader, _ = get_mvtec_dataloaders(
        data_root=args.data_root,
        category=category,
        image_size=image_size,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    inf_cfg = config.get("inference", {})
    calib_cfg = config.get("calibration", {})

    predictor = CAEPredictor(
        model=model,
        device=device,
        image_size=image_size,
        error_metric=inf_cfg.get("error_metric", "l1"),
        top_percent=inf_cfg.get("top_percent", 0.01),
    )

    thresholds_dict = calibrate_thresholds(
        predictor=predictor,
        val_loader=val_loader,
        rule=calib_cfg.get("rule", "percentile"),
        image_percentile=calib_cfg.get("image_percentile", 99.0),
        image_k_std=calib_cfg.get("image_k_std", 3.0),
        pixel_percentile=calib_cfg.get("pixel_percentile", 99.5),
    )

    # Save to outputs/<exp_id>/thresholds/<category>_thresholds.json
    save_path = exp_dir / "thresholds" / f"{category}_thresholds.json"
    save_thresholds(thresholds_dict, save_path)
    logger.info(f"Thresholds successfully saved to: {save_path}")


if __name__ == "__main__":
    main()
