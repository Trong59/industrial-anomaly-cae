import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch

from src.engine.predictor import CAEPredictor, load_cae_from_checkpoint
from src.evaluation.threshold import load_thresholds
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import save_prediction_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Predict anomaly score and mask on new surface image(s).")
    parser.add_argument("--config", type=str, default="configs/bottle.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth.")
    parser.add_argument("--thresholds", type=str, required=True, help="Path to calibrated thresholds JSON.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single image file or a directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/predictions",
        help="Directory to save visual prediction results.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda').")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger("PredictCLI")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, _ = load_cae_from_checkpoint(args.checkpoint, device=device)

    logger.info(f"Loading thresholds: {args.thresholds}")
    thresh_data = load_thresholds(args.thresholds)
    image_thresh = thresh_data["image_threshold"]
    pixel_thresh = thresh_data["pixel_threshold"]
    logger.info(f"Loaded Image Threshold: {image_thresh:.6f} | Pixel Threshold: {pixel_thresh:.6f}")

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [256, 256]))
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

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            [p for p in input_path.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    logger.info(f"Processing {len(image_files)} image(s)...")

    for img_path in image_files:
        result = predictor.predict_single_image(img_path)
        verdict = "ANOMALY (LỖI)" if result["is_anomaly"] else "NORMAL (BÌNH THƯỜNG)"

        logger.info(
            f"[{img_path.name}] -> Score: {result['image_score']:.6f} | "
            f"Threshold: {image_thresh:.6f} | Result: {verdict}"
        )

        # Save visualization panel
        save_path = output_dir / f"{img_path.stem}_prediction.png"
        save_prediction_sample(
            original_img=result["original"],
            recon_img=result["reconstruction"],
            anomaly_map=result["anomaly_map"],
            pred_mask=result["pred_mask"],
            gt_mask=None,
            save_path=save_path,
            image_score=result["image_score"],
            threshold=image_thresh,
            pred_label=result["pred_label"],
            true_label=0 if not result["is_anomaly"] else 1,
        )

    logger.info(f"All predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
