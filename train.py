import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.datasets.mvtec_dataset import get_mvtec_dataloaders
from src.engine.trainer import CAETrainer
from src.models.cae import build_cae_from_config
from src.utils.config import load_config, setup_experiment_dir
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder on MVTec AD.")
    parser.add_argument("--config", type=str, default="configs/bottle.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to MVTec dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Path to output root directory.")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique experiment ID. Defaults to the one specified in config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last_checkpoint.pth in experiment directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda' or 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI arguments override config
    experiment_id = args.experiment_id or config.get("experiment_id", "E1_bottle_mse_seed42")
    config["experiment_id"] = experiment_id

    seed = config.get("seed", 42)
    set_seed(seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup experiment directory structure
    output_dirs = setup_experiment_dir(
        output_root=args.output_root,
        experiment_id=experiment_id,
        config=config,
        resume=args.resume,
    )

    data_cfg = config.get("data", {})
    category = config.get("category", "bottle")
    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    batch_size = data_cfg.get("batch_size", 16)
    val_ratio = data_cfg.get("val_ratio", 0.2)
    num_workers = data_cfg.get("num_workers", 2)

    # On Windows or CPU, if num_workers fails, allow falling back to 0
    if sys.platform.startswith("win") and device.type == "cpu":
        num_workers = 0

    train_loader, val_loader, _ = get_mvtec_dataloaders(
        data_root=args.data_root,
        category=category,
        image_size=image_size,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # Build model
    model = build_cae_from_config(config)

    # Initialize and run trainer
    trainer = CAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dirs=output_dirs,
        device=device,
        resume=args.resume,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
