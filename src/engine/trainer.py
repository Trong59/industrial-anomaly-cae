import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cae import ConvolutionalAutoencoder
from src.utils.logger import CSVMetricsLogger, get_logger
from src.utils.visualization import plot_loss_curve


class CAETrainer:
    """Trainer for Convolutional Autoencoder on normal surface images."""

    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dirs: Dict[str, Path],
        device: torch.device,
        resume: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dirs = output_dirs
        self.device = device
        self.resume = resume

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.lr = train_cfg.get("lr", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
        self.loss_type = train_cfg.get("loss", "mse").lower()

        # Loss function
        if self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Choose 'mse' or 'l1'.")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Logging setup
        category = config.get("category", "bottle")
        log_file = self.output_dirs["logs"] / f"{category}_train.log"
        self.logger = get_logger("CAETrainer", log_file=log_file)

        csv_file = self.output_dirs["logs"] / f"{category}_train.csv"
        self.csv_logger = CSVMetricsLogger(
            csv_path=csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "epoch_time_sec"],
        )

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        if self.resume:
            self._load_last_checkpoint()

    def _load_last_checkpoint(self) -> None:
        last_ckpt_path = self.output_dirs["checkpoints"] / "last_checkpoint.pth"
        if last_ckpt_path.exists():
            self.logger.info(f"Resuming training from checkpoint: {last_ckpt_path}")
            checkpoint = torch.load(last_ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.logger.info(f"Resumed at epoch {self.start_epoch} with best_val_loss: {self.best_val_loss:.6f}")
        else:
            self.logger.warning(f"Resume requested but {last_ckpt_path} not found. Starting from epoch 1.")

    def train_epoch(self) -> float:
        """Run one training epoch on normal samples."""
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad()

            reconstructions = self.model(images)
            loss = self.criterion(reconstructions, images)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self) -> float:
        """Evaluate reconstruction loss on normal validation set."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                reconstructions = self.model(images)
                loss = self.criterion(reconstructions, images)
                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save last checkpoint and best model with full reproduction metadata."""
        category = self.config.get("category", "bottle")
        exp_id = self.config.get("experiment_id", "default_exp")

        # 1. Save last checkpoint for resume
        last_state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "experiment_id": exp_id,
        }
        last_path = self.output_dirs["checkpoints"] / "last_checkpoint.pth"
        torch.save(last_state, last_path)

        # 2. Save best model
        if is_best:
            best_state = {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.get_config(),
                "epoch": epoch,
                "val_loss": val_loss,
                "category": category,
                "experiment_id": exp_id,
                "input_size": self.config.get("data", {}).get("image_size", [256, 256]),
                "loss_type": self.loss_type,
            }
            best_path = self.output_dirs["checkpoints"] / "best_model.pth"
            torch.save(best_state, best_path)
            self.logger.info(f"[*] New best model saved to: {best_path} (val_loss: {val_loss:.6f})")

    def fit(self) -> Dict[str, Any]:
        """Execute full training loop with early stopping and loss curves."""
        category = self.config.get("category", "bottle")
        self.logger.info(
            f"Starting CAE training: epochs={self.epochs}, lr={self.lr}, loss={self.loss_type}, device={self.device}"
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_time = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.epochs:03d}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Log to CSV
            self.csv_logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                    "epoch_time_sec": round(epoch_time, 2),
                }
            )

            # Checkpoint & Early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch=epoch, val_loss=val_loss, is_best=is_best)

            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}! No improvement for {self.early_stopping_patience} epochs."
                )
                break

        # Plot and save loss curve
        plot_path = self.output_dirs["plots"] / f"{category}_loss_curve.png"
        plot_loss_curve(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            save_path=plot_path,
            title=f"Loss Curve - {category} ({self.config.get('experiment_id')})",
        )
        self.logger.info(f"Loss curve saved to: {plot_path}")

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.train_losses),
        }
