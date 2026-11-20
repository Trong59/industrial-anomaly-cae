from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.models.cae import ConvolutionalAutoencoder


def load_cae_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[ConvolutionalAutoencoder, Dict[str, Any]]:
    """Load CAE model and metadata from a saved checkpoint file.

    Guarantees seamless CPU loading on laptop even if trained on Colab GPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Reconstruct model from saved configuration
    model_cfg = checkpoint.get("model_config", {})
    model = ConvolutionalAutoencoder(
        in_channels=model_cfg.get("in_channels", 3),
        base_channels=model_cfg.get("base_channels", 32),
        bottleneck_channels=model_cfg.get("bottleneck_channels", None),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


class CAEPredictor:
    """Inference engine for Anomaly Detection and Localization using CAE."""

    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        device: torch.device,
        image_size: Tuple[int, int] = (256, 256),
        error_metric: str = "l1",
        top_percent: float = 0.01,
        image_threshold: Optional[float] = None,
        pixel_threshold: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.image_size = tuple(image_size)
        self.error_metric = error_metric.lower()
        self.top_percent = top_percent
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def compute_anomaly_map(self, images: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise anomaly map across RGB channels.

        Args:
            images: Tensor of shape (B, 3, H, W).
            reconstructions: Tensor of shape (B, 3, H, W).

        Returns:
            Tensor of shape (B, 1, H, W) containing anomaly heat values.
        """
        if self.error_metric == "l1":
            anomaly_map = torch.abs(images - reconstructions).mean(dim=1, keepdim=True)
        elif self.error_metric == "mse":
            anomaly_map = torch.pow(images - reconstructions, 2).mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown error_metric: {self.error_metric}")
        return anomaly_map

    def compute_image_score(self, anomaly_maps: torch.Tensor) -> torch.Tensor:
        """Compute image-level anomaly score from the top 1% highest pixel errors.

        Args:
            anomaly_maps: Tensor of shape (B, 1, H, W).

        Returns:
            1D Tensor of shape (B,) containing image anomaly scores.
        """
        b = anomaly_maps.size(0)
        flat = anomaly_maps.view(b, -1)
        k = max(1, int(flat.size(1) * self.top_percent))
        topk_vals, _ = torch.topk(flat, k, dim=1)
        return topk_vals.mean(dim=1)

    def predict_tensor_batch(
        self, images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Perform inference on a batch of normalized image tensors."""
        with torch.no_grad():
            images = images.to(self.device)
            recons = self.model(images)
            anomaly_maps = self.compute_anomaly_map(images, recons)
            image_scores = self.compute_image_score(anomaly_maps)

            pred_masks = None
            if self.pixel_threshold is not None:
                pred_masks = (anomaly_maps > self.pixel_threshold).float()

            pred_labels = None
            if self.image_threshold is not None:
                pred_labels = (image_scores >= self.image_threshold).long()

        return {
            "reconstructions": recons,
            "anomaly_maps": anomaly_maps,
            "image_scores": image_scores,
            "pred_masks": pred_masks,
            "pred_labels": pred_labels,
        }

    def predict_single_image(
        self, image_input: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """Perform end-to-end inference on a single image file or PIL Image."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        tensor_img = self.transform(pil_img).unsqueeze(0)  # (1, 3, H, W)
        batch_out = self.predict_tensor_batch(tensor_img)

        # Convert tensors to clean numpy formats for presentation
        orig_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        recon_np = batch_out["reconstructions"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        anomaly_map_np = batch_out["anomaly_maps"].squeeze().cpu().numpy()
        image_score = batch_out["image_scores"].item()

        pred_mask_np = None
        if batch_out["pred_masks"] is not None:
            pred_mask_np = batch_out["pred_masks"].squeeze().cpu().numpy()

        pred_label = None
        if batch_out["pred_labels"] is not None:
            pred_label = batch_out["pred_labels"].item()

        return {
            "original": orig_np,
            "reconstruction": recon_np,
            "anomaly_map": anomaly_map_np,
            "image_score": image_score,
            "pred_mask": pred_mask_np,
            "pred_label": pred_label,
            "is_anomaly": bool(pred_label == 1) if pred_label is not None else None,
        }
