from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """ConvTranspose2d -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder (CAE) for surface anomaly detection.

    Architecture (Default for 256x256 input):
        Encoder:
            - Input: (B, 3, 256, 256)
            - Block 1: Conv 3 -> 32, stride 2  => (B, 32, 128, 128)
            - Block 2: Conv 32 -> 64, stride 2 => (B, 64, 64, 64)
            - Block 3: Conv 64 -> 128, stride 2=> (B, 128, 32, 32)
            - Block 4: Conv 128 -> 256, stride 2=> (B, 256, 16, 16)
        Bottleneck (optional compression):
            - If bottleneck_channels is set, applies 1x1 conv compression & expansion.
        Decoder:
            - Block 1: Deconv 256 -> 128, stride 2 => (B, 128, 32, 32)
            - Block 2: Deconv 128 -> 64, stride 2  => (B, 64, 64, 64)
            - Block 3: Deconv 64 -> 32, stride 2   => (B, 32, 128, 128)
            - Block 4: Deconv 32 -> 3, stride 2    => (B, 3, 256, 256)
            - Output Activation: Sigmoid (ranges [0, 1])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        bottleneck_channels: Optional[int] = None,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bottleneck_channels = bottleneck_channels
        self.output_activation = output_activation.lower()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        # Optional configurable bottleneck compression
        if bottleneck_channels is not None and bottleneck_channels > 0:
            self.bottleneck_compress = nn.Sequential(
                nn.Conv2d(c4, bottleneck_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(inplace=True),
            )
            self.bottleneck_expand = nn.Sequential(
                nn.Conv2d(bottleneck_channels, c4, kernel_size=1, bias=False),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
            )
        else:
            self.bottleneck_compress = nn.Identity()
            self.bottleneck_expand = nn.Identity()

        # Decoder
        self.dec1 = DeconvBlock(c4, c3)
        self.dec2 = DeconvBlock(c3, c2)
        self.dec3 = DeconvBlock(c2, c1)

        # Final reconstruction layer
        self.final_conv = nn.ConvTranspose2d(c1, in_channels, kernel_size=4, stride=2, padding=1, bias=True)

        if self.output_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.output_activation == "tanh":
            self.act = nn.Tanh()
        elif self.output_activation in ["none", "linear"]:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and bottleneck."""
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_compress(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck expansion and decoder."""
        x = self.bottleneck_expand(z)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.final_conv(x)
        return self.act(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction forward pass."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return model hyperparameter dictionary for saving in checkpoint."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "output_activation": self.output_activation,
        }


def build_cae_from_config(config: Dict[str, Any]) -> ConvolutionalAutoencoder:
    """Build a ConvolutionalAutoencoder from a configuration dictionary."""
    model_cfg = config.get("model", {})
    return ConvolutionalAutoencoder(
        in_channels=model_cfg.get("in_channels", 3),
        base_channels=model_cfg.get("base_channels", 32),
        bottleneck_channels=model_cfg.get("bottleneck_channels", None),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
    )
