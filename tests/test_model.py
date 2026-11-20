import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.models.cae import ConvolutionalAutoencoder, build_cae_from_config


def test_cae_forward_backward():
    # 1. Instantiate baseline model
    model = ConvolutionalAutoencoder(in_channels=3, base_channels=32, bottleneck_channels=None, output_activation="sigmoid")
    num_params = model.count_parameters()
    print(f"Total trainable parameters (baseline): {num_params:,}")
    assert num_params > 0

    # 2. Test forward pass with batch (T04)
    batch_size = 4
    x = torch.rand(batch_size, 3, 256, 256)
    out = model(x)

    assert out.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, f"Sigmoid output not in [0, 1]: min={out.min()}, max={out.max()}"
    assert not torch.isnan(out).any(), "Output contains NaN values!"
    assert not torch.isinf(out).any(), "Output contains Inf values!"
    print("[T04 PASSED] Model forward pass produced matching shape and valid range [0, 1].")

    # 3. Test backward pass with MSE loss (T05)
    criterion = nn.MSELoss()
    loss = criterion(out, x)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()

    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient contains NaN for {name}"
            has_grad = True

    assert has_grad, "No gradients were computed!"
    print(f"[T05 PASSED] Backward pass succeeded. Loss = {loss.item():.6f}, gradients computed cleanly.")

    # 4. Test with bottleneck compression enabled
    compressed_model = ConvolutionalAutoencoder(
        in_channels=3, base_channels=32, bottleneck_channels=64, output_activation="sigmoid"
    )
    compressed_out = compressed_model(x)
    assert compressed_out.shape == x.shape
    print(
        f"[Bottleneck Check PASSED] Model with bottleneck_channels=64: params={compressed_model.count_parameters():,}"
    )


if __name__ == "__main__":
    test_cae_forward_backward()
