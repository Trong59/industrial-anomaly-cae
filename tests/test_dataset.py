import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from src.datasets.mvtec_dataset import MVTecDataset, get_mvtec_dataloaders


def create_dummy_mvtec_dataset(root_dir: Path, category: str = "bottle"):
    """Creates a minimal dummy MVTec AD dataset structure for local testing."""
    cat_dir = root_dir / category
    train_good = cat_dir / "train" / "good"
    test_good = cat_dir / "test" / "good"
    test_defect = cat_dir / "test" / "broken_large"
    gt_defect = cat_dir / "ground_truth" / "broken_large"

    for d in [train_good, test_good, test_defect, gt_defect]:
        d.mkdir(parents=True, exist_ok=True)

    # 10 train normal images
    for i in range(10):
        img_arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        Image.fromarray(img_arr).save(train_good / f"{i:03d}.png")

    # 3 test normal images
    for i in range(3):
        img_arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        Image.fromarray(img_arr).save(test_good / f"{i:03d}.png")

    # 3 test defect images with masks
    for i in range(3):
        img_arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        Image.fromarray(img_arr).save(test_defect / f"{i:03d}.png")

        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[30:70, 30:70] = 255  # Defect square
        Image.fromarray(mask_arr).save(gt_defect / f"{i:03d}_mask.png")


def test_mvtec_dataset_and_loader():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        category = "bottle"
        create_dummy_mvtec_dataset(temp_dir, category=category)

        # 1. Test Dataset Initialization
        train_ds = MVTecDataset(data_root=temp_dir, category=category, split="train", val_ratio=0.2, seed=42)
        val_ds = MVTecDataset(data_root=temp_dir, category=category, split="validation", val_ratio=0.2, seed=42)
        test_ds = MVTecDataset(data_root=temp_dir, category=category, split="test", val_ratio=0.2, seed=42)

        # 10 images total in train: 2 for val (20%), 8 for train (80%)
        assert len(train_ds) == 8, f"Expected 8 train samples, got {len(train_ds)}"
        assert len(val_ds) == 2, f"Expected 2 val samples, got {len(val_ds)}"
        # Test: 3 good + 3 broken_large = 6 total
        assert len(test_ds) == 6, f"Expected 6 test samples, got {len(test_ds)}"
        print("[T01/T02 PASSED] Dataset splits and sample counts verified.")

        # 2. Test single sample shapes and values
        sample = train_ds[0]
        assert sample["image"].shape == (3, 256, 256), f"Image shape: {sample['image'].shape}"
        assert sample["mask"].shape == (1, 256, 256), f"Mask shape: {sample['mask'].shape}"
        assert sample["label"].item() == 0, f"Train label should be 0, got {sample['label']}"
        assert sample["image"].min() >= 0.0 and sample["image"].max() <= 1.0, "Image pixel range not in [0, 1]"
        assert (sample["mask"] == 0).all(), "Train normal mask must be all zeros"

        defect_sample = [s for s in test_ds if s["defect_type"] == "broken_large"][0]
        assert defect_sample["label"].item() == 1, "Defect label should be 1"
        assert defect_sample["mask"].sum() > 0, "Defect mask should contain positive pixels"
        print("[T02 PASSED] Anomaly image-mask pairing and binary mask verified.")

        # 3. Test DataLoaders
        train_loader, val_loader, test_loader = get_mvtec_dataloaders(
            data_root=temp_dir,
            category=category,
            batch_size=4,
            num_workers=0,  # 0 for local test compatibility
            seed=42,
        )

        batch = next(iter(train_loader))
        assert batch["image"].shape == (4, 3, 256, 256), f"Batch image shape: {batch['image'].shape}"
        assert batch["mask"].shape == (4, 1, 256, 256), f"Batch mask shape: {batch['mask'].shape}"
        assert batch["label"].shape == (4,), f"Batch label shape: {batch['label'].shape}"
        print("[T03 PASSED] DataLoader batch generation and tensor shapes verified.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_mvtec_dataset_and_loader()
