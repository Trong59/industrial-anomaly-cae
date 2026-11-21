import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.seed import seed_worker


class MVTecDataset(Dataset):
    """Dataset class for MVTec Anomaly Detection dataset.

    Supports:
        - 'train': 80% of normal samples from train/good (label 0, mask all 0)
        - 'validation': 20% of normal samples from train/good (label 0, mask all 0)
        - 'test': All test samples (good label 0 with all-0 mask, defects label 1 with ground_truth mask)
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        category: str = "bottle",
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.category = category
        self.split = split
        self.image_size = tuple(image_size)
        self.val_ratio = val_ratio
        self.seed = seed

        self.category_dir = self.data_root / category
        if not self.category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {self.category_dir}")

        # Image transformation: resize -> RGB -> normalize to [0, 1]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # Scales to [0.0, 1.0]
            ]
        )

        # Mask transformation: resize with NEAREST -> ToTensor -> binary threshold
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if self.split in ["train", "validation"]:
            train_good_dir = self.category_dir / "train" / "good"
            if not train_good_dir.exists():
                raise FileNotFoundError(f"Train good directory not found: {train_good_dir}")

            all_images = sorted(
                [p for p in train_good_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
            )
            if not all_images:
                raise ValueError(f"No images found in {train_good_dir}")

            # Deterministic train/validation split
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(all_images))
            rng.shuffle(indices)

            val_size = max(1, int(len(all_images) * self.val_ratio))
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]

            selected_indices = train_indices if self.split == "train" else val_indices
            selected_files = [all_images[i] for i in sorted(selected_indices)]

            for img_path in selected_files:
                self.samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": None,
                        "label": 0,  # Normal
                        "defect_type": "good",
                    }
                )

        elif self.split == "test":
            test_dir = self.category_dir / "test"
            gt_dir = self.category_dir / "ground_truth"

            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            # Iterate over all defect folders inside test/
            for defect_dir in sorted([d for d in test_dir.iterdir() if d.is_dir()]):
                defect_type = defect_dir.name
                images = sorted(
                    [p for p in defect_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
                )

                for img_path in images:
                    if defect_type == "good":
                        self.samples.append(
                            {
                                "image_path": img_path,
                                "mask_path": None,
                                "label": 0,  # Normal
                                "defect_type": "good",
                            }
                        )
                    else:
                        # Find corresponding ground truth mask
                        # MVTec AD formats: <stem>_mask.png or <stem>.png
                        mask_path = None
                        candidates = [
                            gt_dir / defect_type / f"{img_path.stem}_mask.png",
                            gt_dir / defect_type / f"{img_path.stem}_mask.PNG",
                            gt_dir / defect_type / f"{img_path.stem}.png",
                            gt_dir / defect_type / f"{img_path.stem}.PNG",
                        ]
                        for c in candidates:
                            if c.exists():
                                mask_path = c
                                break

                        if mask_path is None:
                            raise FileNotFoundError(
                                f"Ground truth mask not found for anomaly image: {img_path}. "
                                f"Checked: {[str(c) for c in candidates]}"
                            )

                        self.samples.append(
                            {
                                "image_path": img_path,
                                "mask_path": mask_path,
                                "label": 1,  # Anomalous
                                "defect_type": defect_type,
                            }
                        )
        else:
            raise ValueError(f"Unknown split: '{self.split}'. Must be 'train', 'validation', or 'test'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        sample = self.samples[idx]

        # Load RGB image
        with Image.open(sample["image_path"]) as img:
            img = img.convert("RGB")
            image_tensor = self.image_transform(img)

        # Load or generate binary mask (1, H, W)
        if sample["mask_path"] is not None:
            with Image.open(sample["mask_path"]) as m:
                m = m.convert("L")
                mask_tensor = self.mask_transform(m)
                mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Good samples have all-zero ground truth mask
            mask_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "defect_type": sample["defect_type"],
            "image_path": str(sample["image_path"]),
        }


def get_mvtec_dataloaders(
    data_root: Union[str, Path],
    category: str = "bottle",
    image_size: Tuple[int, int] = (256, 256),
    val_ratio: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible DataLoaders for train, validation, and test splits."""
    train_dataset = MVTecDataset(
        data_root=data_root,
        category=category,
        split="train",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    val_dataset = MVTecDataset(
        data_root=data_root,
        category=category,
        split="validation",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    test_dataset = MVTecDataset(
        data_root=data_root,
        category=category,
        split="test",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
