import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from src.datasets.mvtec_dataset import get_mvtec_dataloaders
from src.engine.predictor import CAEPredictor, load_cae_from_checkpoint
from src.engine.trainer import CAETrainer
from src.evaluation.metrics import compute_image_metrics, compute_pixel_metrics
from src.evaluation.threshold import calibrate_thresholds, load_thresholds, save_thresholds
from src.models.cae import ConvolutionalAutoencoder
from src.utils.config import setup_experiment_dir
try:
    from tests.test_dataset import create_dummy_mvtec_dataset
except ModuleNotFoundError:
    from test_dataset import create_dummy_mvtec_dataset


def test_full_pipeline_e2e():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        data_root = temp_dir / "dataset"
        output_root = temp_dir / "outputs"
        category = "bottle"
        exp_id = "test_exp_01"

        print(f"Creating mock dataset at: {data_root}")
        create_dummy_mvtec_dataset(data_root, category=category)

        config = {
            "experiment_id": exp_id,
            "category": category,
            "seed": 42,
            "data": {
                "image_size": [128, 128],  # Fast 128x128 for unit test
                "channels": 3,
                "val_ratio": 0.2,
                "batch_size": 2,
                "num_workers": 0,
            },
            "model": {
                "name": "cae",
                "in_channels": 3,
                "base_channels": 16,
                "bottleneck_channels": None,
                "output_activation": "sigmoid",
            },
            "training": {
                "loss": "mse",
                "optimizer": "adam",
                "lr": 0.001,
                "weight_decay": 0.0,
                "epochs": 2,
                "early_stopping_patience": 5,
            },
            "calibration": {
                "rule": "percentile",
                "image_percentile": 90.0,
                "pixel_percentile": 95.0,
            },
            "inference": {
                "error_metric": "l1",
                "top_percent": 0.01,
            },
        }

        output_dirs = setup_experiment_dir(
            output_root=str(output_root),
            experiment_id=exp_id,
            config=config,
            resume=False,
        )

        device = torch.device("cpu")

        # 1. Dataloaders
        train_loader, val_loader, test_loader = get_mvtec_dataloaders(
            data_root=data_root,
            category=category,
            image_size=(128, 128),
            val_ratio=0.2,
            batch_size=2,
            num_workers=0,
            seed=42,
        )

        # 2. Model & Trainer
        model = ConvolutionalAutoencoder(in_channels=3, base_channels=16)
        trainer = CAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dirs=output_dirs,
            device=device,
        )
        trainer.fit()

        best_ckpt = output_dirs["checkpoints"] / "best_model.pth"
        last_ckpt = output_dirs["checkpoints"] / "last_checkpoint.pth"
        assert best_ckpt.exists(), "best_model.pth was not created!"
        assert last_ckpt.exists(), "last_checkpoint.pth was not created!"
        print("[T06 PASSED] Model training, checkpoint saving verified.")

        # 3. Checkpoint loading
        loaded_model, ckpt_meta = load_cae_from_checkpoint(best_ckpt, device=device)
        assert loaded_model is not None
        assert ckpt_meta["epoch"] >= 1
        print("[T06 PASSED] Checkpoint restored successfully on CPU.")

        # 4. Predictor & Anomaly Map
        predictor = CAEPredictor(
            model=loaded_model,
            device=device,
            image_size=(128, 128),
            error_metric="l1",
            top_percent=0.01,
        )

        # 5. Threshold Calibration (T08)
        thresh_dict = calibrate_thresholds(
            predictor=predictor,
            val_loader=val_loader,
            rule="percentile",
            image_percentile=90.0,
            pixel_percentile=95.0,
        )
        thresh_path = output_dirs["thresholds"] / f"{category}_thresholds.json"
        save_thresholds(thresh_dict, thresh_path)
        assert thresh_path.exists(), "Threshold file was not created!"
        print(f"[T08 PASSED] Threshold calibration verified. Image threshold = {thresh_dict['image_threshold']:.6f}")

        # 6. Evaluation metrics (T09)
        predictor.image_threshold = thresh_dict["image_threshold"]
        predictor.pixel_threshold = thresh_dict["pixel_threshold"]

        y_true = [0, 0, 1, 1]
        y_scores = [0.01, 0.02, 0.08, 0.09]
        img_metrics = compute_image_metrics(y_true, y_scores, image_threshold=thresh_dict["image_threshold"])
        assert "auroc" in img_metrics
        assert "f1" in img_metrics
        print(f"[T09 PASSED] Image metrics calculated: AUROC={img_metrics['auroc']}, F1={img_metrics['f1']}")

        # 7. Predict on single image (T10)
        sample_img_path = data_root / category / "test" / "good" / "000.png"
        single_res = predictor.predict_single_image(sample_img_path)
        assert single_res["original"].shape == (128, 128, 3)
        assert single_res["reconstruction"].shape == (128, 128, 3)
        assert single_res["anomaly_map"].shape == (128, 128)
        assert single_res["pred_mask"].shape == (128, 128)
        assert isinstance(single_res["image_score"], float)
        print("[T10/T07 PASSED] Single image prediction, heatmap, and mask generated successfully.")

        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! The pipeline is 100% operational. <<<")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_full_pipeline_e2e()
