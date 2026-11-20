import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional


def get_logger(name: str = "CAE", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard logger printing to console and optionally to file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class CSVMetricsLogger:
    """Logs training and evaluation metrics into a CSV file."""

    def __init__(self, csv_path: Path, fieldnames: List[str]):
        self.csv_path = Path(csv_path)
        self.fieldnames = fieldnames
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Write header if file does not exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, metrics: Dict[str, float or int or str]) -> None:
        """Append a row of metrics to the CSV file."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)
