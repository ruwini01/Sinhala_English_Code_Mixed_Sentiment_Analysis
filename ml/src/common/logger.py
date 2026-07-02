"""
Logging setup for training / evaluation runs.

A thin wrapper around the standard library `logging` module that prints
to the console AND appends to a per-run log file under `logs/`. Using a
logger instead of bare print() gives every line a timestamp and a level,
so when a training run takes hours you can see exactly when each epoch
finished and reconstruct what happened afterwards.

Usage:
    from src.common.logger import get_logger
    log = get_logger("baseline_tfidf")
    log.info("Training started")
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"


def get_logger(name: str, log_dir: str = LOG_DIR) -> logging.Logger:
    """Return a logger that writes to console and logs/<name>_<timestamp>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured — avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{stamp}.log"), encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
