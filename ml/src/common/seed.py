"""
Global seed control for reproducible experiments.

Call set_seed(42) at the very top of every training / evaluation script,
BEFORE building models or data loaders. This pins every source of
randomness (Python, NumPy, PyTorch, CUDA) so that two runs of the same
code produce the same numbers — which is what makes your results
defensible in a viva ("I can reproduce this exactly").

Usage:
    from src.common.seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch + CUDA.

    Returns the seed so it can be logged with the run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch is optional — the sklearn baseline (B1) does not need it.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # deterministic cuDNN: slower but reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
