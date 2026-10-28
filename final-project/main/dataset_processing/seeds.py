"""
Random seeds used
"""

import os
import random

import numpy as np
import torch

GLOBAL_SEED = 42


def set_seed(seed: int = 42):
    """
    Set all seeds for training script
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


DATA_PREPARE_READ_SHUFFLE = 1

SQUIGULATOR_SEED = 42
