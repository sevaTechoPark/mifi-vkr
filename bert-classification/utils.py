import gc
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_device_map_location() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_filtered_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {k: v for k, v in state_dict.items() if k != "class_weights"}