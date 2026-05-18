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
    """
    Возвращает state_dict без `class_weights` (он зарегистрирован как buffer и
    зависит от датасета — не имеет смысла хранить его вместе с весами).
    При загрузке такого state_dict обратно использовать strict=False.
    """
    state_dict = model.state_dict()
    return {k: v for k, v in state_dict.items() if k != "class_weights"}


def load_state_dict_into_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Безопасная загрузка отфильтрованного state_dict.
    strict=False — потому что `class_weights` в модели ЕСТЬ (buffer),
    а в state_dict его НЕТ (отфильтрован).
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # class_weights — единственный ожидаемый missing
    real_missing = [k for k in missing if k != "class_weights"]
    if real_missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch.\n"
            f"  unexpected: {unexpected}\n"
            f"  missing:    {real_missing}"
        )