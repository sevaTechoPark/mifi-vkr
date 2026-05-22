"""Утилиты: воспроизводимость, очистка памяти, фильтрация state_dict."""

import gc
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_global_seed(seed: int) -> None:
    """Фиксирует генераторы Python / NumPy / PyTorch для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory() -> None:
    """Сбор мусора и освобождение кэша CUDA — полезно между прогонами."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: str) -> Path:
    """Создаёт каталог (вместе с родителями) и возвращает Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_device_map_location() -> str:
    """Строка устройства для torch.load(map_location=...)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_filtered_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """state_dict модели без буфера `class_weights`.

    class_weights зависит от обучающего датасета и пересчитывается при каждом
    запуске — хранить его вместе с весами не имеет смысла. Загружать такой
    отфильтрованный state_dict нужно через load_state_dict_into_model.
    """
    state_dict = model.state_dict()
    return {k: v for k, v in state_dict.items() if k != "class_weights"}


def load_state_dict_into_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Загружает отфильтрованный state_dict с правильной обработкой class_weights.

    strict=False обязательно: в модели `class_weights` присутствует (зарегистрирован
    как buffer), а в state_dict — нет (отфильтрован специально). Любые другие
    отсутствующие или лишние ключи считаются ошибкой и приводят к исключению.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if k != "class_weights"]
    if real_missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch.\n"
            f"  unexpected: {unexpected}\n"
            f"  missing:    {real_missing}"
        )
