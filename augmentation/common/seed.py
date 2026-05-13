import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Фиксирует все основные источники рандома:
    - Python random
    - хеши Python
    - NumPy
    - PyTorch (CPU и CUDA)

    deterministic=True включает максимально детерминированный режим
    в духе рекомендаций PyTorch (отключаем benchmark, включаем deterministic). [web:3][web:7]
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # для CuDNN / общих бекендов — максимально детерминированное поведение
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # при наличии PyTorch ≥ 2.0 можно дополнительно включить:
        # torch.use_deterministic_algorithms(True)


def get_seed_or_default(seed: Optional[int] = None, default: int = 42) -> int:
    """
    Утилита: если seed не задан (None) — возвращает дефолтный.
    Удобно использовать вместе с CLI-параметром.
    """
    return default if seed is None else int(seed)