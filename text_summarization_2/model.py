"""Загрузка seq2seq-модели суммаризации и подбор вычислительного устройства."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _detect_device() -> str:
    """Подобрать устройство: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_summarization_model(model_name: str):
    """
    Загрузить токенизатор и модель суммаризации с HuggingFace Hub.

    Возвращает тройку ``(tokenizer, model, device)``. Модель переведена
    в режим ``eval`` и размещена на подобранном устройстве.
    """
    print(f"[INFO] Загружаем модель: {model_name}")
    device = _detect_device()
    print(f"[INFO] Устройство: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device
