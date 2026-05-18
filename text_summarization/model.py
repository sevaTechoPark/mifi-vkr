import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_summarization_model(model_name: str):
    print(f"[INFO] Загружаем модель: {model_name}")
    device = _detect_device()
    print(f"[INFO] Устройство: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device
