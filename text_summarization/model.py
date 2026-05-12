import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_summarization_model(model_name: str):
    print(f"[INFO] Загружаем модель: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Устройство: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device