import torch
from transformers import MarianMTModel, MarianTokenizer

PAIRS = {
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "ru-fr": "Helsinki-NLP/opus-mt-ru-fr",
    "fr-ru": "Helsinki-NLP/opus-mt-fr-ru",
    "ru-es": "Helsinki-NLP/opus-mt-ru-es",
    "es-ru": "Helsinki-NLP/opus-mt-es-ru",
}

_cache: dict = {}


def get_model_by_mode(mode: str, device: torch.device):
    if mode not in _cache:
        name = PAIRS[mode]
        tok = MarianTokenizer.from_pretrained(name)
        mdl = MarianMTModel.from_pretrained(name).to(device)
        _cache[mode] = (tok, mdl)
    return _cache[mode]