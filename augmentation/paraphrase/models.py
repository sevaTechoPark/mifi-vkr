import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PARAPHRASE_MODEL_NAME = "fyaronskiy/ruT5-large-paraphraser"

_tok = None
_model = None


def load_paraphrase_model(device: torch.device):
    global _tok, _model
    if _tok is None:
        _tok   = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL_NAME).to(device)
    return _tok, _model