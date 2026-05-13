import torch
from sentence_transformers import SentenceTransformer, util
from .config import EMBED_MODEL_NAME

def load_embed_model(device: torch.device) -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME, device=device)


def cos_sim(model: SentenceTransformer, text1: str, text2: str) -> float:
    emb1 = model.encode(text1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = model.encode(text2, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb1, emb2).item()