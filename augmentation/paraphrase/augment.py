# augmentation/paraphrase/augment.py

import torch
from razdel import sentenize
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

from ..common.masks import mask_placeholders, unmask_placeholders
from ..common.text_utils import (
    preprocess_text,
    is_highly_formal,
)
from ..common.config import (
    PARA_SIM_MIN,
    PARA_SIM_MAX,
    PARA_MIN_LEN_RATIO,
    PARA_MAX_LEN_RATIO,
)
from .phrase import safe_paraphrase, postprocess_paraphrase_text


# несколько режимов генерации, как GEN_MODES в BT:
GEN_MODES = [
    # базовый консервативный beam-search
    {"do_sample": False, "num_beams": 5},
    # чуть более «жирный» beam-search (немного разнообразия, но без sampling)
    {"do_sample": False, "num_beams": 8},
    # мягкий sampling, если захочешь больше разнообразия
    {"do_sample": True, "num_beams": 1, "top_p": 0.90, "temperature": 1.0},
]


def generate_para_candidates(
    text: str,
    tok,
    model,
    device: torch.device,
) -> list[str]:
    """
    Генерируем несколько кандидатов перефраза для одного предложения.
    Разделение на чанки по длине делается внутри safe_paraphrase.
    """
    candidates: list[str] = []

    for cfg in GEN_MODES:
        para = safe_paraphrase(
            text=text,
            tok=tok,
            model=model,
            max_tokens=120,
            device=device,
            **cfg,
        )
        para = para.strip()
        if para:
            candidates.append(para)

    # дедуп с сохранением порядка
    return list(dict.fromkeys(candidates))


def choose_best_paraphrase(
    source_chunk: str,
    candidates: list[str],
    embed_model: SentenceTransformer,
) -> tuple[str, float]:
    """
    Выбор лучшего перефраза по:
      - окну cosine (PARA_SIM_MIN..PARA_SIM_MAX),
      - ограничению по соотношению длин (PARA_MIN/MAX_LEN_RATIO).
    """
    if not candidates:
        return source_chunk

    len_src = len(source_chunk)
    filtered_by_len: list[str] = []
    for c in candidates:
        r = len(c) / max(1, len_src)
        if PARA_MIN_LEN_RATIO <= r <= PARA_MAX_LEN_RATIO:
            filtered_by_len.append(c)
    if not filtered_by_len:
        filtered_by_len = candidates

    texts = [source_chunk] + filtered_by_len
    embs = embed_model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    src_emb, cand_embs = embs[0], embs[1:]
    sims = util.cos_sim(src_emb, cand_embs)[0]

    # ищем кандидата в «хорошем» окне похожести
    mask = (sims >= PARA_SIM_MIN) & (sims <= PARA_SIM_MAX)
    if mask.any():
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
        best_local = torch.argmax(sims[idxs]).item()
        best_idx = idxs[best_local].item()
        best_text = filtered_by_len[best_idx]
        best_text = postprocess_paraphrase_text(best_text)
        return best_text

    # иначе берём просто максимально похожий
    best_idx = torch.argmax(sims).item()
    best_text = filtered_by_len[best_idx]
    best_text = postprocess_paraphrase_text(best_text)
    return best_text


def paraphrase_document(
    source_text: str,
    tok,
    model,
    embed_model: SentenceTransformer,
    device: torch.device,
) -> str:
    """
    Высокоуровневый проход по документу:
    - гейт по формальности,
    - маскировка плейсхолдеров,
    - разбиение на предложения,
    - для каждого предложения: safe_paraphrase + выбор лучшего кандидата,
    - демаскировка и финальная пост-обработка.
    """
    if is_highly_formal(source_text):
        print("текст слишком формальный пропускаем перефраз")
        return source_text

    masked_text, mapping = mask_placeholders(source_text)
    masked_text = preprocess_text(masked_text)
    sentences = [s.text for s in sentenize(masked_text)]
    paraphrased_sentences: list[str] = []

    for i, s in enumerate(
        tqdm(sentences, total=len(sentences), desc="Sentences", position=0),
        1,
    ):
        s = s.strip()
        if not s:
            continue

        candidates = generate_para_candidates(s, tok, model, device)
        best_text = choose_best_paraphrase(
            source_chunk=s,
            candidates=candidates,
            embed_model=embed_model,
        )
        paraphrased_sentences.append(best_text)

    result_masked = " ".join(paraphrased_sentences)
    para_text = unmask_placeholders(result_masked, mapping)
    para_text = postprocess_paraphrase_text(para_text)

    return para_text