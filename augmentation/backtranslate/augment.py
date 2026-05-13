import torch
from sentence_transformers import SentenceTransformer, util
from razdel import sentenize
from tqdm.auto import tqdm

from .translate import safe_translate, clean_bt_result, preprocess_before_translate
from ..common.masks import mask_placeholders, unmask_placeholders
from ..common.perplexity import rugpt_perplexity_list
from ..common.config import SIM_MIN, SIM_MAX

LANG_PAIRS = [("ru-en", "en-ru"), ("ru-fr", "fr-ru"), ("ru-es", "es-ru")]

GEN_MODES = [
    {"do_sample": False, "num_beams": 5},
    {"do_sample": True,  "num_beams": 1, "top_p": 0.90, "temperature": 1.0},
    {"do_sample": True,  "num_beams": 1, "top_p": 0.95, "temperature": 1.2},
]


def generate_bt_candidates(text: str, device) -> list[str]:
    candidates = []
    for src_lang, tgt_lang in LANG_PAIRS:
        for cfg in GEN_MODES:
            mid = safe_translate(text, mode=src_lang, max_tokens=120, device=device, **cfg)
            bt  = safe_translate(mid,  mode=tgt_lang, max_tokens=300, device=device, **cfg)
            candidates.append(clean_bt_result(bt))
    return list(dict.fromkeys([c.strip() for c in candidates if c.strip()]))


def choose_best_bt(
    source_chunk: str,
    candidates: list[str],
    embed_model: SentenceTransformer,
    rugpt_tok,
    rugpt_model,
    rugpt_device: torch.device,
) -> tuple[str, float]:
    if not candidates:
        return source_chunk, 1.0

    texts = [source_chunk] + candidates
    embs = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    src_emb, cand_embs = embs[0], embs[1:]
    sims = util.cos_sim(src_emb, cand_embs)[0]

    mask = (sims >= SIM_MIN) & (sims <= SIM_MAX)

    if mask.any():
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
        filtered = [candidates[i] for i in idxs.tolist()]
        filtered_sims = sims[idxs]
        ppls = rugpt_perplexity_list(filtered, rugpt_tok, rugpt_model, rugpt_device)
        best_local = min(range(len(filtered)), key=lambda i: (ppls[i], -filtered_sims[i].item()))
        return filtered[best_local], filtered_sims[best_local].item()

    best_idx = torch.argmax(sims).item()
    return candidates[best_idx], sims[best_idx].item()


def back_translate_document(
    text_orig: str,
    embed_model: SentenceTransformer,
    rugpt_tok,
    rugpt_model,
    device: torch.device,
) -> str:
    masked_text, mapping = mask_placeholders(text_orig)
    masked_text = preprocess_before_translate(masked_text)
    sentences = [s.text.strip() for s in sentenize(masked_text) if s.text.strip()]

    bt_sentences = []
    for s in tqdm(sentences, desc="Sentences"):
        candidates = generate_bt_candidates(s, device)
        best_text, _ = choose_best_bt(s, candidates, embed_model, rugpt_tok, rugpt_model, device)
        bt_sentences.append(best_text)

    result_masked = " ".join(bt_sentences)
    bt_text = unmask_placeholders(result_masked, mapping)

    return bt_text
