import re
import torch
from razdel import sentenize
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

from .models import load_paraphrase_model
from ..common.masks import mask_placeholders, unmask_placeholders, placeholders_intact
from ..common.config import SIM_MIN, SIM_MAX


def _get_tokens(text: str, tok, device):
    return tok(text, return_tensors="pt", padding=True, truncation=False).to(device)


def tokens_len(text: str, tok, device) -> int:
    return _get_tokens(text, tok, device).input_ids.shape[1]


def generate_paraphrase(text: str, tok, model, device, **kwargs) -> str:
    x = _get_tokens(text, tok, device)
    src_len = x.input_ids.shape[1]

    if src_len > 400:
        raise ValueError(f"Input too long: {src_len} tokens.")

    if src_len < 30:
        min_len = max(2, int(src_len * 0.7))
        max_len = min(512, int(src_len * 1.5))
        default_temp, default_top_p = 0.95, 0.92
    else:
        min_len = max(2, int(src_len * 0.5))
        max_len = min(512, max(min_len + 1, int(src_len * 2.0)))
        default_temp, default_top_p = 1.1, 0.90

    gen_kwargs = {
        "encoder_no_repeat_ngram_size": 3,
        "max_length": max_len,
        "min_length": min_len,
        "no_repeat_ngram_size": 3,
        "do_sample": True,
        "num_return_sequences": 1,
        "top_k": 50,
        "top_p": kwargs.pop("top_p", default_top_p),
        "temperature": kwargs.pop("temperature", default_temp),
    }
    gen_kwargs.update(kwargs)

    output = model.generate(**x, **gen_kwargs)
    return tok.decode(output[0], skip_special_tokens=True)


def split_long_sentence(sent: str, tok, device, max_tokens: int = 120) -> list[str]:
    if tokens_len(sent, tok, device) <= max_tokens:
        return [sent]

    parts = re.split(r"(?<=[,;:—-])\s+", sent)
    chunks = []

    if len(parts) == 1:
        words = sent.split()
        current = ""
        for w in words:
            candidate = f"{current} {w}".strip() if current else w
            if tokens_len(candidate, tok, device) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)
        return chunks or [sent]

    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if tokens_len(candidate, tok, device) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if tokens_len(part, tok, device) > max_tokens:
                words = part.split()
                sub = ""
                for w in words:
                    sc = f"{sub} {w}".strip() if sub else w
                    if tokens_len(sc, tok, device) <= max_tokens:
                        sub = sc
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = w
                current = sub
            else:
                current = part
    if current:
        chunks.append(current)

    return chunks or [sent]


def generate_best_paraphrase(
    source_text: str,
    tok,
    model,
    embed_model: SentenceTransformer,
    device: torch.device,
) -> tuple[str, float]:
    src_len = tokens_len(source_text, tok, device)
    is_short = src_len < 70
    n_samples  = 7 if is_short else 3
    max_retries = 5 if is_short else 3
    base_temp  = 0.95 if is_short else 1.1

    source_emb = embed_model.encode(
        source_text, convert_to_tensor=True, normalize_embeddings=True
    )

    best_fallback_text = None
    best_fallback_sim  = -1.0

    for retry in range(max_retries):
        if retry == 0 and is_short:
            candidates = [
                generate_paraphrase(source_text, tok, model, device,
                                    temperature=base_temp, top_k=30, top_p=0.85,
                                    repetition_penalty=1.1)
                for _ in range(n_samples)
            ]
        elif retry == 0:
            candidates = [
                generate_paraphrase(source_text, tok, model, device)
                for _ in range(n_samples)
            ]
        elif is_short or max_retries - retry == 1:
            temp = base_temp + 0.05 * retry
            candidates = [
                generate_paraphrase(source_text, tok, model, device,
                                    temperature=temp, top_k=30, top_p=0.85,
                                    repetition_penalty=1.1)
                for _ in range(n_samples)
            ]
        else:
            temp = base_temp + 0.1 * retry
            candidates = [
                generate_paraphrase(source_text, tok, model, device, temperature=temp)
                for _ in range(n_samples)
            ]

        cand_embs = embed_model.encode(
            candidates, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = util.cos_sim(source_emb, cand_embs)[0]

        cur_best = torch.argmax(sims).item()
        if sims[cur_best].item() > best_fallback_sim:
            best_fallback_sim  = sims[cur_best].item()
            best_fallback_text = candidates[cur_best]

        mask = (sims >= SIM_MIN) & (sims <= SIM_MAX)
        if mask.any():
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            best_in_mask = torch.argmax(sims[idxs]).item()
            best_idx = idxs[best_in_mask].item()
            return candidates[best_idx], sims[best_idx].item()

    return best_fallback_text, best_fallback_sim


def paraphrase_document(
    source_text: str,
    tok,
    model,
    embed_model: SentenceTransformer,
    device: torch.device,
) -> str:
    masked_text, mapping = mask_placeholders(source_text)
    sentences = [s.text for s in sentenize(masked_text)]
    paraphrased_sentences = []

    for i, s in enumerate(
        tqdm(sentences, total=len(sentences), desc="Sentences", position=0), 1
    ):
        subs = split_long_sentence(s, tok, device, max_tokens=120)
        paraphrased_subs = []

        for sub in tqdm(
            subs,
            total=len(subs),
            desc=f"Chunks {i}/{len(sentences)}",
            position=1,
            leave=False,
        ):
            para_text, _ = generate_best_paraphrase(sub, tok, model, embed_model, device)
            paraphrased_subs.append(para_text)

        paraphrased_sentences.append(" ".join(paraphrased_subs))

    result_masked = " ".join(paraphrased_sentences)
    para_text = unmask_placeholders(result_masked, mapping)

    return para_text
    