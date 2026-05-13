import re
import torch
from razdel import sentenize
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

from .models import load_paraphrase_model
from ..common.masks import mask_placeholders, unmask_placeholders
from ..common.text_utils import preprocess_text, clean_generated_text, is_highly_formal

def _get_tokens(text: str, tok, device):
    return tok(text, return_tensors="pt", padding=True, truncation=False).to(device)


def tokens_len(text: str, tok, device) -> int:
    return _get_tokens(text, tok, device).input_ids.shape[1]


def generate_paraphrase(text: str, tok, model, device, **kwargs) -> str:
    x = _get_tokens(text, tok, device)
    src_len = x.input_ids.shape[1]

    if src_len > 400:
        raise ValueError(f"Input too long: {src_len} tokens.")

    # длину держим близко к оригиналу
    max_len = int(src_len * 1.5 + 10)
    min_len = max(2, int(src_len * 0.7))

    gen_kwargs = {
        "encoder_no_repeat_ngram_size": kwargs.pop("encoder_no_repeat_ngram_size", 4),
        "num_beams": kwargs.pop("num_beams", 5),
        "do_sample": kwargs.pop("do_sample", False),  # КЛЮЧЕВОЕ: без сэмплинга
        "max_length": kwargs.pop("max_length", max_len),
        "min_length": kwargs.pop("min_length", min_len),
        # no_repeat_ngram_size можно оставить 3, но при encoder_no_repeat часто не нужен
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
    para_text = generate_paraphrase(source_text, tok, model, device)
    return para_text


def paraphrase_document(
    source_text: str,
    tok,
    model,
    embed_model: SentenceTransformer,
    device: torch.device,
) -> str:
    if is_highly_formal(source_text):
        print('текст слишком формальный пропускаем перефраз')
        return source_text

    masked_text, mapping = mask_placeholders(source_text)
    masked_text = preprocess_text(masked_text)
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
            para_text = generate_best_paraphrase(sub, tok, model, embed_model, device)
            paraphrased_subs.append(para_text)

        paraphrased_sentences.append(" ".join(paraphrased_subs))

    result_masked = " ".join(paraphrased_sentences)
    para_text = unmask_placeholders(result_masked, mapping)

    para_text = clean_generated_text(para_text)

    return para_text
    