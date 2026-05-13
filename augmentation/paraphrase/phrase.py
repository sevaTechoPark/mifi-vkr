# augmentation/paraphrase/phrase.py

import re
import torch

from ..common.text_utils import clean_aug_result, clean_generated_text


def _get_tokens(text: str, tok, device: torch.device):
    return tok(text, return_tensors="pt", padding=True, truncation=False).to(device)


def tokens_len(text: str, tok, device: torch.device) -> int:
    return _get_tokens(text, tok, device).input_ids.shape[1]


def split_long_sentence(
    sent: str,
    tok,
    device: torch.device,
    max_tokens: int = 120,
) -> list[str]:
    """
    Делим потенциально длинное предложение на чанки, которые
    не превышают max_tokens по длине (в токенах T5).
    """
    if tokens_len(sent, tok, device) <= max_tokens:
        return [sent]

    parts = re.split(r"(?<=[,;:—-])\s+", sent)
    chunks: list[str] = []

    if len(parts) == 1:
        # нет удобных разделителей — режем по словам
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
                # даже отдельная часть слишком длинная — режем по словам
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


def generate_phrase(
    text: str,
    tok,
    model,
    device: torch.device,
    **gen_kwargs,
) -> str:
    """
    Один проход ruT5 для уже подготовленного куска текста.
    Здесь не режем текст, только проверяем upper-bound по длине (300 токенов),
    как в generate_translate для BT.[web:50]
    """
    x = _get_tokens(text, tok, device)
    src_len = x.input_ids.shape[1]

    if src_len > 300:
        raise ValueError(
            f"Input text too long for paraphrasing: {src_len} tokens (max 300). "
            f"Use split_long_sentence() / safe_paraphrase() before calling generate_phrase()."
        )

    max_len = int(src_len * 1.5 + 10)
    min_len = max(2, int(src_len * 0.7))

    default_gen_kwargs = dict(
        encoder_no_repeat_ngram_size=4,
        num_beams=5,
        do_sample=False,
        max_length=max_len,
        min_length=min_len,
    )
    default_gen_kwargs.update(gen_kwargs)

    with torch.no_grad():
        outputs = model.generate(
            **x,
            **default_gen_kwargs,
        )

    return (
        tok.decode(outputs[0], skip_special_tokens=True)
        .replace("\n", " ")
        .strip()
    )


def safe_paraphrase(
    text: str,
    tok,
    model,
    max_tokens: int,
    device: torch.device,
    **gen_kwargs,
) -> str:
    """
    Безопасный перефраз длинных текстов:
    - сначала режет предложение до max_tokens (split_long_sentence),
    - затем для каждого куска вызывает generate_phrase,
    - при крайней необходимости режет по словам фиксированным окном.
    """
    chunks = split_long_sentence(text, tok=tok, device=device, max_tokens=max_tokens)
    out_chunks: list[str] = []

    for ch in chunks:
        ch_len = tokens_len(ch, tok=tok, device=device)
        if ch_len > 300:
            # крайний случай: всё ещё длинно — режем грубо по словам фиксированным окном
            words = ch.split()
            for i in range(0, len(words), 20):
                sub = " ".join(words[i : i + 20])
                if sub.strip():
                    out_chunks.append(
                        generate_phrase(sub, tok=tok, model=model, device=device, **gen_kwargs)
                    )
        else:
            out_chunks.append(
                generate_phrase(ch, tok=tok, model=model, device=device, **gen_kwargs)
            )

    return " ".join(out_chunks).strip()


def postprocess_paraphrase_text(text: str) -> str:
    """
    Общая нормализация одного перефраза:
    сначала bt-нормализация, затем мягкая чистка.
    """
    text = clean_aug_result(text)
    text = clean_generated_text(text)
    return text