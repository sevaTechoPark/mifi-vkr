import re
import torch

from .models import get_model_by_mode


def mt_tokens_len(text: str, mode: str, device: torch.device) -> int:
    """
    Подсчёт длины текста в токенах для конкретной модели перевода.
    Используется для безопасного разбиения длинных предложений.
    """
    tokenizer, _ = get_model_by_mode(mode, device)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    return inputs.input_ids.shape[1]


def generate_translate(
    text: str,
    mode: str,
    device: torch.device,
    **gen_kwargs,
) -> str:
    """
    Один проход перевода для уже подготовленного куска текста.
    Здесь не режем текст, только проверяем upper-bound по длине (300 токенов).
    """
    tokenizer, model = get_model_by_mode(mode, device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=True,
    ).to(device)

    input_len = inputs.input_ids.shape[1]
    if input_len > 300:
        raise ValueError(
            f"Input text too long for translation: {input_len} tokens (max 300). "
            f"Use split_long_sentence() before calling generate_translate()."
        )

    default_gen_kwargs = dict(
        num_beams=1,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=1.1,
    )
    default_gen_kwargs.update(gen_kwargs)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            **default_gen_kwargs,
        )

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


def split_long_sentence(sent: str, max_tokens: int, mode: str, device: torch.device) -> list[str]:
    """
    Рекурсивно режет слишком длинное предложение на части, чтобы
    каждая часть имела длину ≤ max_tokens в токенах переводческой модели.
    """
    if mt_tokens_len(sent, mode=mode, device=device) <= max_tokens:
        return [sent]

    # пробуем разделить по знакам препинания
    parts = re.split(r"(?<=[,;:—\-])\s+", sent)

    chunks: list[str] = []

    # если разделителей нет — режем по словам
    if len(parts) == 1:
        words = sent.split()
        current = ""
        for w in words:
            candidate = f"{current} {w}".strip() if current else w
            if mt_tokens_len(candidate, mode=mode, device=device) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)
        return chunks if chunks else [sent]

    # есть разделители — собираем по частям, при необходимости рекурсивно
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if mt_tokens_len(candidate, mode=mode, device=device) <= max_tokens:
            current = candidate
        else:
            if current:
                # текущий чанк мог получиться чуть больше max_tokens — подрежем рекурсивно
                chunks.extend(split_long_sentence(current, max_tokens, mode, device))
            # новая часть сама по себе может быть длинной
            if mt_tokens_len(part, mode=mode, device=device) > max_tokens:
                chunks.extend(split_long_sentence(part, max_tokens, mode, device))
                current = ""
            else:
                current = part

    if current:
        chunks.extend(split_long_sentence(current, max_tokens, mode, device))

    return chunks if chunks else [sent]


def safe_translate(text: str, mode: str, max_tokens: int, device: torch.device, **gen_kwargs) -> str:
    """
    Безопасный перевод длинных текстов:
    - не вызывает generate_translate на кусках > 300 токенов,
    - сначала режет предложение до max_tokens, а при крайней необходимости — по словам.
    """
    chunks = split_long_sentence(text, max_tokens=max_tokens, mode=mode, device=device)
    out_chunks: list[str] = []

    for ch in chunks:
        ch_len = mt_tokens_len(ch, mode=mode, device=device)
        if ch_len > 300:
            # крайний случай: всё ещё длинно — режем грубо по словам фиксированным окном
            words = ch.split()
            for i in range(0, len(words), 20):
                sub = " ".join(words[i : i + 20])
                if sub.strip():
                    out_chunks.append(generate_translate(sub, mode=mode, device=device, **gen_kwargs))
        else:
            out_chunks.append(generate_translate(ch, mode=mode, device=device, **gen_kwargs))

    return " ".join(out_chunks)
