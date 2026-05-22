"""
Пакетная суммаризация текстов.

Содержит обработку отдельного батча и обёртку, которая проходит по
DataFrame, сохраняя позиционное соответствие: ``summaries[i]`` всегда
соответствует ``df.iloc[i]``. Пустые и NaN-тексты получают пустую
суммаризацию и не отправляются в модель.
"""

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import (
    BATCH_SIZE,
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH,
    TEXT_COLUMN,
)


def _effective_max_input(tokenizer, max_input_length: int) -> int:
    """
    Эффективный лимит входной длины.

    У некоторых токенизаторов ``model_max_length`` хранит «сторожевое»
    значение в районе 1e30, означающее «без ограничения». В таком случае
    используется заданный пользователем ``max_input_length``.
    """
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is None or tok_max > 10_000:
        return max_input_length
    return min(max_input_length, tok_max)


def summarize_batch(
    texts,
    tokenizer,
    model,
    device,
    max_input_length=MAX_INPUT_LENGTH,
    max_summary_length=MAX_SUMMARY_LENGTH,
    min_summary_length=MIN_SUMMARY_LENGTH,
) -> list[str]:
    """Сгенерировать суммаризации для списка текстов одним батчем."""
    if not texts:
        return []

    effective_max_input = _effective_max_input(tokenizer, max_input_length)

    inputs = tokenizer(
        texts,
        max_length=effective_max_input,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=max_summary_length,
            min_new_tokens=min_summary_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)


def summarize_dataframe(
    df,
    tokenizer,
    model,
    device,
    text_column=TEXT_COLUMN,
    batch_size=BATCH_SIZE,
) -> list[str]:
    """
    Сгенерировать суммаризации для всех строк DataFrame.

    Возвращает список длины ``len(df)``; ``summaries[i]`` соответствует
    ``df.iloc[i]``. Пустые и NaN-тексты в модель не подаются и получают
    пустую строку.
    """
    raw_texts = df[text_column].tolist()

    # Сначала собираем индексы и тексты непустых строк, затем обрабатываем
    # их батчами и раскладываем результаты по исходным позициям.
    queue_texts: list[str] = []
    queue_positions: list[int] = []
    out: list[str] = [""] * len(raw_texts)

    for i, t in enumerate(raw_texts):
        if t is None:
            continue
        if isinstance(t, float) and pd.isna(t):
            continue
        s = str(t).strip()
        if not s:
            continue
        queue_texts.append(s)
        queue_positions.append(i)

    for start in tqdm(range(0, len(queue_texts), batch_size), desc="Суммаризация"):
        batch_texts = queue_texts[start:start + batch_size]
        batch_positions = queue_positions[start:start + batch_size]
        summaries = summarize_batch(batch_texts, tokenizer, model, device)
        for pos, summ in zip(batch_positions, summaries):
            out[pos] = summ

    return out
