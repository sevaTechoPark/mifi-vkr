import torch
import pandas as pd
from tqdm.auto import tqdm

from .config import (
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH,
    BATCH_SIZE,
    TEXT_COLUMN,
)


def _effective_max_input(tokenizer, max_input_length: int) -> int:
    tok_max = getattr(tokenizer, "model_max_length", None)
    # у некоторых токенайзеров model_max_length = очень большое число (вроде 1e30)
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
    Возвращает список summaries длины len(df). summaries[i] строго соответствует df.iloc[i].
    Пустые/NaN-тексты получают пустую строку и не уходят в модель.
    """
    raw_texts = df[text_column].tolist()

    # Готовим параллельные списки: что отдаём в модель + куда положить ответ
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