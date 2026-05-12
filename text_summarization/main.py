# main.py — точка входа модуля

import os
import pandas as pd

from .config import (
    SUMMARIZATION_MODEL,
    TEXT_COLUMN,
    LABEL_COLUMN,
    BATCH_SIZE,
    SEPARATOR,
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH,
)
from .model import load_summarization_model
from .summarize import summarize_dataframe
from .data import (
    load_dataset,
    save_dataset,
    build_summarized_dataset,
    build_combined_dataset,
    derive_output_paths,
)


def run(
    input_path: str,
    output_dir: str,
    model_name: str = SUMMARIZATION_MODEL,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    batch_size: int = BATCH_SIZE,
    separator: str = SEPARATOR,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_summary_length: int = MAX_SUMMARY_LENGTH,
    min_summary_length: int = MIN_SUMMARY_LENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основная функция модуля.

    Args:
        input_path:  путь к исходному датасету (.csv / .json / .jsonl)
        output_dir:  директория для сохранения результатов

    Returns:
        (df_summarized, df_combined) — два итоговых датафрейма

    Сохраняет два файла вида:
        <output_dir>/<stem>_summarized<ext>
        <output_dir>/<stem>_original_plus_summary<ext>
    """
    # 1. Загрузка
    print(f"[INFO] Загружаем датасет: {input_path}")
    df = load_dataset(input_path)
    df = df.reset_index(drop=True)
    df["sample_id"] = df.index

    print(f"[INFO] Загружено строк: {len(df)}, колонки: {list(df.columns)}")

    if text_column not in df.columns:
        raise ValueError(
            f"Колонка '{text_column}' не найдена. Доступные: {list(df.columns)}"
        )

    # 2. Загрузка модели
    tokenizer, model, device = load_summarization_model(model_name)

    # 3. Суммаризация
    summaries = summarize_dataframe(
        df,
        tokenizer,
        model,
        device,
        text_column=text_column,
        batch_size=batch_size,
    )

    # 4. Построение датасетов
    df_summarized = build_summarized_dataset(df, summaries, text_column, label_column)
    df_combined   = build_combined_dataset(df, summaries, text_column, label_column, separator)

    # 5. Сохранение
    summarized_path, combined_path = derive_output_paths(input_path, output_dir)

    save_dataset(df_summarized, summarized_path)
    save_dataset(df_combined,   combined_path)

    print("[INFO] Готово!")
    print(f"  → Суммаризированный:       {summarized_path}")
    print(f"  → Оригинал + суммаризация: {combined_path}")

    return df_summarized, df_combined