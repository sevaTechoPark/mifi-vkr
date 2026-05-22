"""
Точка входа пайплайна суммаризации.

Загружает датасет, прогоняет тексты через seq2seq-модель и сохраняет
два варианта результата:

  * ``<stem>_summarized<ext>``              — текст заменён суммаризацией;
  * ``<stem>_original_plus_summary<ext>``   — оригинал и суммаризация
    конкатенированы через разделитель.
"""

import argparse
from pathlib import Path

import pandas as pd

from .config import (
    BATCH_SIZE,
    LABEL_COLUMN,
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH,
    SEPARATOR,
    SUMMARIZATION_MODEL,
    TEXT_COLUMN,
)
from .data import (
    build_combined_dataset,
    build_summarized_dataset,
    derive_output_paths,
    load_dataset,
    save_dataset,
)
from .model import load_summarization_model
from .summarize import summarize_dataframe


def run(
    input_path: str | Path,
    output_dir: str | Path,
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
    Запустить пайплайн суммаризации.

    Параметры:
      input_path   — путь к исходному датасету (.csv / .json / .jsonl);
      output_dir   — каталог для выходных файлов;
      model_name   — идентификатор модели на HuggingFace Hub;
      text_column  — имя колонки с текстом;
      label_column — имя колонки с меткой (для совместимости интерфейса;
                     внутри пайплайна метки переносятся как есть, отдельной
                     обработки им не нужно);
      batch_size   — размер батча генерации;
      separator    — разделитель оригинала и суммаризации в комбинированном
                     датасете.

    Возвращает пару ``(df_summarized, df_combined)``; те же датасеты
    сохраняются на диск.
    """
    # label_column оставлен для совместимости публичного интерфейса;
    # внутри он не нужен, поэтому не используется явно.
    del label_column

    input_path = str(input_path)
    output_dir = str(output_dir)

    print(f"[INFO] Загружаем датасет: {input_path}")
    df = load_dataset(input_path)
    df = df.reset_index(drop=True)
    df["sample_id"] = df.index

    print(f"[INFO] Загружено строк: {len(df)}, колонки: {list(df.columns)}")

    if text_column not in df.columns:
        raise ValueError(
            f"Колонка '{text_column}' не найдена. Доступные: {list(df.columns)}"
        )

    tokenizer, model, device = load_summarization_model(model_name)

    summaries = summarize_dataframe(
        df,
        tokenizer,
        model,
        device,
        text_column=text_column,
        batch_size=batch_size,
    )

    df_summarized = build_summarized_dataset(df, summaries, text_column)
    df_combined = build_combined_dataset(df, summaries, text_column, separator)

    summarized_path, combined_path = derive_output_paths(input_path, output_dir)

    save_dataset(df_summarized, summarized_path)
    save_dataset(df_combined, combined_path)

    print("[INFO] Готово!")
    print(f"  → Суммаризированный:       {summarized_path}")
    print(f"  → Оригинал + суммаризация: {combined_path}")

    return df_summarized, df_combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text summarization pipeline")
    parser.add_argument(
        "--input-path", type=Path, required=True,
        help="Path to input dataset (.csv, .json, .jsonl)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory where output files will be saved",
    )
    parser.add_argument(
        "--model-name", type=str, default=SUMMARIZATION_MODEL,
        help="Hugging Face model name for summarization",
    )
    parser.add_argument(
        "--text-column", type=str, default=TEXT_COLUMN,
        help="Name of text column",
    )
    parser.add_argument(
        "--label-column", type=str, default=LABEL_COLUMN,
        help="Name of label column",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Batch size for summarization",
    )
    parser.add_argument(
        "--separator", type=str, default=SEPARATOR,
        help="Separator between original text and summary in combined dataset",
    )
    parser.add_argument(
        "--max-input-length", type=int, default=MAX_INPUT_LENGTH,
        help="Maximum input token length",
    )
    parser.add_argument(
        "--max-summary-length", type=int, default=MAX_SUMMARY_LENGTH,
        help="Maximum generated summary length",
    )
    parser.add_argument(
        "--min-summary-length", type=int, default=MIN_SUMMARY_LENGTH,
        help="Minimum generated summary length",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        text_column=args.text_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        separator=args.separator,
        max_input_length=args.max_input_length,
        max_summary_length=args.max_summary_length,
        min_summary_length=args.min_summary_length,
    )


if __name__ == "__main__":
    main()
