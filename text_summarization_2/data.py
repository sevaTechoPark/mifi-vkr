"""
Загрузка, сохранение и сборка датасетов для пайплайна суммаризации.

Поддерживаемые форматы — CSV, JSON, JSONL (определяются по расширению).
Сборка возвращает два варианта датасета: с заменой текста суммаризацией
и с конкатенацией оригинала и суммаризации через разделитель.
"""

import os

import pandas as pd

from .config import SEPARATOR, TEXT_COLUMN


def load_dataset(path: str) -> pd.DataFrame:
    """Загрузить датасет из CSV или JSON/JSONL по расширению файла."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=(ext == ".jsonl"))
    raise ValueError(
        f"Неподдерживаемый формат файла: {ext}. "
        f"Используйте .csv / .json / .jsonl"
    )


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Сохранить датасет в CSV или JSON/JSONL по расширению пути."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    elif ext == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    print(f"[INFO] Сохранено: {path} ({len(df)} строк)")


def build_summarized_dataset(
    df: pd.DataFrame,
    summaries: list[str],
    text_column: str = TEXT_COLUMN,
) -> pd.DataFrame:
    """
    Сборка датасета, в котором текст заменён суммаризацией.

    Элемент ``summaries[i]`` соответствует строке ``df.iloc[i]``.
    Все остальные колонки (включая метки) сохраняются как есть.
    """
    df_out = df.copy()
    df_out[text_column] = summaries
    return df_out


def build_combined_dataset(
    df: pd.DataFrame,
    summaries: list[str],
    text_column: str = TEXT_COLUMN,
    separator: str = SEPARATOR,
) -> pd.DataFrame:
    """
    Сборка датасета с конкатенацией оригинала и суммаризации.

    В колонке ``text_column`` получается строка
    ``<оригинал><separator><суммаризация>``. Остальные колонки сохраняются.
    """
    df_out = df.copy()
    df_out[text_column] = [
        orig + separator + summ
        for orig, summ in zip(df[text_column].tolist(), summaries)
    ]
    return df_out


def derive_output_paths(input_path: str, output_dir: str) -> tuple[str, str]:
    """
    Сформировать пути выходных файлов на основе имени входного файла.

    Пример::

        input_path = ".../train_augmented.csv"
        → <output_dir>/train_augmented_summarized.csv
        → <output_dir>/train_augmented_original_plus_summary.csv
    """
    basename = os.path.basename(input_path)
    stem, ext = os.path.splitext(basename)

    summarized_path = os.path.join(output_dir, f"{stem}_summarized{ext}")
    combined_path = os.path.join(output_dir, f"{stem}_original_plus_summary{ext}")

    return summarized_path, combined_path
