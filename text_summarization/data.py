import os
import pandas as pd

from .config import TEXT_COLUMN, LABEL_COLUMN, SEPARATOR


def load_dataset(path: str) -> pd.DataFrame:
    """Загружает датасет из CSV или JSON/JSONL по расширению файла."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=(ext == ".jsonl"))
    else:
        raise ValueError(
            f"Неподдерживаемый формат файла: {ext}. Используйте .csv / .json / .jsonl"
        )


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Сохраняет датасет в CSV или JSON/JSONL по расширению пути."""
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
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """
    Датасет 1: текст заменяется суммаризацией, метки сохраняются.
    summaries[i] строго соответствует df.iloc[i].
    """
    df_out = df.copy()
    df_out[text_column] = summaries
    return df_out


def build_combined_dataset(
    df: pd.DataFrame,
    summaries: list[str],
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    separator: str = SEPARATOR,
) -> pd.DataFrame:
    """
    Датасет 2: оригинал + суммаризация через разделитель, метки сохраняются.
    summaries[i] строго соответствует df.iloc[i].
    """
    df_out = df.copy()
    df_out[text_column] = [
        orig + separator + summ
        for orig, summ in zip(df[text_column].tolist(), summaries)
    ]
    return df_out


def derive_output_paths(input_path: str, output_dir: str) -> tuple[str, str]:
    """
    Формирует имена выходных файлов на основе имени входного файла.

    Пример:
        input_path = ".../train_augmented.csv"
        →  <output_dir>/train_augmented_summarized.csv
        →  <output_dir>/train_augmented_original_plus_summary.csv
    """
    basename = os.path.basename(input_path)
    stem, ext = os.path.splitext(basename)

    summarized_path = os.path.join(output_dir, f"{stem}_summarized{ext}")
    combined_path   = os.path.join(output_dir, f"{stem}_original_plus_summary{ext}")

    return summarized_path, combined_path