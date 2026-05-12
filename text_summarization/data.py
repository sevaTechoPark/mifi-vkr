import os
import pandas as pd
from .config import TEXT_COLUMN, LABEL_COLUMN, SEPARATOR

def load_dataset(path: str) -> pd.DataFrame: ...
def save_dataset(df: pd.DataFrame, path: str) -> None: ...
def build_summarized_dataset(df, summaries, text_column=TEXT_COLUMN, ...) -> pd.DataFrame: ...
def build_combined_dataset(df, summaries, text_column=TEXT_COLUMN, separator=SEPARATOR, ...) -> pd.DataFrame: ...

def derive_output_paths(input_path: str, output_dir: str) -> tuple[str, str]:
    # "train_augmented.csv" → "train_augmented_summarized.csv"
    #                       → "train_augmented_original_plus_summary.csv"
    stem, ext = os.path.splitext(os.path.basename(input_path))
    return (
        os.path.join(output_dir, f"{stem}_summarized{ext}"),
        os.path.join(output_dir, f"{stem}_original_plus_summary{ext}"),
    )