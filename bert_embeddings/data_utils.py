from pathlib import Path
import pandas as pd


def load_text_csv(path: str, text_col: str = "text") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    return df


def build_mlm_corpus(train_path: str, test_path: str, text_col: str = "text") -> pd.DataFrame:
    train_df = load_text_csv(train_path, text_col=text_col)
    test_df = load_text_csv(test_path, text_col=text_col)
    raw_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    raw_df = raw_df[[text_col]].dropna().copy()
    raw_df[text_col] = raw_df[text_col].astype(str).str.strip()
    raw_df = raw_df[raw_df[text_col] != ""].reset_index(drop=True)
    return raw_df


def save_texts_parquet(df: pd.DataFrame, out_path: str):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)