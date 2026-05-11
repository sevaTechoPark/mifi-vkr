import os
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

train_path = data_dir / "train.csv"
para_path  = data_dir / "train_paraphrase_3.csv"
bt_path    = data_dir / "train_backtranslate_3.csv"


def normalize_text(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


def merge_augmentations():
    df_orig = pd.read_csv(train_path)
    df_para = pd.read_csv(para_path)
    df_bt   = pd.read_csv(bt_path)

    print("shapes:", df_orig.shape, df_para.shape, df_bt.shape)

    cols = ["label", "text"]

    # 1) базовый набор — только оригиналы
    df_base = df_orig[cols].copy()

    # нормализованный ключ для сравнения (не меняем исходные данные)
    def add_key(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_key"] = df["label"].astype(str) + "|||" + normalize_text(df["text"])
        return df

    base_keys = set(add_key(df_base)["_key"])

    # 2) из para оставляем только строки, которых нет в оригинале
    df_para_keyed = add_key(df_para[cols])
    df_para_only  = df_para_keyed[~df_para_keyed["_key"].isin(base_keys)][cols].copy()

    # 3) из bt оставляем только строки, которых нет ни в оригинале, ни в para
    para_keys      = set(add_key(df_para_only)["_key"])
    seen_keys      = base_keys | para_keys
    df_bt_keyed    = add_key(df_bt[cols])
    df_bt_only     = df_bt_keyed[~df_bt_keyed["_key"].isin(seen_keys)][cols].copy()

    print("only para:", df_para_only.shape, "only bt:", df_bt_only.shape)

    # 4) объединяем: оригиналы + чистые аугментации
    df_all = pd.concat(
        [df_base, df_para_only, df_bt_only],
        ignore_index=True
    )

    # финальная страховочная дедупликация (по нормализованному ключу)
    df_all = add_key(df_all)
    df_all = df_all.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    df_all = df_all.reset_index(drop=True)

    # перемешиваем
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_path = data_dir / "train_augmented_3.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Итого строк в train_augmented_3.csv: {len(df_all)}")
    print(f"Сохранено в: {out_path}")


def main():
    merge_augmentations()


if __name__ == "__main__":
    main()