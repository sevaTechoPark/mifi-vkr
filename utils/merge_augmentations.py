import os
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

train_path      = data_dir / "train.csv"
para_path       = data_dir / "train_paraphrase_3.csv"
bt_path         = data_dir / "train_backtranslate_3.csv"

def merge_augmentations():
    df_orig  = pd.read_csv(train_path)
    df_para  = pd.read_csv(para_path)
    df_bt    = pd.read_csv(bt_path)
    
    print("shapes:", df_orig.shape, df_para.shape, df_bt.shape)
    
    cols = ["label", "text"]
    
    # 1) базовый набор — только оригиналы
    df_base = df_orig[cols].copy()

    # 2) убираем из paraphrase все тексты, которые уже есть в оригинале
    df_para_only = df_para[cols].merge(
        df_base[cols].drop_duplicates(),
        on=["label", "text"],
        how="left",
        indicator=True
    )
    df_para_only = df_para_only[df_para_only["_merge"] == "left_only"][cols]
    
    # 3) аналогично для back-translation
    df_bt_only = df_bt[cols].merge(
        df_base[cols].drop_duplicates(),
        on=["label", "text"],
        how="left",
        indicator=True
    )
    df_bt_only = df_bt_only[df_bt_only["_merge"] == "left_only"][cols]
    
    print("only para:", df_para_only.shape, "only bt:", df_bt_only.shape)
    
    # 4) объединяем: оригиналы + чистые аугментации
    df_all = pd.concat(
        [
            df_base,
            df_para_only,
            df_bt_only,
        ],
        ignore_index=True
    )
    
    # на всякий случай уберём возможные дубликаты по (label, text)
    df_all = df_all.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    
    # перемешаем
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    df_all.to_csv(data_dir / "train_augmented_3.csv", index=False)
    print(f"Итого строк в train_augmented.csv: {len(df_all)}")

def main():
    merge_augmentations()

if __name__ == "__main__":
    main()
