import argparse
from pathlib import Path

import pandas as pd


def normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_key"] = df["label"].astype(str) + "|||" + normalize_text(df["text"])
    return df


def print_label_counts(df: pd.DataFrame, title: str) -> None:
    print(title)
    label_counts = df["label"].value_counts(dropna=False).sort_index()
    for label, count in label_counts.items():
        print(f"label={label} | count={int(count)}")


def merge_augmentations(
    para_file: str | Path,
    bt_file: str | Path,
    output_dir: str | Path,
    output_name: str = "train_augmented.csv",
) -> pd.DataFrame:
    para_file = Path(para_file)
    bt_file = Path(bt_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not para_file.exists():
        raise FileNotFoundError(f"Paraphrase file not found: {para_file}")
    if not bt_file.exists():
        raise FileNotFoundError(f"Backtranslate file not found: {bt_file}")

    df_orig = pd.read_csv(train_file)
    df_para = pd.read_csv(para_file)
    df_bt = pd.read_csv(bt_file)

    cols = ["label", "text"]
    for name, df in {
        "train.csv": df_orig,
        "para-file": df_para,
        "bt-file": df_bt,
    }.items():
        missing_cols = set(cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} is missing columns: {sorted(missing_cols)}")

    print(f"Shape train.csv: {df_orig.shape}")
    print(f"Shape para-file: {df_para.shape}")
    print(f"Shape bt-file: {df_bt.shape}")

    df_base = df_orig[cols].copy()

    base_keys = set(add_key(df_base)["_key"])

    df_para_keyed = add_key(df_para[cols])
    df_para_only = df_para_keyed[~df_para_keyed["_key"].isin(base_keys)][cols].copy()

    para_keys = set(add_key(df_para_only)["_key"])
    seen_keys = base_keys | para_keys

    df_bt_keyed = add_key(df_bt[cols])
    df_bt_only = df_bt_keyed[~df_bt_keyed["_key"].isin(seen_keys)][cols].copy()

    df_all = pd.concat(
        [df_base, df_para_only, df_bt_only],
        ignore_index=True,
    )

    df_all = add_key(df_all)
    df_all = df_all.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    df_all = df_all.reset_index(drop=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_path = output_dir / output_name
    df_all.to_csv(out_path, index=False)

    print(f"Shape train_augmented.csv: {df_all.shape}")
    print_label_counts(df_all, "Label counts in train_augmented.csv:")
    print(f"Saved to: {out_path}")

    return df_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge train, paraphrase and backtranslate datasets")
    parser.add_argument(
        "--para-file",
        type=Path,
        required=True,
        help="Path to train_paraphrase.csv",
    )
    parser.add_argument(
        "--bt-file",
        type=Path,
        required=True,
        help="Path to train_backtranslate.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory with train.csv and where train_augmented.csv will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_augmentations(
        para_file=args.para_file,
        bt_file=args.bt_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()