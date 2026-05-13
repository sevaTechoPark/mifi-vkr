import argparse
from pathlib import Path

import pandas as pd

from utils.clean_original_df import make_df_clean
from utils.split_original_df import split_original_df


def read_original_df(original_dataset: str | Path) -> pd.DataFrame:
    path = Path(original_dataset)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_json(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-dataset",
        type=Path,
        required=True,
        help="Path to original_data.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory where cleaned_df.csv, train.csv and test.csv will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_dataset = args.original_dataset
    data_dir = args.data_dir

    df = read_original_df(original_dataset)
    cleaned_df = make_df_clean(
        df=df,
        output_dir=data_dir,
        text_col="text",
        label_col="label",
    )
    split_original_df(
        df=cleaned_df,
        output_dir=data_dir,
        text_col="text",
        label_col="label",
    )


if __name__ == "__main__":
    main()