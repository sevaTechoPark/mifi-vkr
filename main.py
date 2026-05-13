import argparse
from pathlib import Path

import pandas as pd

from utils.clean_original_df import make_df_clean
from utils.split_original_df import split_original_df


def read_original_df(data_dir: str | Path, filename: str = "original_data.json") -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_json(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with original_data.json and output CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    df = read_original_df(data_dir)
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

 # python main.py --data-dir=/Users/v.papadyk/ml/mifi-vkr/data   