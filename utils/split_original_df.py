from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


label_col = "label"
test_size = 0.2
random_state = 42
min_count_for_strat = 5


def split_original_df(
    df: pd.DataFrame,
    output_dir: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {label_col, text_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df[[label_col, text_col]].copy().reset_index(drop=True)
    print(f"Shape: df={df.shape}")
    print(f"Unique labels: df={df[label_col].nunique()}")

    target_test_n = int(round(len(df) * test_size))

    test_minimal_df = df.groupby(label_col, group_keys=False).sample(
        n=1,
        random_state=random_state,
    )
    remaining_df = df.drop(index=test_minimal_df.index)

    remaining_test_n = target_test_n - len(test_minimal_df)
    if remaining_test_n < 0:
        raise ValueError(
            "Target test size is too small: fewer test samples requested than number of labels. "
            "Increase dataset size or test_size."
        )

    remaining_vc = remaining_df[label_col].value_counts()
    ok_labels = remaining_vc[remaining_vc >= 2].index
    remaining_ok_df = remaining_df[remaining_df[label_col].isin(ok_labels)]
    remaining_bad_df = remaining_df[~remaining_df[label_col].isin(ok_labels)]
    train_remaining_df = remaining_bad_df

    if remaining_test_n == 0:
        to_train = remaining_ok_df
        to_test = remaining_ok_df.iloc[0:0].copy()
    else:
        to_train, to_test = train_test_split(
            remaining_ok_df,
            test_size=remaining_test_n,
            random_state=random_state,
            stratify=remaining_ok_df[label_col],
        )

    train_df = pd.concat([to_train, train_remaining_df], ignore_index=True)
    test_df = pd.concat([to_test, test_minimal_df], ignore_index=True)

    save_train_test(train_df, test_df, output_dir=output_dir)
    return train_df, test_df


def save_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"saved stratified train/test split to {output_dir}")
    print(f"Shapes: train={train_df.shape}, test={test_df.shape}")
    print(
        "Unique labels: "
        f"train={train_df['label'].nunique()}, "
        f"test={test_df['label'].nunique()}"
    )


def main() -> None:
    raise RuntimeError("Use split_original_df(df=..., output_dir=...) from main.py")


if __name__ == "__main__":
    main()