import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils.get_original_df import get_processed_df

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
    
label_col="label"
test_size=0.2
random_state=42
min_count_for_strat=5

def split_original_df():
    """
    1) Классы с count >= min_count_for_strat: обычный stratified train_test_split.
    2) Классы с count <  min_count_for_strat: вручную кладём 1 пример в test, остальное в train.
       (предполагается, что в каждом таком классе >= 2 примера)
    """
    df = get_processed_df().reset_index(drop=True)
    print(f"Shape: df={df.shape}")
    print(f"Unique labels: df={df['label'].nunique()}")

    target_test_n = int(round(len(df) * test_size))
     # 1) по 1 в test из каждого класса
    test_minimal_df = df.groupby("label").sample(n=1, random_state=random_state)    
    remaining_df = df.drop(index=test_minimal_df.index)

    remaining_test_n = target_test_n - len(test_minimal_df)

    # 2) stratify можно только по классам, где в remaining осталось >=2
    remaining_vc = remaining_df[label_col].value_counts()
    ok_labels = remaining_vc[remaining_vc >= 2].index
    remaining_ok_df = remaining_df[remaining_df[label_col].isin(ok_labels)]
    # тут классы, где остался 1 пример
    remaining_bad_df = remaining_df[~remaining_df[label_col].isin(ok_labels)]
    # смысловое переименование, эта констуркция одновременно и remaining_bad_df, и train_remaining_df в зависимости от контекста
    train_remaining_df = remaining_bad_df

    # 3) добираем remaining_test_n примеров в test стратифицированно из remaining_ok_df
    to_train, to_test = train_test_split(
        remaining_ok_df,
        test_size=remaining_test_n,
        random_state=random_state,
        stratify=remaining_ok_df[label_col],
    )

    train_df = pd.concat([to_train, train_remaining_df], ignore_index=True)
    test_df = pd.concat([to_test, test_minimal_df], ignore_index=True)

    save_train_test(train_df, test_df)
    return train_df, test_df

def save_train_test(train_df, test_df):
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    print(f'save stratify traint_test_split to {data_dir}')
    print(f"Shapes: train={train_df.shape}, test={test_df.shape}")
    print(
        "Unique labels: "
        f"train={train_df['label'].nunique()}, "
        f"test={test_df['label'].nunique()}"
    )

def main():
    split_original_df()

if __name__ == "__main__":
    main()