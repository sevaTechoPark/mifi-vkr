import argparse
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)

TEXT_COL = "text"
LABEL_COL = "label"

def make_base_pipeline():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\[[A-Z_]+\]|\b\w\w+\b",
        lowercase=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=True,
    )
    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    clf = LinearSVC(class_weight="balanced")

    return Pipeline([
        ("tfidf", features),
        ("clf", clf),
    ])


def eval_model(model, X_test, y_test, ndigits=3):
    pred = model.predict(X_test)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
    }
    return {k: round(v, ndigits) for k, v in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline text classification")
    parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Path to training dataset CSV",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
        help="Path to test dataset CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = args.train_path.expanduser()
    test_path = args.test_path.expanduser()

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[TEXT_COL].astype(str)
    y_train = train_df[LABEL_COL].astype(str)
    X_test = test_df[TEXT_COL].astype(str)
    y_test = test_df[LABEL_COL].astype(str)

    model = make_base_pipeline()
    model.fit(X_train, y_train)

    metrics = eval_model(model, X_test, y_test)
    print("BASELINE METRICS:", metrics)


if __name__ == "__main__":
    main()