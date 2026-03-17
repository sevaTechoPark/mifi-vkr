import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
import joblib
from pathlib import Path

from utils.get_baseline_model import get_baseline_model

MODEL_PATH = Path("models") / "baseline_tfidf_wordchar_linearsvc.joblib"

TRAIN_PATH = "data/train_augmentation.csv" # train_augmentation.csv / train.csv
TEST_PATH  = "data/test.csv"
TEXT_COL = "text"
LABEL_COL = "label"

def make_base_pipeline():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)$$[A-Z_]+$$|\b\w\w+\b",
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
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "micro_f1": f1_score(y_test, pred, average="micro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, pred),
    }
    return {k: round(v, ndigits) for k, v in metrics.items()}

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df[TEXT_COL].astype(str)
    y_train = train_df[LABEL_COL].astype(str)
    X_test  = test_df[TEXT_COL].astype(str)
    y_test  = test_df[LABEL_COL].astype(str)

    model = make_base_pipeline()
    model.fit(X_train, y_train)

    metrics = eval_model(model, X_test, y_test)
    print("BASELINE METRICS:", metrics)

    joblib.dump(model, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)

if __name__ == "__main__":
    main()
