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

MODEL_PATH = Path("models") / "baseline_tfidf_wordchar_linearsvc.joblib"

TRAIN_PATH = "data/train_augmented.csv" # train.csv / train_paraphrase.csv / train_backtranslate.csv / train_augmented.csv
TEST_PATH  = "data/test.csv"
TEXT_COL = "text"
LABEL_COL = "label"

print("train.csv", pd.read_csv("data/train.csv").shape)
print("train_paraphrase.csv", pd.read_csv("data/train_paraphrase.csv").shape)
print("train_backtranslate.csv", pd.read_csv("data/train_backtranslate.csv").shape)
print("train_augmented.csv", pd.read_csv("data/train_augmented.csv").shape)
df_orig  = pd.read_csv("data/train.csv")
df_para  = pd.read_csv("data/train_paraphrase.csv")
df_bt    = pd.read_csv("data/train_backtranslate.csv")
df_aug   = pd.read_csv("data/train_augmented.csv")
print("=== train.csv ===")
print(df_orig["label"].value_counts().sort_index())
print("\n=== train_paraphrase.csv ===")
print(df_para["label"].value_counts().sort_index())
print("\n=== train_backtranslate.csv ===")
print(df_bt["label"].value_counts().sort_index())
print("\n=== train_augmented.csv ===")
print(df_aug["label"].value_counts().sort_index())

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
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
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


# baseline train_augmentation.csv balanced_accuracy: 0.444, macro_f1: 0.447
# rubert_tiny2 train_augmentation.csv  balanced_accuracy: 0.116, macro_f1: 0.103

# baseline train.csv balanced_accuracy: 0.443, macro_f1: 0.44
# baseline train_paraphrase.csv balanced_accuracy: 0.462, macro_f1: 0.466
# baseline train_backtranslate.csv balanced_accuracy: 0.463, macro_f1: 0.472
# baseline train_augmented.csv balanced_accuracy: 0.463, macro_f1: 0.472

# rubert-base-case train.csv  balanced_accuracy: 0.387, macro_f1: 0.387
# rubert-base-case train_augmentation.csv  balanced_accuracy: 0.404, macro_f1: 0.4
# rubert-base-case train_paraphrase.csv balanced_accuracy 0.429, macro_f1: 0.424
# rubert-base-case train_backtranslate.csv balanced_accuracy: 0.431, macro_f1: 0.423
# rubert-base-cased train_augmentated.csv balanced_accuracy: 0.648, macro_f1: 0.641